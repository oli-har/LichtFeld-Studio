# Open upstream bugs: fixes and plan

This covers bug reports from the newest 100 of the 210 open issues on
[MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio/issues) as of
2026-10-31. Most of the older issues are feature requests and roadmap epics. The
Windows test reports (#3944–#3948) are in [windows_test_failures.md](windows_test_failures.md).

Everything here comes from reading upstream `dev` (`809251714`) on Linux, without a GPU,
so nothing has been run. The confidence marks are the same as in the Windows plan:

- **Confirmed**: the code clearly does this.
- **Likely**: it fits the evidence.
- **Open**: needs a run.

| Issue | Summary | State |
| --- | --- | --- |
| [#3876](https://github.com/MrNeRF/LichtFeld-Studio/issues/3876) | RAD import runs out of VRAM before paging is chosen | Plan, cause confirmed |
| [#3604](https://github.com/MrNeRF/LichtFeld-Studio/issues/3604) | `--use-depth-loss` training hangs at iteration 906 (JPEG 2000 sidecar cache) | Workaround switch added, plan |
| [#2570](https://github.com/MrNeRF/LichtFeld-Studio/issues/2570) | SSOG export of ~100M splats runs out of VRAM on 24 GB | Plan, cause confirmed |
| [#2783](https://github.com/MrNeRF/LichtFeld-Studio/issues/2783) | CVT1100 duplicate VERSION resource | Already fixed by #3198; can be closed |
| [#1569](https://github.com/MrNeRF/LichtFeld-Studio/issues/1569) | Source builds have no uv, so plugin installs fail | Partly fixed, plan |
| [#3205](https://github.com/MrNeRF/LichtFeld-Studio/issues/3205) | Windows build instructions | Partly fixed, plan |
| [#3940](https://github.com/MrNeRF/LichtFeld-Studio/issues/3940) | `--enable-mip` ignored with `--gut` | `dev` already refuses the combination; a related timelapse bug fixed; plan for mip in 3DGUT |
| [#2135](https://github.com/MrNeRF/LichtFeld-Studio/issues/2135) | Mesh to Splat changes orientation | No code difference between CLI and GUI; plan for an `--up-axis` option |
| [#1651](https://github.com/MrNeRF/LichtFeld-Studio/issues/1651) | Save steps ignored | One cause fixed, one planned |
| [#1756](https://github.com/MrNeRF/LichtFeld-Studio/issues/1756) | 0.5.3 checkpoint can't be resumed | Plan, cause confirmed; needs a maintainer decision |
| [#1786](https://github.com/MrNeRF/LichtFeld-Studio/issues/1786) | MCP bind failure invisible | Already visible since #1658; plan for the identity part |

## #3876: RAD import runs out of VRAM before paging is chosen

**Cause (confirmed; the numbers match the log exactly).** Three things combine:

- `RadLoader::load` (`src/io/loaders/rad_loader.cpp`) calls `load_rad` first and checks
  `radPagedGpuUploadRequested` only afterwards.
- `load_rad` decodes into CPU tensors with SH in canonical layout, then constructs
  `SplatData` with the default `ShNLayout::Canonical`. That constructor
  (`src/core/splat_data.cpp`, `allocate_swizzled_shN` and
  `reorder_canonical_into_swizzled`) allocates the swizzled SH on the GPU (48 B/node at
  SH1) and uploads the canonical tensor as a temporary (36 B/node).
- The out-of-core decision in `rad.cpp` compares only against host RAM. With 128 GB of
  RAM it picks in-core.

For 319.6M nodes that is 14.3 GiB + 10.7 GiB. The failed 10972.8 MiB allocation is the
canonical temporary.

**A related problem (Likely):** `migrateSplatTensorsToAllocator` (`loader_service.cpp`)
computes the paged decision a second time, after the upload. Free VRAM is lower by then,
so the answer can flip, and the model then stays in plain CUDA tensors.

**Plan, smallest first:**

1. **No GPU allocation inside `load_rad`.**
   - Build the swizzled SH on the host and construct with `ShNLayout::Swizzled`, which
     adopts the tensor.
   - Either convert the whole canonical tensor with `sh_codec`'s CPU path, or swizzle per
     chunk the way the PLY loader does. In the per-chunk version, do the index math in
     64 bits: the `uint32` helper overflows at about 358M rows at SH3.
   - Make the host estimate use swizzled bytes per node.
2. **Decide paging once, with a VRAM estimate.**
   - Size the model as nodes × (56 + swizzled SH bytes). The non-paged route peaks at
     about twice that, while the CUDA copy and the renderer copy both exist.
   - Decide right after `read_rad_file_info` and store the result on `SplatLodTree`.
   - `RadLoader::load` and `pagedRadGpuResidencyRequested` read the stored flag instead
     of querying free VRAM again.
3. **Lower the non-paged peak.**
   - With a renderer allocator, copy host → renderer storage directly.
   - Catch `MemoryAllocationError` in `RadLoader::load` and retry paged.

**Verify:**

- **Existing tests:** `lichtfeld_tests --gtest_filter=PlyToRadLod.*:RadLodLeaves.*`.
- **A new test:** `alloc_counter::delta_since(snapshot) == 0` across `load_rad`, and
  `shN_raw().device() == Device::CPU`. Its SH must be bit-identical to the old path,
  compared through `shN_canonical()`.
- **The real file:**
  `LFS_RAD_VALIDATE_FILE=<file.rad> lichtfeld_tests --gtest_filter=PlyToRadLod.ValidateExternalRad`
  should print `paged=1`.
- **In the app on a 24 GB card:**
  - It should log `RAD paged LOD active: deferring full GPU tensor migration`.
  - There should be no "Failed to stage splat file".
  - `nvidia-smi` should stay well under 24 GB.

## #3604: `--use-depth-loss` training hangs at iteration 906

**What's on the branch.** Setting the environment variable `LFS_JPEG2K_CACHE=0` turns the
JPEG 2000 cache off (`PipelinedImageLoader`'s constructor). Depth and normal sidecars are
then decoded from their sources every epoch. That is the state the reporter reached by
hiding the nvJPEG2000 extension, and that run trained to 30k.

**Cause (Likely: a hang inside nvJPEG2000's high-throughput decode; the exact trigger is
Open).** There is no switch other than this one; the flag was only cleared when an encode
failed. Iteration 906 is special in three ways, and the run without the extension avoids
all three:

- It is the first JPEG 2000 decode of the run. Epoch 1 only encodes, and 8-bit JPEGs
  never touch nvJPEG2000.
- It is where the nvJPEG2000 decoder object is first created; decoders are created
  lazily.
- It overlaps cold-worker encodes on the same shared `NvCodecImageLoader` instance.

**Ruled out by reading:** an LFS lock cycle or event-ordering deadlock. No mutex is held
across the decode wait, and every stream wait refers to work already enqueued. The
symptoms (GPU at 100 % but low power, two busy cores) fit a GPU kernel that never
finishes, with two threads spin-waiting on it.

**Also latent:** the hot thread acquires the whole decoded-frame ring batch before it
decodes that batch's sidecars. Decoding the sidecars first removes that dependency.

**Plan:**

1. **Make the switch a real option.** Add a config field and a CLI flag. Ideally make it
   sidecar-only, so 16-bit RGB caching isn't affected.
2. **Cache depth uncompressed by default.** The raw path already exists
   (`SidecarCacheFormat::DepthFloat`), or add a 16-bit raw variant. It is bit-exact,
   decodes as a plain copy and takes nvJPEG2000 out of the per-iteration path, at the
   cost of RAM.
3. **A watchdog.**
   - Record sequence, sidecar path and image path with each hot decode's fence.
   - Poll the fences with `cudaEventQuery`.
   - After N seconds, log the camera, the path and the switch, and stop with a clear
     error. Recovering from a hung kernel isn't realistic.
4. **Pin the trigger.** Serialize JPEG 2000 encode and decode inside `NvCodecImageLoader`,
   or warm the decoder with a tiny decode at loader start. If either makes the hang go
   away, keep it.

**Verify:**

- **Tests:**
  `timeout 600 lichtfeld_tests --gtest_filter=NvCodecImageLoaderJpeg2k16Bit.*:PipelinedImageLoaderSidecarJpeg2k.*:PipelinedDepthDecode.*`.
  If `RoundTrips2160pGrayAndRgbLossless` hangs, it is nvJPEG2000 alone. If only
  `DepthFirstTouchTranscodesThenHotDecodes` hangs, it is the loader pipeline.
- **The reporter's case:**
  `LichtFeld-Studio --headless -d <data> --strategy mrnf --gut --use-depth-loss -i 3000 --log-level debug`,
  with and without `LFS_JPEG2K_CACHE=0`.
- **On a hang:**
  - `procdump -ma`, then `~*k` in WinDbg.
  - `nsys profile --trace=cuda,nvtx` should show the stuck kernel under the
    `nvjpeg2kDecodeImage` range.

## #2570: SSOG export of ~100M splats runs out of VRAM

**Cause (confirmed).** `save_ssog` → `decimate_splats` → `gpu_candidates`
(`src/core/tensor/backend/cuda/kernels/export_decimate.cu`) allocates every per-splat
buffer at full N, and all of them are live at the `candidates` launch. There is no memory
check first. The reporter's estimate is right:

| Buffer | Bytes per splat |
| --- | --- |
| Inputs | 56 |
| Morton keys and order | 12 |
| Sort scratch | ~12 |
| BVH boxes | ~8 |
| 16-neighbour list | 64 |
| Gaussian precompute cache | 136 |
| 4 candidates (index and cost) | 32 |

That is about 308 B/splat, or 32 GB at 105M. Export can also clone the merged model
first. The Vulkan, HIP and Metal paths have the same shape (`export_pipeline.cpp`).

**Plan:**

0. **Check memory before each GPU pass.**
   - Needed: n × 244 B, plus the boxes and the sort scratch.
   - Available: free VRAM plus the pool's reserved-but-unused memory.
   - If it doesn't fit, return `RESOURCE_EXHAUSTED` with both numbers instead of a raw
     CUDA error.
1. **Cut memory without changing the results.**
   - Drop the 136 B cache and compute it inside the `candidates` kernel. It is a
     deterministic function of the inputs.
   - Free the keys and boxes after the neighbour search.
   - Run the neighbour search and the costing in Morton-ordered chunks, scattering
     results by `order`.

   That brings 105M from about 32 GB to about 8–9 GB.
2. **What must stay global:** the kNN index, the greedy `select()` (already on the host)
   and the merge's prefix sums. Neighbour lists, precompute, pair costs and the merge
   math can be done per chunk.
3. **A CPU fallback.** Parallelize `cpu_candidates` and `cpu_merge` with TBB and pick
   them automatically when the GPU path doesn't fit.
4. **Only if the inputs alone exceed VRAM:** spatial buckets with a halo of at least the
   16-NN radius, keeping `select()` global.

**Verify:**

- **Tests:** `lichtfeld_tests --gtest_filter=SplatDecimate.*:SsogFormat.*`.
  `GpuMatchesCpuReference` must keep ≥ 99 % candidate parity.
- **Peak memory:**
  `LFS_DECIMATE_PERF=1 LFS_DECIMATE_PERF_COUNT=10000000 lichtfeld_tests --gtest_filter=SplatDecimate.DecimatePerfSmoke`
  while sampling `nvidia-smi --query-gpu=memory.used --format=csv -lms 100`.
- **End to end:** `LichtFeld-Studio convert big.ply out.ssog --lod-levels 4`.

## #2783: CVT1100 duplicate VERSION resource

**Already fixed by #3198 (`e1e6bc647`), which is on `dev`, so the issue can be closed.**

**What happened.** The generated version resource was `LichtFeld-Studio.rc`, which
differs from `resources/lichtfeld-studio.rc` only in case. The Visual Studio generator
compiles every `.rc` to `$(IntDir)%(Filename).res`, and CMake can't rename that, so the
same `.res` was linked twice. Ninja keeps the directory in object names, and CI uses
Ninja, which is why CI never saw it. The generated file is now
`<target>.version.rc`.

**Optional guard.** Because CI never builds with the Visual Studio generator, a rename
could bring this back unnoticed. In `lfs_add_windows_version_resource`
(`cmake/WindowsVersionResource.cmake`), fail at configure time when a target has two
`.rc` files whose names match, ignoring case.

**Verify:** configure with `-G "Visual Studio 17 2022" -A x64` and build `LichtFeld-Studio`.
`build-vs\LichtFeld-Studio.dir\Release` should contain `lichtfeld-studio.res` and
`LichtFeld-Studio.version.res`.

## #1569: source builds have no uv, so plugin installs fail

**Three causes, all confirmed.**

1. **New, and fixed on the branch.** With a multi-config generator (Visual Studio, which
   CMake picks when `-G` is left out), uv was copied to `build\bin`, but the exe is in
   `build\<Config>` and looks for `<exe dir>\bin\uv.exe`. So "uv not found" happened even
   when the download worked. `copy_uv_to_build` (`cmake/FetchUV.cmake`) now also copies
   to `build\<Config>\bin`.
2. **A failed download or extraction is only a warning** unless `BUILD_PORTABLE` is set.
3. **There is no PATH fallback, on purpose.** It was removed in `a03841e2d`, and
   `tests/python/test_plugin_system.py` asserts it.

**Plan:**

- **Surface the failure at configure time.** Add an end-of-configure summary line
  (`Bundled uv: 0.13.5` or `MISSING`). Optionally make it fatal by default, with an
  `LFS_REQUIRE_UV=OFF` escape hatch. Point out that a cached archive at
  `UV_ARCHIVE_PATH` already works offline.
- **An actionable runtime message.**
  - Add `PackageManager::uv_missing_message()`, naming the paths searched and how to fix
    it.
  - Use it in `execute_uv`, `plugin_runner.cpp`, `uv_runner.cpp`,
    `lfs/py_packages.cpp` and `installer.py`. Keep the "uv not found" prefix that
    `test_python_integration.cpp` matches.
- **The maintainers' call: a PATH fallback.** If it's wanted, limit it to non-portable
  builds, require `uv --version` ≥ 0.13.4 (the CA security floor), and log the found and
  the pinned version. A more conservative option is an explicit `LFS_UV` environment
  override.
- **Side findings in `FetchUV.cmake`:**
  - The macOS arm64 checksum is still keyed on 0.10.2, so 0.13.5 is downloaded there
    without verification.
  - A `-DUV_VERSION` override is downloaded without verification.
  - `install_uv()` copies any `uv` from PATH with no version floor.

**Verify:** configure with `-G "Visual Studio 17 2022" -A x64`; `build-vs\Release\bin\uv.exe`
should now exist. Then `LichtFeld-Studio.exe plugin create uvprobe` should get past the uv
step.

## #3205: Windows build instructions

**What's on the branch.** Without `VCPKG_ROOT` or `CMAKE_TOOLCHAIN_FILE`, configure used
to fail with "Could not find toolchain file: /scripts/buildsystems/vcpkg.cmake" (the same
error appears in #2783). It now stops with a message saying to set `VCPKG_ROOT` or pass
the toolchain file, and checks that `VCPKG_ROOT` really holds vcpkg (top of
`CMakeLists.txt`).

**The reported errors (confirmed):**

- **(a) An old CMake was first on PATH.** CMake 3.7.1 reads `-B build` as the source
  directory, hence "build does not exist", and then fails the 3.30 minimum. The docs
  never say to check `where cmake`.
- **(b) Ninja missing.** Ninja is only in the developer prompt when Visual Studio has
  "C++ CMake tools for Windows", but the docs present Ninja as optional.
- **(b) clang-cl.** vcpkg builds ports in a clean environment, built from the Visual
  Studio installation it selects: the newest stable one, unless
  `VCPKG_VISUAL_STUDIO_PATH` is set.
  - So clang-cl on PATH, or a different prompt, can't help, although the libplacebo
    portfile's message suggests both.
  - The check fails when "C++ Clang Compiler for Windows" is missing from that particular
    installation, for example when it is in VS 2022 but vcpkg picks VS 2026 or the Build
    Tools.

**Plan:**

- **The libplacebo portfile** (`.github/overlays/libplacebo/portfile.cmake`).
  - Reword the error: name the installation vcpkg uses (`VSINSTALLDIR`) and the
    component ID `Microsoft.VisualStudio.Component.VC.Llvm.Clang`, and mention
    `VCPKG_VISUAL_STUDIO_PATH`.
  - Probe that installation before PATH, so clang-cl matches its STL.
  - Optionally, preflight with `vswhere` before `project()`, so users find out before
    vcpkg has built dozens of ports.
- **Docs.** Update `docs/building_and_distribution.md` and `docs/docs/development/build.md`,
  and bring `docs/docs/installation/building/windows.md` back from "Moved to Wiki", or sync
  the Wiki.
  - **Prerequisites:**
    - Visual Studio 2022 17.10+ with "Desktop development with C++", MSVC v143, a
      Windows 11 SDK, "C++ CMake tools for Windows" and "C++ Clang Compiler for Windows".
    - CMake ≥ 3.30 first on PATH, and Ninja.
    - CUDA ≥ 12.8.
    - A bootstrapped vcpkg clone, with `VCPKG_ROOT` pointing at it. Developer prompts can
      point it at Visual Studio's bundled vcpkg, which CI doesn't use.
    - Long paths enabled, and a short root such as `C:\src`.
  - **Prompt:** "x64 Native Tools Command Prompt for VS 2022". The plain Developer
    prompt targets x86.
  - **Steps:** `cmake --preset windows-release` and
    `cmake --build --preset windows-release`. Mention `-G Ninja` wherever plain
    `cmake -B` is shown: without it, Windows gets the Visual Studio generator, which CI
    never builds.
  - **Consistency:** `docs/docs/installation/index.md` says binaries are on GitHub
    Releases and `README.md` says they aren't. `CMakeLists.txt` enforces VS 17.4 while the
    docs say 17.10.
- **A `windows-portable` preset** mirroring the nightly build (`BUILD_PORTABLE=ON`).
- **`build_lichtfeld.ps1`** (no doc references it):
  - It forces the Visual Studio generator. Switch it to the presets.
  - Its CUDA check (`-match "12\."`) rejects 13.x and accepts 12.0–12.7. Require ≥ 12.8.
  - It makes LibTorch mandatory (about 3 GB per config) although the app doesn't use it.
  - It runs `git pull` on vcpkg on every run.
  - It reads `$Generator` before assigning it.

**Verify:**

- In a plain shell without `VCPKG_ROOT`, configure should show the new message.
- In the build prompt, check `where cmake`, `cmake --version` and `where ninja`.
- `vcpkg env --triplet x64-windows "echo %VSINSTALLDIR%"` shows which installation the
  ports use. Check that `<it>\VC\Tools\Llvm\x64\bin\clang-cl.exe` exists.

## #3940: `--enable-mip` ignored with `--gut`

**On `dev` the combination is already refused.** Since #2048 (`06c8c677b`),
`training_backend.hpp` marks the mip filter unsupported for 3DGUT, and `validate()`
returns "3DGUT cannot be used with Mip Filter. Change this setting or select 3DGS."
That covers the CLI, the trainer, the GUI (which also greys out the checkbox) and
checkpoint loads. The issue's line numbers match `master`, which doesn't have the check:
cherry-pick `06c8c677b` there, or add it next to the igs+ conflict in `parameters.cpp`.

**A correction to the issue (confirmed): passing the flag through would change nothing in
the image.**

- In the 3DGUT path, `eps2d` (0.3) and the compensation feed only culling and tile
  extents (`ProjectionUT3DGSFused.cu`). The `compensations` buffer is never read.
- The pixel kernels evaluate the raw 3D Gaussian with raw opacity, forward and backward.
  Vulkan, Metal and the viewer's GUT renderer do the same.
- So `--gut` trains with no screen-space filter at all, not with the 0.3 px² dilation.
- A side effect (Likely): culling uses opacity scaled by the 0.3 compensation while the
  pixels use raw opacity, so some sub-pixel splats are culled that the evaluator would
  draw.

**Fixed on this branch (a related bug on `dev` and `master`).** The 3DGS timelapse rendered
with `mip_filter = false` (`trainer.cpp`), so timelapse frames of `--enable-mip` runs used
the 0.3 px² dilation without compensation. It now passes `step_params.mip_filter`, as the
evaluation render does.

**Plan for mip with 3DGUT:**

1. Rename `antialiased` to `mip_filter` through `GsplatParams`, `gsplat_render` and
   `gsplat_infer`, and pass it at all six 3DGUT call sites:
   - the evaluation render, training step and timelapse in `trainer.cpp`;
   - `trainer_popspa.cpp`;
   - two in `metrics.cpp`.
2. Use `eps2d = mip ? 0.1f : 0.3f`, and compute compensations when mip is on.
3. **Projection.** Inflate the scales by the world-space variance of 0.1 px² at the
   mean's depth: σw² = 0.1·(z/f)², with f chosen per camera model. The compensation is
   sqrt(det(s) / det(s′)).
4. **Pixel forward.** Evaluate with s′ and multiply the opacity by the compensation.
5. **Backward.** Use the same s′ and compensation:
   - multiply the scale gradient by s/s′;
   - multiply the opacity gradient by the compensation;
   - treat the compensation as stop-gradient, as fastgs does.
6. The same in Vulkan, Metal and the viewer's GUT renderer, or the viewer won't match
   training.
7. Mark it supported in `training_backend.hpp`, and update `test_training_parameters.cpp`
   and `training_panel.py`.

**The two fastgs inconsistencies in the issue are real:**

- **(a) The backward compensation threshold.** In `kernels_backward.cuh`, replace the
  `det_raw > min_cov2d_determinant` branch with what the forward pass does:
  ```cpp
  const float det_raw = fmaxf(raw_a * raw_c - raw_b * raw_b, 0.0f);
  opacity_compensation = sqrtf(det_raw * determinant_rcp);
  ```
  The same change applies to Vulkan `fast_projection_backward.slangh` and Metal
  `fast_backward.metal`.
- **(b) `dilation_scale` missing in backward.** This is latent today: only evaluation
  forwards use a scale other than 1, and they have no backward. Either thread it through
  `preprocess_backward_cu`, `backward()`, `backward_raw()` and `CudaFastFrame`, and the
  Vulkan and Metal backward, or assert scale == 1 in `fast_ops_backward`.

**Verify:**

- On `dev`, `LichtFeld-Studio -d <colmap> -o out --gut --enable-mip -i 100 --headless`
  exits with the conflict error.
- Compare a timelapse frame of a 3DGS `--enable-mip` run with an evaluation render at the
  same step.
- Add finite-difference checks for (a) and (b): `MipFilter_Enabled` in
  `tests/test_fastgs_fuzz.cpp`, with a needle-shaped splat whose determinant is between 0
  and 1e-6.

## #2135: Mesh to Splat changes orientation

**There's no code difference between the paths (confirmed).**

- The CLI `mesh2splat` and the GUI use the same loader (the same assimp flags, with node
  transforms baked into both since #3285) and the same converter.
- The converter applies no axis conversion.
- `tests/test_mesh_loader.cpp` checks the CLI chain.

The reporter's last comment says "Case closed".

**What differs (Likely):**

- The GUI result inherits the mesh node's transform. Users often rotate imported meshes,
  because LFS stores data Y-down and flips it for display.
- Dragging a splat out of the GUI applies an up-axis rotation (+Z up for Blender, which
  is 90° about X). The CLI doesn't.
- Builds before 2026-10-11 also dropped glTF/FBX node rotations in both paths.

**Plan:**

- Add `--up-axis {-y,+y,-z,+z}` (default `-y`, which is identity) to `mesh2splat` and
  `convert`. Reuse `SplatDragAxes::matrix()` after moving it to core.
- Document that conversion keeps mesh coordinates.
- The remaining request, starting training from a loaded mesh or splat, is a feature
  request.

**Verify:**

1. Import a glb, convert it in the GUI and export a PLY; run `mesh2splat x.glb -o cli.ply`.
   The two should be equal.
2. Rotate the mesh node, convert and export again: the export differs by exactly that
   rotation.

## #1651: save steps ignored

**How save steps work.**

- Steps come from the training panel, the status-bar markers, or `save_steps` in
  parameter JSON or a `.licht`. There's no CLI flag.
- A step saves when the trainer may save at a step boundary, which needs a bound `.licht`.
  Otherwise it is skipped with only `LOG_DEBUG`.
- With "Keep all checkpoints" off (the default), each checkpoint replaces the previous
  one.

**Two causes, both confirmed:**

1. **Fixed on this branch.**
   - **The bug:** the trainer picked up GUI edits only through the ParameterManager dirty
     flag. That is also the project's "unsaved changes" flag, which every explicit save
     clears. So pausing, adding a step, pressing Ctrl+S and resuming meant the trainer
     never got the step, while the panel and status bar showed it.
   - **The fix:** the trainer now also tracks `dirtySerial()`, so an edit is applied even
     after a save has cleared the flag (`training_manager.cpp`, `training_manager.hpp`).
     The flag is still consumed as before, so save prompts don't change.
2. **Plan.** `--resume project.licht` (headless) takes its parameters from the
   checkpoint, and re-reads only `keep_all_checkpoints` from the saved project
   (`training_setup.cpp`). So save steps edited after the last checkpoint are ignored.
   Add `ParametersChapter::save_steps(strategy)` next to `checkpoint_retention` and apply
   it unless the command line overrides it.

Also, log skipped step saves at INFO instead of DEBUG.

**Verify:**

1. Turn on "Keep all checkpoints" and start training.
2. Pause around iteration 2000, add a step at 3000, press Ctrl+S and resume.
3. After 3000, `LichtFeld-Studio licht p.licht --checkpoints` should list 3000.

## #1756: a 0.5.3 checkpoint can't be resumed

**Cause (confirmed): it always fails for 0.5.3 checkpoints.**

- 0.5.3 wrote Adam state version 2: uint8 moments with per-row scales.
- That codec was removed in #1588 (`9169a2a00`), and `AdamOptimizer::deserialize` throws
  "AdamOptimizer checkpoint uses removed legacy moment codec; retrain or convert to joint
  (u,log_s) checkpoints".
- By then the scene manager has already cleared the scene and loaded the dataset, hence
  "the dataset loads, then an error".
- Model-only import (`convert x.resume`, `CheckpointLoader`) still works. That is what
  "can still be imported" means.

**Also (Likely):** 0.5.3 parameter JSON has no `use_exposure_correction`, and `from_json`
starts from the MRNF default, which is true. So 0.5.3 MRNF checkpoints silently get
exposure correction, and ones with bilateral grid or PPISP fail validation.

**Plan; the maintainers decide which:**

- **Refuse early**, in the resume popup and before clearing the scene, with a clear
  message, and offer to import the model instead.
- **Or resume with fresh Adam moments:** for state version ≤ 2, read and discard the
  state (which keeps the stream aligned) and log a warning.
- **Or convert:** decode the old moments and re-encode them with the joint codec.

Whichever is chosen:

- Default a missing `use_exposure_correction` to false in `parse_checkpoint_params_json`.
- Show the reason in the popup when reading the header or the parameters fails. Today
  `read_checkpoint_params` drops the error, and the popup just doesn't appear.

**Verify:** build 0.5.3 (`d8c50c6a3`), train about 1k iterations and save a checkpoint. On
`dev`, drag it in, or run `--resume old.resume -d <dataset> --headless`: today this gives
the Adam error.

## #1786: MCP bind failure invisible

**The reported part is already handled.**

- Since #1658, a bind failure sets the MCP status to Failed.
- The status bar then shows a red "MCP Error" ("could not listen on {addr}:{port}"), and
  Preferences shows "Error".
- `--mcp-port` came with #1797.
- `docs/docs/development/mcp/connecting-clients.md` covers running two instances and
  checking `serverInfo.version`.

**What's left is the identity request.**

- Add `_meta` to the `initialize` result, with `app.lichtfeld/pid`, `config_dir`, `port`
  and the current project. Cache the project on open and save rather than querying it
  from the HTTP thread.
- Show the same PID in the status-bar MCP popup, and document the check.

**Speculative, macOS only:** with `SO_REUSEADDR`, a second instance listening on `0.0.0.0`
may coexist with one on `127.0.0.1`, showing a healthy chip while loopback clients reach
the first instance. Check this on a Mac.

**Verify:**

- Start two instances on the same port: the second shows the red chip.
- After the change, `initialize` on 127.0.0.1:45677 returns the first instance's PID.
