# Windows test failures: fixes and plan

This covers the Windows `lichtfeld_tests` reports filed against dev `ab623c39d`:

- [#3944](https://github.com/MrNeRF/LichtFeld-Studio/issues/3944): three test programs don't build
- [#3945](https://github.com/MrNeRF/LichtFeld-Studio/issues/3945): node tests keep a `Node&` across `add_node`
- [#3946](https://github.com/MrNeRF/LichtFeld-Studio/issues/3946): `PinnedBlockReusedOnlyAfterAllStreamsDone` hangs
- [#3947](https://github.com/MrNeRF/LichtFeld-Studio/issues/3947): CUDA connected components read past the union-find array
- [#3948](https://github.com/MrNeRF/LichtFeld-Studio/issues/3948): the other failing tests, grouped by cause

Other open upstream bugs are planned in [open_bug_plan.md](open_bug_plan.md).

The branch `claude/vibrant-lovelace-747k7o` is upstream `dev` (`809251714`) plus these fixes.
Everything was worked out by reading the code on Linux, without MSVC, CUDA or a GPU, so
**nothing here has been compiled or run on Windows yet**. The only exception is the
`UsdHalfTest` reference converter, which was checked numerically on Linux.

Each cause below is marked:

- **Confirmed**: the code clearly does this.
- **Likely**: it fits the evidence but depends on Windows behaviour that couldn't be observed here.
- **Open**: needs a run on the Windows machine.

## Status

| Issue | Tests | State |
| --- | --- | --- |
| #3944 | 3 programs | Fixed: exports, a `far` macro clash and link libraries |
| #3945 | 5 node tests, plus 1 more found | Fixed in the tests |
| #3946 | 1 | Fixed in the test |
| #3947 | 8 node and tensor tests on CUDA | Fixed in the CUDA launch; one ordering question left to check |
| #3948 | Open files, RAD sidecar, atomic writes (12) | Fixed: 8 test changes, plus 2 product fixes (RAD sidecar, atomic commit retry) |
| #3948 | Paths, Unicode, case (7) | Fixed in the tests |
| #3948 | `UsdHalfTest`, `LFS_HOME`, scene source path, camera node | Fixed (scene source path is a product fix) |
| #3948 | `ViewerGpuMemory`, `SessionBreadcrumb`, `TensorReserveInplaceCat`, `LpipsOps` | Fixed (LPIPS by naming the operation in its messages) |
| #3948 | `HostMetrics` | Plan: Windows never samples CPU usage (product gap) |
| #3948 | `NvCodecSentinelValidator`, `VideoFrameExtractorTrim`, `VideoExportUtils` | Plan: one experiment each decides the fix |
| #3948 | Camera validation (5) | Fixed: a product bug in `Camera`'s move constructor, on every platform |
| #3948 | CUDA–Vulkan parity (5), masked SSIM | Plan: see [GPU numerics](#gpu-numerics) |
| #3948 | Fused normal gradients (3), `NormalLossHunt`, stack overflow, VMA objects, crash at exit | Plan: see [Gradients and teardown](#gradients-stack-overflow-vma-objects-and-the-crash-at-exit). The exit crash and the stack overflow are product bugs with one shared cause |

## Getting started on the Windows CUDA PC

Use the **x64 Native Tools Command Prompt for VS 2022**: the libplacebo port needs
`clang-cl` from it (see #3205 in [open_bug_plan.md](open_bug_plan.md)).

```bat
git fetch origin claude/vibrant-lovelace-747k7o
git switch claude/vibrant-lovelace-747k7o
cmake --preset windows-release -DBUILD_TESTS=ON
cmake --build --preset windows-release --target lichtfeld_tests vksplat_custom_effect_tests shader_service_tests
```

The build folder is `build-windows-release`. `/FORCE:UNRESOLVED` is no longer needed. If
a link still fails, that is #3944 not fully fixed: note the unresolved symbols.

Set `LFS_HOME` to a scratch folder as before, so the tests don't write into your real
user folder. The tests that depend on it now clear it themselves. `LFS_LOG_LEVEL=debug`
adds detail when a test needs it.

Then, in this order:

1. **Check the fixes.** Run each filter in [Checking the fixes](#checking-the-fixes). They
   are small and fast, and a failure points at one change.
2. **Run everything**, once per backend, and compare with the lists in #3948:
   ```bat
   lichtfeld_tests --tensor-backend=cuda --gtest_brief=1 --gtest_output=xml:cuda.xml
   lichtfeld_tests --tensor-backend=vulkan --gtest_brief=1 --gtest_output=xml:vulkan.xml
   ```
3. **Work through the open items** in [Still open](#still-open). Each has the experiment
   that decides it.

Tools that help:

- `compute-sanitizer`, from the CUDA toolkit, for #3947.
- Sysinternals `handle64` and Process Monitor (filter on Result = `SHARING VIOLATION`) for
  "file in use" errors.
- `procdump -e -x . lichtfeld_tests.exe ...` for crashes.

### Checking the fixes

| Fix | Command (add `--tensor-backend=cuda` unless noted) |
| --- | --- |
| #3944 | Build the three targets above |
| #3945 | `--gtest_filter=Backends/NodesCore.*`, with each backend |
| #3946 | `--gtest_filter=TensorMultiStreamTest.*` |
| #3947 | `--gtest_filter=Backends/NodesCore.*/CUDA:Backends/RestoredTensor.ConnectedComponentsOverload/Cuda`, then the same under `compute-sanitizer --tool memcheck` and `--tool initcheck` |
| Open files | `--gtest_filter=MetricsReporter.*:ViewEvaluationJson.*:MetricsEvaluatorControlTest.*:NormalAutoGenerateIntegration.*:PriorAutoGenerateIntegration.*:SsogFormat.*:PlyToRadLod.*:RadMetaSidecar.*:RadSparkPagingTest.*:RadLodLeaves.*` |
| Atomic writes | `--gtest_filter=UserPathsContractTest.* --gtest_repeat=200` |
| Paths and Unicode | `--gtest_filter=UnicodePathTest.*:CrashHandlerTest.*:PlyErrorTaxonomyTest.*:MissingDatasetImagesTest.*:VideoExtractorNamingUiTest.*` |
| Camera move | `--gtest_filter=CameraMove.*:CameraModels/GutScreenShare.*:GsplatRasterizerQuantTest.*` |
| Smaller fixes | `--gtest_filter=UsdHalfTest.*:SceneGraphIdentityTest.*:ViewerGpuMemory.*:GpuMemoryQuery.*:SessionBreadcrumbTest.*:TensorReserveInplaceCat*.*:LpipsOpsTest.*` |

## Fixed on this branch

### #3944: build

| Program | Cause | Change |
| --- | --- | --- |
| `lichtfeld_tests` | `gui::isBoundsGizmoActive()` is not exported from the `lfs_visualizer` DLL | `LFS_VIS_API` on its declaration (`src/visualizer/gui/bounds_gizmo.hpp`) |
| `vksplat_custom_effect_tests` | `<windows.h>` defines `far` as an empty macro | The local is renamed `far_offset` (`tests/test_vksplat_custom_effect.cpp`) |
| `shader_service_tests` | `lfs_visualizer` links the static `lfs_io` and `lfs_sequencer` privately, so their symbols don't reach the test through the DLL. `readOutputImageRgba` was the one `readOutputImage*` member without `LFS_VIS_API` | The test links `lfs_io` and `lfs_sequencer` itself, as `lichtfeld_visualizer_tests` does (`LFS_IO_API` is empty on Windows, so that's safe), and `readOutputImageRgba` is exported |

### #3945: node references kept across `add_node`

**Confirmed.** `NodeTree::nodes` is a `std::vector<Node>`, and `add_node` returns
`nodes.back()`. MSVC grows a vector by half, so a test's second node moves its first.
The five tests now keep names and look nodes up with `find_node` after the last
`add_node`, as the same file already did elsewhere.

A scan of every test found a sixth case,
`NodesCore.CameraNodesMatchSyntheticRingAndCapturedDistance`. It ignored the result of
`add_link`, so on Windows the camera node was never linked and `missing.ok` came back
true (listed in #3948). Its `add_link` calls are now `ASSERT_TRUE`.

`add_node` in `tree.hpp` now has a comment about the invalidation. Changing `nodes` to a
container with stable references (a `std::deque`, or `std::unique_ptr<Node>` elements)
would remove the trap. But `nodes` is a public member used directly in dozens of places,
including the Python bindings, so that's a separate change.

### #3946: the pinned-memory test hang

**Likely.** The test closed a gate (a `cudaLaunchHostFunc` that spins), then blocked in
`cudaStreamSynchronize` on an independent stream. On Windows that copy never completed
behind the spinning host function. Two changes:

- **No blocking while the gate is closed.** The D2H use of the block now finishes before
  the gate closes. While it is closed, the test waits for `d2h_stream` with a 10-second
  `cudaStreamQuery` loop, so a stall fails the test instead of hanging it (the destructor
  releases the gate).
- **No CUDA allocation while the gate is closed.** A spare block is cached first, so the
  allocation made while the gate is closed is a cache hit. A miss would call
  `cudaHostAlloc`, and CUDA's implicit-synchronization rules cover page-locked
  allocations.

The assertions are unchanged.

### #3947: CUDA connected components read past the union-find array

**What the code shows (confirmed):**

- The union kernels (`union_components`, `point_tree_union`) only ever point a root at a
  smaller index, and path halving only writes ancestors.
- `next[i]` is written for every point that enters a hash chain, and only those are
  followed.
- So no kernel can write a parent outside the array. An out-of-range parent can only come
  from `parent`, which is the `labels` tensor, holding something other than its
  initialization `[0, 1, …, n−1]` when the kernels read it.
- That matches initcheck ("reads entries of the `parent` array that were never written").
- It also matches the one address in the issue: subtracting 4 × `0x3F800000` (the bits of
  `1.0f`) from it gives a cleanly aligned base. That fits stale float data, or the
  `cumsum − 1` result before its cast to Int32.

**The change:** `launch_radius_connected_components` and `launch_point_tree_components`
now write `parent[i] = i` themselves, with a new `init_components` kernel. It runs on the
stream the union and flatten kernels use, right before them
(`src/core/tensor/backend/cuda/kernels/tensor_spatial.cu`). Both callers in
`tensor_spatial.cpp` pass exactly that identity, so nothing else changes. CPU, Vulkan and
HIP are untouched.

**Open:** why `labels` wasn't initialized on that stream. `tensor_spatial.cpp` builds it
as `(allocate_like(..., Int32, 1.0f).cumsum(0) - 1).to(Int32)`. That is a deferred
expression, materialized by `pin_operands`, and the kernels run on `labels.stream()`. If
the materializer runs on a different stream from the one `labels.stream()` reports, or
the lazy-executor materialization cache hands back a result produced elsewhere, other
code using this pattern is exposed too. To check:

1. Run the #3947 filter above with `CUDA_LAUNCH_BLOCKING=1` on the commit before this
   branch's fix (`f1ad7d687`). If the failures disappear, it is an ordering race.
2. On this branch, the filter should pass and compute-sanitizer should report nothing. If
   labels are still wrong, look for a late write to `labels` from another stream: log
   `labels.stream()` and `getCurrentCUDAStream()` in `radius_connected_components`, and
   compare with the stream `materialize_deferred_slow` binds.
3. Add a small test next to `tests/test_tensor_stream.cpp`'s late-initialization test:
   build `(Tensor::full({n}, 1, GPU, Int32).cumsum(0) - 1).to(Int32)` on one stream, read
   it from a kernel on `labels.stream()`, and check that it equals `[0, n)`.

The reporter didn't try CUDA on Linux, so this may not be Windows-specific.

### #3948: files still open

**Confirmed for all of these.** Windows can't delete a file, or rename over it, while a
handle to it is open without `FILE_SHARE_DELETE`, and MSVC streams and memory maps don't
share delete.

| Test | Held file | Change |
| --- | --- | --- |
| `MetricsReporter.VulkanRunRecordsCpuJpegDecoding` | the report the test reads (`ifstream report`) | Close it before `remove_all` |
| `ViewEvaluationJson.OneFileKeyedByImageNameInTheOutputFolder` | `per_image_metrics.json` (`in`) | Same |
| `MetricsEvaluatorControlTest.StopDuringPausedFinalEvaluationCancelsWithoutPartialMetrics` | `metrics.csv` (`csv`) | Same |
| `NormalAutoGenerateIntegration.OverwritesExistingMapWhenCameraHasNone` | `stale.png` (`in`) | Same |
| `PriorAutoGenerateIntegration.OneJobPerImageCarriesOnlyMissingMaps` | `normals/a.png` (`kept`) | Same |
| `SsogFormat.BundleCancellationPreservesDestination` | `scene.ssog` (`file`), removed mid-test | Same |
| `PlyToRadLod.OutOfCoreLoadKeepsTreeAndStreamsChunks` | `synthetic.rad.meta`, memory-mapped by the out-of-core model `partial` (`MappedFile` opens with `FILE_SHARE_READ` only) | The models live in a block that ends before `remove_all` |
| `RadMetaSidecar.RoundtripQuantizationAndInvalidation` | `sidecar.rad.meta`, mapped by `view` while the test rebuilds the sidecar over it | `view` lives in a block that ends before the rebuild |
| `RadMetaSidecar.ConverterEmitsSidecarInline`, `RadSparkPagingTest.StreamedExportOpensInThePagedViewer`, `RadLodLeaves.OutOfCoreModelRefusesToFlatten` | The temp sidecar, held by the builder's own workers | **Product fix:** `build_rad_meta_sidecar` closes each TBB worker's stream on the temp file before the rename. They were only flushed (`src/io/formats/rad.cpp`) |
| `UserPathsContractTest.ConcurrentAtomicWritesDoNotShareTemporaryFiles` | `preferences.json`, the destination of eight concurrent commits | **Product fix:** the Windows commit in `src/core/atomic_file.cpp` called `MoveFileExW` once. It now retries access denied, sharing and lock violations 20 × 50 ms, like `native_fs::rename`. This affects users: preferences, UI state, projects and image outputs all go through it |

The three RAD tests in the second-to-last row weren't traced individually. If one still
fails, the exception names the file, and `handle64 -p lichtfeld_tests.exe` shows the
holder.

### #3948: paths, Unicode and case

All of these are **confirmed** test bugs. The product code goes through `native_fs` and
the UTF-8 path helpers correctly.

| Test | Cause | Change |
| --- | --- | --- |
| `SsogFormat.BundleReadersAcceptSuffixCaseWithoutChangingRows` | It copies to `copy.sog`, then `copy.SOG`, which is the same file on NTFS | Each copy gets its own name |
| `MissingDatasetImagesTest.TransformsImagePathResolvesUppercasePngWithoutExtension` | On NTFS the `.png` probe already finds `frame-one.PNG` and returns the probe's spelling | On a case-insensitive volume, compare with `fs::equivalent` |
| `UnicodePathTest.CheckpointPathConstructionWithBufferPadding` | `path::string()` converts through the ANSI code page and throws | Compare `native()` sizes |
| `UnicodePathTest.Utf8ToPathHandlesUnicodeWithEmbeddedNulls` | `test_root_ / "日本語_output"` decodes the narrow literal with the ANSI code page | `utf8_to_path("日本語_output")`. The helper's stop at the first NUL is intended, and other tests rely on it |
| `CrashHandlerTest.InstallationReportsUnicodeCrashLogPathAsUtf8` | Windows death tests re-run the test body in a child, which inherits `LFS_HOME=root`. `test_main` opens its log file under `root`, so `remove_all(root)` fails and `ASSERT_FALSE(ec)` exits the child with 1 | The cleanup is best effort |
| `PlyErrorTaxonomyTest.ExportSupportsLongLegalFileName` | The path passes MAX_PATH. The exporter handles that through `native_fs`, but the test checked with plain `std::filesystem`, and `lichtfeld_tests` has no `longPathAware` manifest | The test and its `TearDown` use `native_fs` |
| `VideoExtractorNamingUiTest.BatchRowsDisambiguateDuplicateNamesAndExposePaths` | The output folder is joined with `operator/`, so the tooltip shows `batch_output\clip` | Expect the native separator |

Don't make these pass by giving `lichtfeld_tests` the app's manifest
(`activeCodePage=UTF-8`, `longPathAware`). That would hide the code-page and MAX_PATH
bugs that `native_fs` exists to prevent.

My earlier guesses for this group were wrong. Checkpoint paths don't call `string()`,
and `utf8_to_path` doesn't need the explicit length.

### #3948: smaller fixes

| Test | Cause | Change |
| --- | --- | --- |
| `UsdHalfTest.RoundTripAndRoundingMatchIEEE754` | **Confirmed.** The test's own reference converter was wrong, not `float_to_half`. MSVC has no `_Float16`, so it uses the fallback, which ORed a mantissa that still has its implicit bit into the exponent | The fallback adds it. It matches hardware `_Float16` on 611 million values on Linux |
| `SceneGraphIdentityTest.SceneInfoSourcePathIsMostRecentlyAddedNode` | **Confirmed product bug.** `getSceneInfo()` reported `splat_paths_.begin()` of an `unordered_map`, so the result depended on hash order on every platform | Picks the path of the node with the highest id (`scene_manager.cpp`) |
| `UserPathsContractTest.PortableRoot*` (2) | `LFS_HOME` takes precedence over the portable root | Both tests clear `LFS_HOME` |
| `ViewerGpuMemory.CudaStatusKeepsDeviceUsageAndItsOptionalUtilizationReading` | **Confirmed.** On Windows `process_budget` is the DXGI local budget, by design (`GpuMemoryQuery.OversubscriptionIsExplicitAndDeviceUsageIsPhysical` expects it) | The `== 0` check is Linux-only |
| `SessionBreadcrumbTest.SnapshotsOnlyTheUncleanPreviousLogTail` | **Likely.** The breadcrumb resolves its log through `UserPaths`, which honours `LFS_HOME`. `Logger::default_log_file_path()` ignores it, so with `LFS_HOME` set the two point at different files | `HomeDirectoryGuard` clears `LFS_HOME`. The product side is in [Still open](#still-open) |
| `TensorReserveInplaceCatCuda.AllocationFailurePreservesInstalledStorage` | **Likely.** Under WDDM, the driver's sysmem fallback lets `cudaMalloc` go past dedicated memory into shared system memory, so "total memory + 4 bytes" succeeds | It asks for 256 TiB. To confirm, set "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback" for `lichtfeld_tests.exe`: the old test then passes |
| `CameraModels/GutScreenShare.*/4` (4), `GsplatRasterizerQuantTest.DecodedFp32AndQ16AreBitwiseEqualAcrossBandsAndWarmup` | **Confirmed product bug, on every platform.** `Camera`'s move constructor and move assignment (`src/core/camera.cpp`) copied every member except `_camera_model_type` and `_has_alpha`. A moved thin-prism or fisheye camera became a pinhole camera that kept its 4 radial coefficients, and the PINHOLE validation (since #2683) throws "Pinhole cameras require at most three polynomial radial coefficients…". The tests pick their camera with `cond ? make_a() : make_b()`. GCC and Clang build that in place, while MSVC appears to move it (Likely), which is why only Windows saw it. `_has_alpha` was also missing from the transform copy, so a moved camera stopped using its alpha as a mask | Both members are copied, and the new `CameraMove.KeepsModelTypeAndAlpha` test (CPU only) covers it. Expect `GutScreenShare.*/2` (fisheye) and `/3` (equirectangular) to change on Windows: they ran as pinhole before and passed, and now really use their models |
| `LpipsOpsTest.RejectsOperandDtypesBeforeLaunching` | **Likely.** The test looks for `convolution` in the host stack trace. On MSVC, internal-linkage functions only resolve with private PDB symbols, which a Release `lfs_training` doesn't have. The test's own frame has an external name, so it resolves | The operand checks now name their operation ("CUDA LPIPS convolution output must be…"), so the report says it without symbols. This also helps Release diagnostics |

## Still open

### Product gaps found along the way

- **`HostMetrics.CpuAndPerCoreBecomeValidAfterSecondSample`.** **Confirmed.** CPU sampling
  in `src/core/host_metrics.cpp` exists only for POSIX (it reads `/proc`). The Windows
  branch only sizes `per_core_cpu_percent` to −1, so the HUD always shows "--" for CPU on
  Windows. Implement `read_cpu` with:
  - `GetProcessTimes` for the process;
  - `GetSystemTimes` for the system (its kernel time includes idle);
  - `NtQuerySystemInformation(SystemProcessorPerformanceInformation)` for each core.

  All three count in 100 ns units, so `ticks_per_second` = 1e7 on Windows. Then reuse the
  POSIX delta code. Over the test's 10 ms a delta of 0 is legitimate and gives 0 %, which
  the test accepts. Verify with `--gtest_filter=HostMetrics.*`.
- **The log path ignores `LFS_HOME`, portable and explicit roots.** "Open Log"
  (`error_event_bridge.cpp`) and Python `lfs.log.log_file_path()` call
  `Logger::default_log_file_path()`. That always returns `~/.lichtfeld/logs/lichtfeld.log`,
  while the app logs wherever `UserPaths` says. Better than copying the `UserPaths` rules
  into the logger: have the logger remember the file it opened, and return that once it
  is initialized.

### Need one experiment each

- **`NvCodecSentinelValidatorTest.SkippedMemberIsRetriedOrFailsHardBeforeReturn`** (Likely).
  - **Suspected cause:** the test compares two different JPEG decoders bit for bit. When
    `nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE)` succeeds, the first member goes to the
    hardware JPEG engine (registered at VERY_HIGH priority). The retried member goes
    through the CUDA decoder (`nvcodec_image_loader.cpp`, the HYBRID_CPU_GPU retry
    decoder).
  - **Experiment:** print the mismatch count and the largest difference. A few levels of
    rounding confirms this; a leftover sentinel differs everywhere.
  - **Fix if confirmed:** compare with a PSNR threshold (> 40 dB) instead of `EXPECT_EQ`.
- **`VideoFrameExtractorTrim.SparseFpsStopsAtTheLastDecodableFrameInAStreamCopy`** (Likely).
  - **Suspected cause:** "sequential" means `chooseFrameSelection` decided before any
    seek. FFmpeg is pinned to the same version on both platforms, so the input that can
    differ is `using_hw_decode`: NVDEC probably didn't take the 64×48 fixture.
  - **Experiment:** run it alone and read the "Frame selection: sequential (reason=…)"
    line, any `h264_cuvid` errors, and `processing.decoder.backend` in
    `extraction_metadata.json`.
  - **Fix:** if it's the software fallback, expect "sequential" when the backend isn't
    NVDEC, or regenerate the fixture at 128×128 with the same timestamps.
- **`VideoExportUtilsTest.EquirectangularMarkerPixelsFollowViewportCamera`** (flaky, Open).
  - **What was ruled out:** the readback waits on the render timeline and invalidates
    before copying, and no unsynchronized readback was found.
  - **Experiments:**
    - `--gtest_repeat=200 --gtest_break_on_failure` alone, per backend;
    - then `--gtest_filter=VideoExportUtilsTest.* --gtest_repeat=50` and a full run with
      `--gtest_shuffle`, to find an order dependence.

    The failing assertion line decides where to look. Possibilities:
    - `:320`: the Vulkan backend shutdown, while earlier tests still hold Vulkan tensors;
    - `:416`: the marker position;
    - the bounded wait in `vulkan_wait.hpp` timing out.

### GPU numerics

None of these run in CI, on Linux or Windows: hosted jobs run only CPU and media labels,
and the opt-in self-hosted GPU job runs only node and spatial tests.

**CUDA versus Vulkan parity** (Open):

- `TrainingOpsFastParity.NonzeroImageGradientAndFusedAdamVulkan`,
  `PixelErrorDensificationWithoutStrategyModeVulkan` and
  `SeptemberRegularizersAndMeanStepsMatchCuda`
- `TrainingOpsGeometryParity.LargeUnweightedImagesVulkan`
- `Backends/TrainingOpsFamilyParity.MatchesCuda/Photometric_Vulkan`

All are in `tests/test_training_ops_parity.cpp`.

**Ruled out by reading:**

- The inputs are deterministic integer patterns, not `std::` random distributions.
- Both backends run in one process.
- The nvcc flags are the same on both platforms.
- The SPIR-V is built the same way on both.

That leaves the NVIDIA driver's shader compiler, host dispatch compiled by MSVC, and the
GPU architecture.

**The quoted mismatch**, `1101.adam.2.parameter index 2 -0.081000 vs -0.079000`, comes
from the Fast tests, not Photometric:

- It is the first Adam step on splat 0's rotation (quaternion y = −0.08, lr = 1e-3).
- A first step moves by exactly ±lr, so CUDA and Vulkan disagree on the sign of that
  gradient.
- Everything compared before it matched: the image, alpha, depth and normal, and the
  means and scales.
- FastGS normals reach the rotation gradient only through `grad_normal_helper`. That
  points at the CUDA normal-channel failures in the next section, so start there.

**Experiments, in order:**

1. Fix the [fused normal-channel gradients](#gradients-stack-overflow-vma-objects-and-the-crash-at-exit)
   first, then rerun the Fast parity tests.
2. Run each failing test alone. Record the first failing field, since the tests stop at
   the first mismatch, and whether it passes in isolation (the suite has global
   `set_*_for_testing` hooks).
3. Compare against Linux with the built-in dumps:
   - `LFS_TRAINING_OPS_GOLDEN=<dir>` with `--gtest_filter=TrainingOpsParity.WriteGoldens`.
     If the CUDA goldens are byte-identical across the two OSes, only the Vulkan side
     moved.
   - `LFS_PARITY_DIAGNOSTICS=<dir>` with
     `--gtest_filter=TrainingOpsFastParity.SeptemberRegularizersAndMeanStepsMatchCuda`.
4. Vary the Vulkan side with `--tensor-no-atomic-float` (it changes the summation order),
   or another driver version. Failures that follow the driver are codegen.

**Likely fixes, once the first failing fields are known:**

- **Geometry** compares fp64 per-block partial sums byte-exactly over 1.09M pixels. Across
  two compilers, use the `kReduce` tolerance between backends and keep exact comparison
  within one backend. Alternatively, mark the shader arithmetic that must match as
  `precise`, as the expression JIT already does (NoContraction).
- **Photometric** stores SSIM partials as fp16, which amplifies 1-ULP driver differences.
  Same choice as for Geometry.
- **Fast:** compare raw gradients (`prepare_fastgs_gradient_capture`) or a second Adam
  step, not the sign that a first step exposes.

**`MaskedSSIMMetricTest.AllSupportedMapsMatchUnmaskedMaps`** (Vulkan run only, Likely):

- **What differs:** 0.98671073 against 0.98671055, which is 3 ULP. The two arms run the
  same `photometric_fused` shader with different specialization constants, so the driver
  compiles two pipelines, and the shared unweighted arithmetic isn't `precise`.
- **Experiment:** temporarily change the `ASSERT_EQ` at
  `tests/test_masked_ssim_metrics.cpp:172` to `EXPECT_EQ`, and count the differing
  elements.
- **Fix:** a few scattered 1–4 ULP differences mean codegen: allow ≤ 4 ULP for Vulkan, or
  make the shared path `precise`. Nearly every element differing would mean the masked
  arm took the weighted path because `1.0f / max(1.0f, mask)` wasn't exactly 1. That
  would be a product bug.

### Gradients, stack overflow, VMA objects and the crash at exit

**One mechanism behind the crash at exit and the stack overflow.** On Windows, the app,
the tests and most dependencies are DLLs (Linux links them statically).

- `std::_Exit`, used by `flush_and_exit`, ends in `ExitProcess`, which still runs every
  DLL's static destructors, after killing the other threads.
- By then, the delay-loaded `nvcuda.dll` and the Vulkan driver have already shut down.

So the promise in `tests/test_main.cpp` ("do NOT return into C++ static/TLS destruction")
doesn't hold on Windows. On Linux, `_Exit` skips all of that.

**Fused normal-channel gradients** (Open; a product bug if confirmed, because it affects
training with the normal loss):

- **Tests:** `FastGSNormalChannelTest.BackwardNormalRotationGradientMatchesFiniteDifferences`
  and `BackwardNormalRotationGradientUsesCompactVisibleIndex` (the gradient is 0 where
  finite differences give −0.0429), and
  `NormalLossRegression.OffCenterOverlappingFullFusedGradients` (3.162 against −0.0602).
- **Ruled out:**
  - random inputs (the scenes are hard-coded);
  - `long` and `long double`;
  - `_WIN32` and `_MSC_VER` branches;
  - kernel argument layout;
  - nvcc flags;
  - cross-stream races (everything runs on one stream);
  - different kernel variants.
- **What is left:** `DeferredNormalRotationGradientMatchesFiniteDifferences` (same scene
  and kernels) and the fused `FastGSDepthGradientTest.*` are not in the failing list. So
  what fails is the fused Adam path combined with the normal channel. 3.162 = √10 is the
  value the moment codec stores after one step from zero, which hints that the test read
  moment storage rather than a gradient.
- **Experiments:**
  1. `--tensor-backend=cuda --gtest_filter=FastGSNormalChannelTest.*:FastGSDepthGradientTest.*:NormalLossRegression.OffCenterOverlappingFullFusedGradients`:
     confirm that the deferred test passes and the fused ones fail.
  2. Repeat with `CUDA_LAUNCH_BLOCKING=1`, then run
     `compute-sanitizer --tool racecheck` and `--tool initcheck` on
     `FastGSNormalChannelTest.BackwardNormalRotationGradientMatchesFiniteDifferences`.
  3. Run `nsys profile -t cuda -o nrm lichtfeld_tests.exe --tensor-backend=cuda --gtest_filter=FastGSNormalChannelTest.BackwardNormalRotationGradientMatchesFiniteDifferences`
     and `nsys stats -r cuda_gpu_kern_sum nrm.nsys-rep`. The `blend_backward_cu<…>` row
     must have its normal-channel template argument set to `true`.
  4. After the backward, print `opt.get_state(ParamType::Rotation)->joint_bounds.cpu()`
     and the first 16 bytes of `exp_avg`. All-zero bounds mean the rotation row was never
     updated; non-zero bounds point at per-cell encoding or decoding.
  5. Run the same commit on Linux with an RTX 50-series card (sm_120, CUDA 12.8), to
     separate Windows from the GPU and toolkit.

**`NormalLossHunt.JointRotationCodec100kSteps`** (Likely test bug):

- **Cause:** it asserts the codec's error is at most a pre-fix error measured on Linux,
  with no margin (0.000488 for mode 1; Windows gave 0.00088). Mode 1 is the only one that
  calls `std::sin`, and the codec uses `log`, `log1p`, `exp` and `expm1`. glibc and the
  MSVC runtime can differ in the last bit, and 100k feedback steps compound that.
- **Unaffected:** the test's real purpose, zero-variance cells decoding as exactly zero
  (#2066), is checked exactly.
- **Fix:** compute the pre-fix error in the same run, with the decoder from before #2066
  (no zero-variance guard), and assert the new error doesn't exceed it. Or give mode 1 a
  margin.
- **Confirm:** compare the printed `HUNT codec mode=1 …` line with Linux. If `tiny_fp32`
  differs, the FP32 reference itself differs (`sinf` or `pow`).

**`MeshDataTest.ProcessExitsCleanlyAfterMesh2Splat`** (Vulkan run, Likely, product bug):

- **Cause:**
  - The death-test child builds the converter's output on the default tensor backend,
    which creates the tensor library's Vulkan context, then calls `std::exit(0)`.
  - The static `g_context_slot` (`vk_context.cpp`) is then destroyed after the Vulkan
    driver has shut down. `~VulkanContext` and `shutdown()` call into the driver.
  - mesh2splat leaks its own `VkDevice` for exactly this reason (`mesh2splat.cpp`, the
    comment near the end), but the tensor library's context isn't leaked.
- **Decide it in one step:** add `lfs::core::shutdown_gpu_backend(lfs::core::GpuBackend::Vulkan);`
  before the test's `std::exit(0)`. If the child then exits 0, the cause is confirmed.
  Or capture a dump with Windows Error Reporting LocalDumps for `lichtfeld_tests.exe` and
  run `!analyze -v; kc 200` in WinDbg.
- **Fix:** allocate the context slot and its mutex on the heap and never free them, as
  `vulkan_queue_sync.cpp` already does, and make `~ThreadRecorderToken` do nothing once
  teardown has started.
- **Who else is affected:** any Windows process on the Vulkan tensor backend that exits
  through `exit()` without the GPU teardown. For example, the Python module registers
  `std::atexit(shutdown_python_tensor_backend)`, which runs only when the `.pyd` unloads,
  after the driver.

**`Backends/ExpressionTest.CpuScalarUsesInlineParameters/Vulkan`** (Open):

- **How the count works:** it includes freed-but-pending buffers, and the recorder batches
  up to 64 commands. This kernel passes its 5 words as push constants, so it shouldn't
  allocate at all.
- **Experiment:** run it alone with `--gtest_repeat=3`.
  - +3 on the first repetition only means objects created on first use. That's a test
    bug: launch once, `synchronize_device()` and trim before taking `before`, then
    synchronize and trim before comparing.
  - +3 every time means an allocation per launch, which is a product bug. Break on
    `VulkanMemory::allocate_storage` and `ensure_staging` after the first count to see
    who allocates.

**`lichtfeld_tests` exits with 0xC0000005 after the report** (mechanism confirmed, product
bug: the app exits the same way):

- **Cause:** DLL static destructors run after the drivers shut down, as above. One
  example: the static `RandomGenerator` owns a cuRAND generator, and `~CudaRandomState`
  calls `curandDestroyGenerator` then. Killed worker threads stop at different places
  each run, which fits a crash in 3 of 4 runs.
- **Confirm first:** run `procdump -accepteula -ma -e -x C:\dumps lichtfeld_tests.exe --tensor-backend=cuda --gtest_brief=1`,
  then `!analyze -v; kc` in WinDbg. Expect
  `LdrShutdownProcess → <module>!dllmain_crt_process_detach → _execute_onexit_table → dynamic atexit destructor for '…'`.
- **Fix:**
  - On Windows, make `flush_and_exit` (`src/core/crash_handler.cpp`) flush stdio and the
    logger, then call `TerminateProcess(GetCurrentProcess(), code)`. That has the effect
    `_Exit` already has on Linux.
  - Release the random generator's CUDA state in `teardown_gpu_before_exit`.
- **Verify:** 10 full runs per backend with exit code 0.

## Keep Windows from regressing

Windows CI doesn't build `lichtfeld_tests`. It builds only the standalone
`unicode_path_test`, which leaves out the tests under `LFS_HAS_FULL_LIBRARY`
([#677](https://github.com/MrNeRF/LichtFeld-Studio/issues/677),
[#1481](https://github.com/MrNeRF/LichtFeld-Studio/issues/1481)).

- **Build only.** Building the three test programs with `-DBUILD_TESTS=ON` in the Windows
  job, without running them, would have caught all of #3944.
- **CPU subset.** Running `ctest -L cpu` would catch most of the file, path and Unicode
  bugs above. Those are the ones that affect users too.
