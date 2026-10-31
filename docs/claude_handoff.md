# Handoff prompt for Claude Code on the Windows CUDA PC

Give this file to Claude Code on the Windows machine, or paste it in as the prompt. Issue
numbers refer to [MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio/issues).

---

You're picking up work on LichtFeld Studio that a Claude Code cloud session started.
That session ran on Linux without a GPU, CUDA or MSVC, so **nothing it changed has been
compiled or run**. This machine is the Windows CUDA PC where the bugs were found. Your job:
build the work, check it, and carry on with what's left.

## Where everything is

- **Repository:** my fork, https://github.com/oli-har/LichtFeld-Studio. Upstream is
  MrNeRF/LichtFeld-Studio.
- **Branch:** `claude/vibrant-lovelace-747k7o`. It is upstream `dev` at `809251714` plus
  five commits:
  - `f1ad7d687`:
    - the Windows test build (#3944);
    - node references kept across `add_node` (#3945);
    - the pinned-memory test hang (#3946);
    - first #3948 fixes: the scene source path, the `UsdHalfTest` reference converter
      and the `LFS_HOME` tests.
  - `456edca3f`:
    - CUDA connected components (#3947);
    - the Windows atomic-commit retry;
    - RAD sidecar handles;
    - the open-file, Unicode and path test fixes;
    - the #3946 spare block.
  - `4e69ae8ca`: `Camera` move keeps its model type and alpha flag (a product bug on every
    platform), named LPIPS messages, more test fixes, and the rewritten plan.
  - `c9836860f`:
    - the `LFS_JPEG2K_CACHE=0` switch (#3604);
    - uv copied next to multi-config executables (#1569);
    - a clear error when vcpkg is missing (#3205).
  - `16ce812cf`: save-step edits are applied even after a project save (#1651), and the
    timelapse honours `--enable-mip`.
- **Read these two files before anything else.** They are the plan and the record:
  - `docs/windows_test_failures.md`, for #3944–#3948. It starts with "Getting started on
    the Windows CUDA PC" (the prompt, the build commands and one gtest filter per fix).
    Then it covers what was fixed and why. Then every open failure, with its cause, its
    confidence (Confirmed, Likely or Open), the fix and the experiment that decides it.
  - `docs/open_bug_plan.md`, for 11 other open upstream bugs: #3876, #3604, #2570, #2783,
    #1569, #3205, #3940, #2135, #1651, #1756 and #1786.
- **The original reports.** I filed #3944–#3948 from a Windows run of `lichtfeld_tests`
  at dev `ab623c39d`: Windows 11 10.0.26100, RTX 5090 32 GB, driver 617.14, MSVC
  14.44.35207, CUDA 12.8, LibTorch 2.7.1+cu128. Read the full reports with
  `gh issue view <number> --repo MrNeRF/LichtFeld-Studio`.

## Step 1: get it and build it

Use the **x64 Native Tools Command Prompt for VS 2022**.

1. Run `git fetch origin claude/vibrant-lovelace-747k7o`, then
   `git switch claude/vibrant-lovelace-747k7o`, then
   `git submodule update --init --recursive`.
2. Run `cmake --preset windows-release -DBUILD_TESTS=ON`, then
   `cmake --build --preset windows-release --target lichtfeld_tests vksplat_custom_effect_tests shader_service_tests`.
   - The build folder is `build-windows-release`.
   - `/FORCE:UNRESOLVED` should no longer be needed. If a link still fails, #3944 isn't
     fully fixed: note the unresolved symbols.

If it doesn't compile, fix the cloud session's mistakes first; they were never compiled.
The edits most likely to need attention:

- `src/core/atomic_file.cpp`: the Windows-only `MoveFileExW` retry loop.
- `src/core/tensor/backend/cuda/kernels/tensor_spatial.cu`: the new `init_components`
  kernel.
- `src/visualizer/training/training_manager.cpp` and
  `src/visualizer/core/training_manager.hpp`: the `applied_param_serial_` edit serial.
- `src/io/pipelined_image_loader.cpp`: the `LFS_JPEG2K_CACHE` switch.
- `cmake/FetchUV.cmake` and the top of `CMakeLists.txt`.
- Tests:
  - `tests/test_metrics_eval_mask.cpp` (`CameraMove.KeepsModelTypeAndAlpha`);
  - `tests/test_missing_dataset_images.cpp` (a lambda);
  - `tests/test_system_info.cpp` (`HomeDirectoryGuard`);
  - `tests/test_video_frame_extractor_naming.cpp` (a new include);
  - `tests/test_ply_to_rad_lod.cpp` and `tests/test_rad_meta_sidecar.cpp` (re-indented
    blocks).

## Step 2: check the fixes

1. Set `LFS_HOME` to a scratch folder.
2. Run every filter in the "Checking the fixes" table in `docs/windows_test_failures.md`,
   with `--tensor-backend=cuda`, and also `vulkan` where the table says so.
3. For #3947, also run its filter under `compute-sanitizer --tool memcheck` and
   `--tool initcheck`. To learn the root cause rather than just the fix, check out
   `f1ad7d687` and see whether `CUDA_LAUNCH_BLOCKING=1` makes the failures disappear. The
   suspicion is an ordering race in how the `labels` tensor is initialized; see the #3947
   section.
4. Do two full runs:
   `lichtfeld_tests --tensor-backend=cuda --gtest_brief=1 --gtest_output=xml:cuda.xml`,
   and the same with `vulkan`. Compare them with the lists in #3948.
5. Tell me which fixes are confirmed, and which failed and why. Update the docs to match:
   turn Likely into Confirmed, or correct the cause.

## Step 3: what's left, roughly in order of value

1. **The crash at exit (0xC0000005) and `MeshDataTest.ProcessExitsCleanlyAfterMesh2Splat`.**
   These are product bugs, probably with one cause: on Windows, `std::_Exit` still runs
   DLL static destructors after the CUDA and Vulkan drivers have shut down.
   - Capture a dump with procdump first.
   - Then apply the fixes in the doc: `TerminateProcess` in `flush_and_exit`, and leak
     the tensor library's Vulkan context the way mesh2splat already does.
2. **The fused normal-channel gradients**: `FastGSNormalChannelTest` (2 tests) and
   `NormalLossRegression.OffCenterOverlappingFullFusedGradients`. This may be a real
   training bug on Windows.
   - The doc has the experiments: an nsys check of the `blend_backward_cu` template
     argument, and a dump of the rotation `joint_bounds`.
   - Then rerun the CUDA–Vulkan parity tests; the Fast ones probably follow from it.
3. **HostMetrics.** Implement Windows CPU sampling. There's a sketch in the doc using
   `GetProcessTimes`, `GetSystemTimes` and `NtQuerySystemInformation`.
4. **Items that need one experiment each:** `NvCodecSentinelValidatorTest`,
   `VideoFrameExtractorTrim`, `VideoExportUtilsTest` (flaky),
   `Backends/ExpressionTest.CpuScalarUsesInlineParameters/Vulkan`, `MaskedSSIMMetricTest`
   and `NormalLossHunt`.
5. **Larger product work, with plans ready:**
   - #3876: the RAD import builds the full SH on the GPU before deciding to page.
   - #2570: SSOG export peak memory.
   - #3604: make the switch a real option, and add a watchdog.
   - #1651 part 2: headless `--resume` ignores edited save steps.
   - #3940: the fastgs forward/backward inconsistencies.
6. **Needs a decision from me or the maintainers before coding:**
   - #1756: resume 0.5.3 checkpoints with fresh Adam state, or refuse them up front.
   - #1569: whether uv may fall back to one found on PATH.
   - #3940: full mip-filter support in 3DGUT.
7. **Can probably be closed upstream** (tell me; don't comment yourself):
   - #2783: fixed by #3198.
   - #3940 as filed: `dev` refuses the combination; the issue describes `master`.
   - #1786: the failure has been visible since #1658; only the identity request remains.
   - #2135: the reporter closed it in a comment.

## How to work

- Commit to `claude/vibrant-lovelace-747k7o`, or a branch made from it, with clear
  messages, and push to `origin`, my fork. Never push to MrNeRF/LichtFeld-Studio, and
  don't open pull requests or post on GitHub issues unless I ask.
- Keep each fix to what the failure needs, match the surrounding code, and run
  `git clang-format` on the changed lines.
- Only call something fixed after its test passes on this machine. Mark everything else
  Likely or Open in the docs.
- To drive the app, for example to check the save-step fix in the GUI, follow `AGENTS.md`
  (the MCP bridge in `.mcp.json`).

## Notes from the cloud session

**The Windows behaviour behind most of the failures:**

- MSVC grows a vector by half, not double.
- MSVC appears to move through `cond ? prvalue : prvalue`. This is unconfirmed; there is
  a ten-line repro in the doc.
- A file can't be deleted or replaced while it is open or memory-mapped without delete
  sharing.
- `path::string()` and narrow string literals use the ANSI code page.
- MAX_PATH applies, because `lichtfeld_tests` has no `longPathAware` manifest. Don't give
  the tests the app's manifest: that would hide the bugs `native_fs` exists to prevent.
- Death tests re-run the test body in a child process.
- DLL static destructors run after the GPU drivers have shut down.
- Under WDDM, `cudaMalloc` can spill into system RAM.
- DXGI reports a per-process memory budget.
- Functions with internal linkage don't resolve in Release stack traces without private
  PDB symbols.

**Unconfirmed assumptions worth checking early:**

- The MSVC `?:` move, which explains the camera-validation failures.
- What really hung in #3946: the D2H synchronize or `cudaHostAlloc` while the gate was
  closed. The test now avoids both.
- The WDDM sysmem fallback, for `TensorReserveInplaceCatCuda`.
- Whether RmlUi keeps backslashes in tooltip attributes.

**Earlier versions of the plan had some wrong guesses** (Unicode, HostMetrics, NvCodec, the
flaky render test). The current docs replace them.
