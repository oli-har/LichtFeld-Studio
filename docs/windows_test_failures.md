# Windows test failures: fixes and plan

Status of the Windows test reports filed against dev `ab623c39d`:

- [#3944](https://github.com/MrNeRF/LichtFeld-Studio/issues/3944): three test programs don't build
- [#3945](https://github.com/MrNeRF/LichtFeld-Studio/issues/3945): node tests keep a `Node&` across `add_node`
- [#3946](https://github.com/MrNeRF/LichtFeld-Studio/issues/3946): `PinnedBlockReusedOnlyAfterAllStreamsDone` hangs
- [#3948](https://github.com/MrNeRF/LichtFeld-Studio/issues/3948): the other failing tests, grouped by cause

The changes on this branch were made on Linux without CUDA, MSVC or a GPU. None of
them has been compiled or run on Windows yet. Each section says what still needs
to be checked there.

## Fixed on this branch

### #3944: build

| Program | Cause | Change |
| --- | --- | --- |
| `lichtfeld_tests` | `gui::isBoundsGizmoActive()` is not exported from the `lfs_visualizer` DLL | `LFS_VIS_API` on its declaration (`src/visualizer/gui/bounds_gizmo.hpp`) |
| `vksplat_custom_effect_tests` | `<windows.h>` defines `far` as an empty macro | Renamed the local to `far_offset` (`tests/test_vksplat_custom_effect.cpp`) |
| `shader_service_tests` | `lfs_visualizer` links the static `lfs_io` and `lfs_sequencer` privately, so their symbols don't reach the test through the DLL; `readOutputImageRgba` was the one `readOutputImage*` member without `LFS_VIS_API` | Linked `lfs_io` and `lfs_sequencer` into the test, as `lichtfeld_visualizer_tests` already does, and exported `readOutputImageRgba` (`tests/CMakeLists.txt`, `src/visualizer/rendering/vksplat_viewport_renderer.hpp`) |

To check on Windows: `cmake --preset windows-release -DBUILD_TESTS=ON`, then build
all three targets without `/FORCE:UNRESOLVED`. If `shader_service_tests` still
has unresolved symbols, they come from the remaining private dependencies of
`lfs_visualizer`. List them with `dumpbin /exports lfs_visualizer.dll`.

### #3945: node references kept across `add_node`

`NodeTree::nodes` is a `std::vector<Node>`, and `add_node` returns `nodes.back()`.
MSVC grows a vector by half, so the fourth node (the second one a test adds) moves
the third. The five tests in the issue now keep a name, and look the node up with
`find_node` after the last `add_node`. The file already did this further down
("Node references do not survive later add_node calls; keep names").

A scan of every `tests/*.cpp` for a `Node&` used after a later `add_node` found one
more: `NodesCore.CameraNodesMatchSyntheticRingAndCapturedDistance`. It ignored
the result of `add_link`, so on Windows the camera node was never linked and
`missing.ok` came back true. This is the "Other" entry in #3948. It is fixed the
same way, and its `add_link` calls are now `ASSERT_TRUE`.

`add_node` in `src/core/include/core/nodes/tree.hpp` now has a comment saying the
reference is invalidated by later calls. The production callers don't keep one,
so the container type is unchanged. Making `nodes` a `std::deque`, or storing
`std::unique_ptr<Node>`, would remove the trap. But `nodes` is a public member
used in about 40 places, including the Python bindings, so that is a separate change.

To check on Windows: `lichtfeld_tests --gtest_filter=Backends/NodesCore.*` with
each `--tensor-backend`.

### #3946: pinned-memory test deadlock

Before, the test closed the gate (a `cudaLaunchHostFunc` that spins), enqueued
the gated H2D copy and an independent D2H copy, and then blocked in
`cudaStreamSynchronize(d2h_stream)`. On Windows the D2H copy never completed
while the host function spun.

Now the D2H use of the block finishes before the gate closes. After
`deallocate(host_block, d2h_stream)`, the test waits for `d2h_stream` with a
10-second `cudaStreamQuery` loop instead of a blocking synchronize. The
assertions are unchanged: a block is not reused while the gated H2D copy is
pending, and is reused once every stream is done. If Windows still serializes
even the event that `deallocate` records, the test now fails with
"d2h_stream did not finish while the gate was closed" instead of hanging, because
`~GateStream` releases the gate.

To check on Windows: `lichtfeld_tests --gtest_filter=TensorMultiStreamTest.*`.

### #3948: fixed so far

| Test(s) | Cause | Change |
| --- | --- | --- |
| `UsdHalfTest.RoundTripAndRoundingMatchIEEE754` | The test's reference converter was wrong, not `float_to_half`. MSVC has no `_Float16`, so the fallback is used there. It ORed a mantissa that includes the implicit bit into the exponent field, so 2^-14 came out as `0x0800` | The fallback adds the rounded mantissa to `(exponent - 113) << 10`. On Linux it matches hardware `_Float16` on every 7th float bit pattern (611 million non-NaN values) |
| `RadMetaSidecar.*` (2), `RadSparkPagingTest.StreamedExportOpensInThePagedViewer`, `RadLodLeaves.OutOfCoreModelRefusesToFlatten` | **Library bug.** `build_rad_meta_sidecar` gives each TBB worker its own `std::fstream` on the temporary sidecar. The streams were only flushed, and stayed open until the function returned, after the `rename`. Windows refuses to rename a file that has an open handle, so this also affects users | Workers' streams are closed before the rename (`src/io/formats/rad.cpp`) |
| `SceneGraphIdentityTest.SceneInfoSourcePathIsMostRecentlyAddedNode` | **Library bug.** `getSceneInfo()` reported `splat_paths_.begin()`, and `splat_paths_` is a `std::unordered_map`, so the "source path" depended on hash order. It is wrong on every platform, but only showed up on MSVC | Picks the path whose node has the highest id. Node ids only grow, and an undone delete gets its old id back (`src/visualizer/scene/scene_manager.cpp`) |
| `UserPathsContractTest.PortableRootUsesExecutableDirectory`, `PortableRootRequiresExecutableDirectory` | `LFS_HOME` takes precedence over the portable root, and was set for the run | Both tests clear `LFS_HOME`, as other tests in the file already do |
| `Backends/NodesCore.CameraNodesMatchSyntheticRingAndCapturedDistance` | Same as #3945 | See above |

## Plan for the rest of #3948

Unless a cause is stated as found, it is a hypothesis from reading the code and
needs a Windows run to confirm. Order of work: the user-facing items first
(open handles, atomic writes, Unicode), then the test-only ones.

### 1. Files still open when a temp folder is removed

`MetricsEvaluatorControlTest.StopDuringPausedFinalEvaluationCancelsWithoutPartialMetrics`,
`MetricsReporter.VulkanRunRecordsCpuJpegDecoding`,
`NormalAutoGenerateIntegration.OverwritesExistingMapWhenCameraHasNone`,
`PlyToRadLod.OutOfCoreLoadKeepsTreeAndStreamsChunks`,
`PriorAutoGenerateIntegration.OneJobPerImageCarriesOnlyMissingMaps`,
`ViewEvaluationJson.OneFileKeyedByImageNameInTheOutputFolder`,
`SsogFormat.BundleCancellationPreservesDestination`.

The same family as the rad sidecar: something still holds a handle (a stream, a
`FILE*`, a memory map, a log sink, an image decoder) when the test's
`TearDown` calls `remove_all`. Plan:

1. Rerun one of them alone and use Sysinternals `handle.exe <temp dir>`, or
   Process Explorer, at the failing `remove_all` to see which file is held open.
2. `PlyToRadLod.OutOfCoreLoad...` probably holds the out-of-core RAD mapping
   until the model is destroyed. Check that the test destroys the loaded model
   before removing the folder, and that the mapping is released on destruction.
3. For the metrics and view-evaluation tests, look for output streams owned by
   objects that outlive the job (reporters, writers kept in a `shared_ptr`).
4. Prefer fixing the owner (close on completion or cancel) over retrying the
   delete in the test. Only add a retry in `TearDown` if the handle is held
   by a third-party library, such as an antivirus or indexer scan.

### 2. Atomic writes

`UserPathsContractTest.ConcurrentAtomicWritesDoNotShareTemporaryFiles` fails
with "Atomic file commit failed ... Access is denied"
(`src/core/atomic_file.cpp`, the `_WIN32` commit). Concurrent
`MoveFileExW(..., MOVEFILE_REPLACE_EXISTING)` onto the same target returns
`ERROR_ACCESS_DENIED` while another thread has the target open, or is itself
replacing it. Plan: on `ERROR_ACCESS_DENIED` and `ERROR_SHARING_VIOLATION`,
retry the commit with a short backoff (about 10 × 10 ms) before failing. This
is the usual Windows pattern. Also check that the temp files are opened with
`FILE_SHARE_DELETE`. This affects users: concurrent preference writes.

### 3. File names that differ only in case

- `SsogFormat.BundleReadersAcceptSuffixCaseWithoutChangingRows` copies
  `lower.sog` to `copy.SOG` in the same folder, which already exists on a
  case-insensitive file system. Copy to a name that differs in more than case
  (`copy_upper.SOG`).
- `MissingDatasetImagesTest.TransformsImagePathResolvesUppercasePngWithoutExtension`
  expects the on-disk spelling `frame-one.PNG`. On Windows the resolver gets
  the spelling it probed with. Either compare with
  `std::filesystem::equivalent`, or expect the case only when the file system
  is case-sensitive (probe once in the fixture).

### 4. Unicode, long names and paths

- `UnicodePathTest.CheckpointPathConstructionWithBufferPadding` ("No mapping for
  the Unicode character exists in the target multi-byte code page"): something
  calls `path::string()` (ANSI code page) instead of `lfs::core::path_to_utf8`.
  Find the call in the checkpoint path construction. This affects users with
  non-ASCII project folders.
- `UnicodePathTest.Utf8ToPathHandlesUnicodeWithEmbeddedNulls`: the converter
  probably stops at the first `\0` (it uses `MultiByteToWideChar` with `-1`, or
  a `c_str()`). Pass the explicit length.
- `CrashHandlerTest.InstallationReportsUnicodeCrashLogPathAsUtf8`: the death
  test exits with status 1. Run the child alone to see which assertion fails,
  probably the same narrow-string conversion.
- `PlyErrorTaxonomyTest.ExportSupportsLongLegalFileName`: the full path is longer
  than `MAX_PATH`. Either the exporter opens it without the `\\?\` prefix, or
  the test binary has no `longPathAware` manifest. Check which. The app's
  manifest matters too, if users can hit this.
- `VideoExtractorNamingUiTest.BatchRowsDisambiguateDuplicateNamesAndExposePaths`:
  the tooltip contains `batch_output\clip`. Compare with `generic_string()`, or
  build the expectation with `std::filesystem::path`.

### 5. Camera validation (likely not Windows-specific)

`CameraModels/GutScreenShare.*/4` (4 tests) and
`GsplatRasterizerQuantTest.DecodedFp32AndQ16AreBitwiseEqualAcrossBandsAndWarmup`
throw "Pinhole cameras require at most three polynomial radial coefficients or
six rational radial coefficients" (`gsplat_rasterizer_cuda.cpp`). Variant 4 is
`THIN_PRISM_FISHEYE`, with four radial coefficients, so a thin-prism camera reaches the
`PINHOLE` branch. Check whether `Camera` or the rasterizer maps the model
type before that `switch`, and whether CI on Linux runs these GPU tests at all.
If Linux CI fails too, this is a real regression, not a Windows issue.

### 6. Numbers that differ

- CUDA vs Vulkan parity (`TrainingOpsFastParity.*Vulkan`,
  `TrainingOpsGeometryParity.LargeUnweightedImagesVulkan`,
  `TrainingOpsFamilyParity.MatchesCuda/Photometric_Vulkan`), and gradients
  against finite differences (`FastGSNormalChannelTest.*`,
  `NormalLossRegression.OffCenterOverlappingFullFusedGradients`,
  `NormalLossHunt.JointRotationCodec100kSteps`): first run them on Linux with the
  same commit and an RTX 50-series GPU. If they pass there, compare MSVC
  host-side code (`/fp:precise` vs GCC's contraction defaults; `std::` math
  differences) and the SPIR-V compiled on Windows. A gradient that is exactly 0
  where finite differences give -0.0429 looks like a code path that was skipped
  (a compile-time branch or an unsupported feature), not rounding.
- `MaskedSSIMMetricTest.AllSupportedMapsMatchUnmaskedMaps` (3 ULP, Vulkan only):
  compare with a ULP tolerance, unless the test intentionally requires bitwise
  equality.

### 7. Other

| Test | Likely cause / next step |
| --- | --- |
| `MeshDataTest.ProcessExitsCleanlyAfterMesh2Splat` (Vulkan) | Stack overflow (0xC00000FD) in the child. The default Windows stack is 1 MB (8 MB on Linux), so look for a deep recursion or a large stack array in mesh2splat or exit-time teardown. Get a dump with `procdump -e` |
| `Backends/ExpressionTest.CpuScalarUsesInlineParameters/Vulkan` | 3 more VMA objects after the test. Probably lazily created caches (staging buffers) that are created on first use on this driver. Check whether the count is stable on a second run in the same process |
| `HostMetrics.CpuAndPerCoreBecomeValidAfterSecondSample` | `GetProcessTimes` has ~15.6 ms resolution. If the two samples are closer than that, the delta is 0 and the code reports -1. Sleep past one tick between samples, or treat a zero interval as "not yet valid" |
| `LpipsOpsTest.RejectsOperandDtypesBeforeLaunching` | The error text comes from a different layer on Windows. Print the report and compare with Linux |
| `SessionBreadcrumbTest.SnapshotsOnlyTheUncleanPreviousLogTail` | Probably CRLF, or the log not flushed or still open when the snapshot is read (see group 1) |
| `NvCodecSentinelValidatorTest.SkippedMemberIsRetriedOrFailsHardBeforeReturn` | Driver/NVDEC version dependent. Compare with Linux on the same GPU and driver branch |
| `TensorReserveInplaceCatCuda.AllocationFailurePreservesInstalledStorage` | Under WDDM, allocations can exceed physical memory (the system pages to host memory), so "more than total device memory" can succeed. Skip on WDDM (`cudaDeviceProp::tccDriver == 0` on Windows) or use a size above the address-space limit |
| `ViewerGpuMemory.CudaStatusKeepsDeviceUsageAndItsOptionalUtilizationReading` | Windows reports a process budget (DXGI), Linux does not. The test should accept a budget when the platform provides one |
| `VideoFrameExtractorTrim.SparseFpsStopsAtTheLastDecodableFrameInAStreamCopy` | The strategy choice depends on the FFmpeg build (keyframe index or seek support). Compare `av_version_info()` with the Linux build |
| `VideoExportUtilsTest.EquirectangularMarkerPixelsFollowViewportCamera` | Flaky (1 of 4). Run with `--gtest_repeat=50` to get a failure rate, and look for an unsynchronized readback |
| `lichtfeld_tests` exits with 0xC0000005 after the report | Static destruction order: a singleton (CUDA, Vulkan, pinned allocator, logger) is used after its DLL or driver shut down. Get a dump with `procdump -e -x . lichtfeld_tests.exe` and fix the order, or skip work in destructors once shutdown has started |

### 8. Keep Windows from regressing

Windows CI doesn't build the test programs
([#677](https://github.com/MrNeRF/LichtFeld-Studio/issues/677),
[#1481](https://github.com/MrNeRF/LichtFeld-Studio/issues/1481)). Even
without a GPU runner, building them with `-DBUILD_TESTS=ON` in the Windows job
(build only, no run) would have caught all of #3944. Running the CPU-labelled
`ctest -L cpu` subset would also catch most of groups 1–4.
