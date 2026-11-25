// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// MRNF growth at the cap, per splat: the per-view error fold and its mean (mrnf_growth_kernels.cu), the blend peak,
// keep-covered factor and retire key (mrnf_retire_kernels.cu). An absent input is a null pointer.
struct Parameters {
    Ptr<float, Access.ReadWrite> error;   // fold_view, mean_view, fold_need: the refine weights [N]
    Ptr<float, Access.ReadWrite> info;    // fold_view, mean_view: densification rows [4, N]
    Ptr<float, Access.ReadWrite> views;   // fold_view, optional: views that covered each splat [N]
    Ptr<float, Access.ReadWrite> step;    // blend_peak: this step's peak [N], read and cleared
    Ptr<float, Access.ReadWrite> peak;    // blend_peak, covered_factor, retire_key: running peak [N]
    Ptr<float, Access.ReadWrite> renders; // blend_peak, covered_factor: renders since birth [N]
    Ptr<float, Access.Read> opacity;      // covered_factor, retire_key: raw opacities [N]
    Ptr<float, Access.Read> scales;       // covered_factor: raw log scales [N, 3]
    Ptr<float, Access.ReadWrite> result;  // covered_factor, retire_key [N]
    uint count;
    float x, y, z;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

// Exponent applied to each view's per-splat error sum before it enters the window.
static const float kViewErrorPower = 0.2f;

// A row's reach: clamp(peak / max(sigmoid(raw opacity), 1e-6), 0, 1), exactly 1 at and above the opacity.
float row_reach(float peak, float raw_opacity) {
    const float opacity = max(1.0f / (1.0f + exp(-raw_opacity)), 1e-6f);
    return peak >= opacity ? 1.0f : clamp(peak / opacity, 0.0f, 1.0f);
}

// Adds this view's compressed error sum (row 1) to the window's weights, counts the views with error (row 3) and,
// when `views` is given, the views that covered the splat; adds the coverage (row 0) to row 2 and clears rows 0
// and 1. x unused.
[shader("compute")][numthreads(256, 1, 1)]
void fold_view(uint3 id : SV_DispatchThreadID) {
    const uint i = id.x;
    if (i >= p.count)
        return;
    const uint64_t n = p.count;
    const float coverage = p.info[i];
    if ((uint64_t)p.views != 0 && coverage > 0.0f)
        p.views[i] += 1.0f;
    const float value = p.info[n + i];
    const float compressed = value > 0.0f ? pow(value, kViewErrorPower) : 0.0f;
    p.error[i] += compressed;
    if (compressed > 0.0f)
        p.info[3 * n + i] += 1.0f;
    p.info[2 * n + i] += coverage;
    p.info[i] = 0.0f;
    p.info[n + i] = 0.0f;
}

// The window's error sum over the views that saw an error (row 3, at least one).
[shader("compute")][numthreads(256, 1, 1)]
void mean_view(uint3 id : SV_DispatchThreadID) {
    const uint i = id.x;
    if (i >= p.count)
        return;
    p.error[i] /= max(p.info[3 * uint64_t(p.count) + i], 1.0f);
}

// NEED: (error / x clamped at y) to the power z, 0 without error.
[shader("compute")][numthreads(256, 1, 1)]
void fold_need(uint3 id : SV_DispatchThreadID) {
    const uint i = id.x;
    if (i >= p.count)
        return;
    const float e = p.error[i];
    p.error[i] = e > 0.0f ? pow(min(e / p.x, p.y), p.z) : 0.0f;
}

// Folds the step's peak into the running peak, which decays by x per step, counts a render where the step saw the
// splat, and clears the step's peak. The clear comes last: through MoltenVK a use of the step read after its clear
// saw the cleared value.
[shader("compute")][numthreads(256, 1, 1)]
void blend_peak(uint3 id : SV_DispatchThreadID) {
    const uint i = id.x;
    if (i >= p.count)
        return;
    const float value = p.step[i];
    if (value > 0.0f)
        p.renders[i] += 1.0f;
    p.peak[i] = max(p.peak[i] * p.x, value);
    p.step[i] = 0.0f;
}

// Keep covered: the reach of a splat rendered at least x times whose reach is below y and whose largest log scale
// is at least z; 1 for every other splat (within the grace the reach is unknown and counts as visible).
[shader("compute")][numthreads(256, 1, 1)]
void covered_factor(uint3 id : SV_DispatchThreadID) {
    const uint i = id.x;
    if (i >= p.count)
        return;
    const float reach = p.renders[i] >= p.x ? row_reach(p.peak[i], p.opacity[i]) : 1.0f;
    const uint64_t row = 3 * uint64_t(i);
    const float largest = max(max(p.scales[row], p.scales[row + 1]), p.scales[row + 2]);
    p.result[i] = reach < p.y && largest >= p.z ? reach : 1.0f;
}

// Retire key: dead rows (peak below x) land in [-1, -1 + x), below every reach in [0, 1].
[shader("compute")][numthreads(256, 1, 1)]
void retire_key(uint3 id : SV_DispatchThreadID) {
    const uint i = id.x;
    if (i >= p.count)
        return;
    const float peak = p.peak[i];
    p.result[i] = peak < p.x ? peak - 1.0f : row_reach(peak, p.opacity[i]);
}
