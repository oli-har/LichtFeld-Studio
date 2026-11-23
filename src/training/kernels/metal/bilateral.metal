// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// BilateralOps kernels: ports of bilateral_grid_forward.cu,
// bilateral_grid_backward.cu and bilateral_grid_tv.cu. Grids are [C][L][H][W]
// per image; the exposure-chroma transform uses the PPISP stages of ppisp.metal.

constant constexpr float kC2G_r = 0.299f;
constant constexpr float kC2G_g = 0.587f;
constant constexpr float kC2G_b = 0.114f;

struct BilateralSliceParams {
    device const float* grid;
    device const float* rgb;
    device const float* grad_output;
    device const float* offset;
    device float* output; // forward output, or the RGB gradient
    device float* grad_grid;
    int L, H, W, h, w;
    uint chw;
    uint exposure_chroma;
};

// Trilinear cell of one pixel and its guidance.
struct BilateralCell {
    int x0, y0, z0, x1, y1, z1;
    float fx, fy, fz, z;
};

static BilateralCell bilateral_cell(constant BilateralSliceParams& p, const int wi, const int hi, const float3 rgb) {
    const float x = p.w > 1 ? float(wi) / float(p.w - 1) * float(p.W - 1) : 0.0f;
    const float y = p.h > 1 ? float(hi) / float(p.h - 1) * float(p.H - 1) : 0.0f;
    const float guidance = fmin(1.0f, fmax(0.0f, kC2G_r * rgb.x + kC2G_g * rgb.y + kC2G_b * rgb.z));
    BilateralCell c;
    c.z = guidance * float(p.L - 1);
    c.x0 = int(floor(x));
    c.y0 = int(floor(y));
    const int z0 = int(floor(c.z));
    c.x1 = min(c.x0 + 1, p.W - 1);
    c.y1 = min(c.y0 + 1, p.H - 1);
    c.z0 = min(max(z0, 0), p.L - 1);
    c.z1 = min(max(z0 + 1, 0), p.L - 1);
    c.fx = x - float(c.x0);
    c.fy = y - float(c.y0);
    c.fz = c.z - float(c.z0);
    return c;
}

static float bilateral_offset(constant BilateralSliceParams& p, const int ci) { return p.offset ? p.offset[ci] : 0.0f; }

static float bilateral_sample(constant BilateralSliceParams& p, const BilateralCell c, const int ci) {
    device const float* g = p.grid + ci * p.L * p.H * p.W;
    const int H = p.H, W = p.W;
    const float v000 = g[(c.z0 * H + c.y0) * W + c.x0], v001 = g[(c.z0 * H + c.y0) * W + c.x1];
    const float v010 = g[(c.z0 * H + c.y1) * W + c.x0], v011 = g[(c.z0 * H + c.y1) * W + c.x1];
    const float v100 = g[(c.z1 * H + c.y0) * W + c.x0], v101 = g[(c.z1 * H + c.y0) * W + c.x1];
    const float v110 = g[(c.z1 * H + c.y1) * W + c.x0], v111 = g[(c.z1 * H + c.y1) * W + c.x1];
    const float c00 = v000 * (1 - c.fx) + v001 * c.fx;
    const float c01 = v010 * (1 - c.fx) + v011 * c.fx;
    const float c10 = v100 * (1 - c.fx) + v101 * c.fx;
    const float c11 = v110 * (1 - c.fx) + v111 * c.fx;
    const float c0 = c00 * (1 - c.fy) + c01 * c.fy;
    const float c1 = c10 * (1 - c.fy) + c11 * c.fy;
    return c0 * (1 - c.fz) + c1 * c.fz + bilateral_offset(p, ci);
}

static float3 bilateral_load(constant BilateralSliceParams& p, device const float* image, const int pixel) {
    const int hw = p.h * p.w;
    return p.chw != 0 ? float3(image[pixel], image[hw + pixel], image[2 * hw + pixel])
                      : float3(image[pixel * 3], image[pixel * 3 + 1], image[pixel * 3 + 2]);
}

static void bilateral_store(constant BilateralSliceParams& p, device float* image, const int pixel, const float3 v) {
    const int hw = p.h * p.w;
    if (p.chw != 0) {
        image[pixel] = v.x;
        image[hw + pixel] = v.y;
        image[2 * hw + pixel] = v.z;
    } else {
        image[pixel * 3] = v.x;
        image[pixel * 3 + 1] = v.y;
        image[pixel * 3 + 2] = v.z;
    }
}

static float3 bilateral_finite_or(const float3 v, const float fallback) {
    return float3(ppisp_isfinite(v.x) ? v.x : fallback, ppisp_isfinite(v.y) ? v.y : fallback,
                  ppisp_isfinite(v.z) ? v.z : fallback);
}

static PpispColor bilateral_color(thread const float* sampled) {
    return {float2(sampled[1], sampled[2]), float2(sampled[3], sampled[4]), float2(sampled[5], sampled[6]),
            float2(sampled[7], sampled[8])};
}

// Coefficient of affine channel ci: ci % 4 selects r, g, b or 1.
static float bilateral_coefficient(const float3 rgb, const int ci) {
    const int si = ci % 4;
    return si == 3 ? 1.0f : rgb[si];
}

// d(output) / d(sampled[0..8]) and d(output) / d(rgb) of the exposure-chroma transform.
static float3 bilateral_exposure_chroma_vjp(const float3 rgb, thread const float* sampled, const float3 grad_out,
                                            thread float* grad_sampled) {
    const float3 exposed = ppisp_apply_exposure(rgb, sampled[0]);
    for (int i = 1; i < 9; ++i)
        grad_sampled[i] = 0.0f;
    const float3 grad_exposed = ppisp_apply_color_correction_bwd(exposed, bilateral_color(sampled), grad_out,
                                                                 grad_sampled + 1);
    return ppisp_apply_exposure_bwd(rgb, sampled[0], grad_exposed, grad_sampled[0]);
}

kernel void bilateral_slice_forward(constant BilateralSliceParams& p [[buffer(0)]],
                                    uint index [[thread_position_in_grid]]) {
    const int pixel = int(index);
    if (pixel >= p.h * p.w)
        return;
    const float3 rgb = bilateral_finite_or(bilateral_load(p, p.rgb, pixel), 0.5f);
    const BilateralCell c = bilateral_cell(p, pixel % p.w, pixel / p.w, rgb);
    float3 out = 0.0f;
    if (p.exposure_chroma != 0) {
        float sampled[9];
        for (int ci = 0; ci < 9; ++ci)
            sampled[ci] = bilateral_sample(p, c, ci);
        out = ppisp_apply_color_correction(ppisp_apply_exposure(rgb, sampled[0]), bilateral_color(sampled));
    } else {
        for (int ci = 0; ci < 12; ++ci)
            out[ci / 4] += bilateral_sample(p, c, ci) * bilateral_coefficient(rgb, ci);
    }
    bilateral_store(p, p.output, pixel, bilateral_finite_or(out, 0.5f));
}

kernel void bilateral_slice_backward_rgb(constant BilateralSliceParams& p [[buffer(0)]],
                                         uint index [[thread_position_in_grid]]) {
    const int pixel = int(index);
    if (pixel >= p.h * p.w)
        return;
    const float3 rgb = bilateral_finite_or(bilateral_load(p, p.rgb, pixel), 0.5f);
    const float3 grad = bilateral_finite_or(bilateral_load(p, p.grad_output, pixel), 0.0f);
    const BilateralCell c = bilateral_cell(p, pixel % p.w, pixel / p.w, rgb);
    const int H = p.H, W = p.W, L = p.L;
    const float w00 = (1 - c.fx) * (1 - c.fy), w01 = c.fx * (1 - c.fy);
    const float w10 = (1 - c.fx) * c.fy, w11 = c.fx * c.fy;
    float3 value = 0.0f;
    float gz_grad = 0.0f;
    if (p.exposure_chroma != 0) {
        float sampled[9], dl[9];
        for (int ci = 0; ci < 9; ++ci)
            sampled[ci] = bilateral_sample(p, c, ci);
        value = bilateral_exposure_chroma_vjp(rgb, sampled, grad, dl);
        const int cx[8] = {c.x0, c.x1, c.x0, c.x1, c.x0, c.x1, c.x0, c.x1};
        const int cy[8] = {c.y0, c.y0, c.y1, c.y1, c.y0, c.y0, c.y1, c.y1};
        const int cz[8] = {c.z0, c.z0, c.z0, c.z0, c.z1, c.z1, c.z1, c.z1};
        const float dwdz[8] = {-w00, -w01, -w10, -w11, w00, w01, w10, w11};
        for (int ci = 0; ci < 9; ++ci)
            for (int k = 0; k < 8; ++k)
                gz_grad += dwdz[k] * (L - 1) * p.grid[(ci * L + cz[k]) * H * W + cy[k] * W + cx[k]] * dl[ci];
    } else {
        for (int ci = 0; ci < 12; ++ci) {
            const int si = ci % 4, di = ci / 4;
            const float r_coeff = bilateral_coefficient(rgb, ci);
            const float gout = grad[di];
            device const float* g0 = p.grid + (ci * L + c.z0) * H * W;
            device const float* g1 = p.grid + (ci * L + c.z1) * H * W;
            const int r0 = c.y0 * W, r1 = c.y1 * W;
            const float off = bilateral_offset(p, ci);
            const float p0 = w00 * g0[r0 + c.x0] + w01 * g0[r0 + c.x1] + w10 * g0[r1 + c.x0] + w11 * g0[r1 + c.x1] + off;
            const float p1 = w00 * g1[r0 + c.x0] + w01 * g1[r0 + c.x1] + w10 * g1[r1 + c.x0] + w11 * g1[r1 + c.x1] + off;
            const float v = p0 + (p1 - p0) * c.fz;
            if (si < 3)
                value[si] += v * gout;
            gz_grad += (p1 - p0) * (L - 1) * r_coeff * gout;
        }
    }
    gz_grad *= float(float(c.z0) != c.z && float(c.z1) != c.z);
    bilateral_store(p, p.output, pixel, value + float3(kC2G_r, kC2G_g, kC2G_b) * gz_grad);
}

// The grid gradient takes one threadgroup of 16x16 threads per 32x32 pixel tile,
// each thread 2x2 pixels.
constant constexpr int kBilateralTile = 32;

// Grid levels the threadgroup sums cover; grids with more levels add every total with device atomics.
constant constexpr int kBilateralSliceLevels = 8;
constant constexpr int kBilateralSimdGroups = 8; // 16x16 threads
// One SIMD group's sums for its tile's base cell: channel, level, spatial corner.
constant constexpr int kBilateralSlice = 12 * kBilateralSliceLevels * 4;

kernel void bilateral_slice_backward_grid(constant BilateralSliceParams& p [[buffer(0)]],
                                          uint2 tile [[threadgroup_position_in_grid]],
                                          uint2 thread_in_tile [[thread_position_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]],
                                          uint simd [[simdgroup_index_in_threadgroup]],
                                          uint rank [[thread_index_in_threadgroup]]) {
    threadgroup float slices[kBilateralSimdGroups * kBilateralSlice];
    const int channels = p.exposure_chroma != 0 ? 9 : 12;
    const int L = p.L, H = p.H, W = p.W, h = p.h, w = p.w;
    const int px0 = int(tile.x) * kBilateralTile, py0 = int(tile.y) * kBilateralTile;
    // The tile's base cell is its first pixel's, as bilateral_cell places it. Cells span hundreds of pixels at full
    // resolution, so most sums are for it; they collect in the SIMD group's slice and go out once per tile.
    const int base_x = w > 1 ? int(floor(float(min(px0, w - 1)) / float(w - 1) * float(W - 1))) : 0;
    const int base_y = h > 1 ? int(floor(float(min(py0, h - 1)) / float(h - 1) * float(H - 1))) : 0;
    const bool sliced = L <= kBilateralSliceLevels;
    for (uint e = rank; e < uint(kBilateralSimdGroups * kBilateralSlice); e += kBilateralSimdGroups * 32u)
        slices[e] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* const own = slices + simd * uint(kBilateralSlice);

    // The thread's 2x2 pixels: their gradients with respect to the sampled grid channels, and their cells.
    float dl[4][12];
    float fx[4], fy[4], fz[4];
    int cell[4];
    uint pending = 0;
    for (int n = 0; n < 4; ++n) {
        const int wi = px0 + int(thread_in_tile.x) * 2 + n % 2;
        const int hi = py0 + int(thread_in_tile.y) * 2 + n / 2;
        const bool valid = wi < w && hi < h;
        const int pixel = valid ? hi * w + wi : 0;
        const float3 rgb = bilateral_finite_or(bilateral_load(p, p.rgb, pixel), 0.5f);
        const float3 grad = valid ? bilateral_finite_or(bilateral_load(p, p.grad_output, pixel), 0.0f) : 0.0f;
        const BilateralCell c = bilateral_cell(p, min(wi, w - 1), min(hi, h - 1), rgb);
        if (p.exposure_chroma != 0) {
            float sampled[9];
            for (int ci = 0; ci < 9; ++ci)
                sampled[ci] = bilateral_sample(p, c, ci);
            bilateral_exposure_chroma_vjp(rgb, sampled, grad, dl[n]);
            dl[n][9] = dl[n][10] = dl[n][11] = 0.0f;
        } else {
            for (int ci = 0; ci < 12; ++ci)
                dl[n][ci] = bilateral_coefficient(rgb, ci) * grad[ci / 4];
        }
        fx[n] = c.fx;
        fy[n] = c.fy;
        fz[n] = c.fz;
        cell[n] = (c.z0 * H + c.y0) * W + c.x0;
        if (valid)
            pending |= 1u << n;
    }

    // A 32x4-pixel SIMD group spans few grid cells: per cell its lanes sum their pixels' gradients, four channels at a
    // time through SIMD sums that every lane receives, and lane l keeps channel 4q + l / 8 at corner l % 8. Metal has
    // no threadgroup float atomics, so the base cell's totals go to the SIMD group's own slice (the lanes' slots
    // differ, except a top-level cell's two levels, whose upper corners then carry exact zeros, which are skipped)
    // and other cells' to the grid with device atomics.
    const int k = int(lane % 8u);
    while (simd_any(pending != 0)) {
        const ushort leader = ushort(simd_min(pending != 0 ? lane : 32u));
        int first = 0;
        for (int n = 3; n >= 0; --n)
            if ((pending & (1u << n)) != 0)
                first = cell[n];
        const int lead_cell = simd_shuffle(first, leader);
        uint mine = 0;
        for (int n = 0; n < 4; ++n)
            if ((pending & (1u << n)) != 0 && cell[n] == lead_cell)
                mine |= 1u << n;
        pending &= ~mine;
        const int x0 = lead_cell % W, y0 = lead_cell / W % H, z0 = lead_cell / (W * H);
        const int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1), z1 = min(z0 + 1, L - 1);
        const int z = (k & 4) != 0 ? z1 : z0;
        const int corner = (z * H + ((k & 2) != 0 ? y1 : y0)) * W + ((k & 1) != 0 ? x1 : x0);
        const bool base = sliced && x0 == base_x && y0 == base_y;
        for (int q = 0; q < 3; ++q) {
            float total = 0.0f;
            for (int kk = 0; kk < 8; ++kk) {
                float4 value = 0.0f;
                for (int n = 0; n < 4; ++n) {
                    if ((mine & (1u << n)) == 0)
                        continue;
                    const float wx = (kk & 1) != 0 ? fx[n] : 1 - fx[n];
                    const float wy = (kk & 2) != 0 ? fy[n] : 1 - fy[n];
                    const float wz = (kk & 4) != 0 ? fz[n] : 1 - fz[n];
                    value += wx * wy * wz * float4(dl[n][4 * q], dl[n][4 * q + 1], dl[n][4 * q + 2], dl[n][4 * q + 3]);
                }
                const float4 sums = simd_sum(value);
                if (kk == k)
                    total = sums[lane / 8u];
            }
            const int ci = 4 * q + int(lane / 8u);
            if (ci >= channels || total == 0.0f)
                continue;
            if (base)
                own[(ci * kBilateralSliceLevels + z) * 4 + (k & 3)] += total;
            else
                ppisp_atomic_add(p.grad_grid + ci * L * H * W + corner, total);
        }
        // The next cell's lanes may update this cell's slots (an upper level of one cell is a lower level of the
        // next): order the SIMD group's threadgroup accesses between iterations.
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!sliced)
        return;
    const int base_x1 = min(base_x + 1, W - 1), base_y1 = min(base_y + 1, H - 1);
    for (uint e = rank; e < uint(channels * kBilateralSliceLevels * 4); e += kBilateralSimdGroups * 32u) {
        float total = 0.0f;
        for (int g = 0; g < kBilateralSimdGroups; ++g)
            total += slices[uint(g * kBilateralSlice) + e];
        if (total == 0.0f)
            continue;
        const int ci = int(e) / (kBilateralSliceLevels * 4), z = int(e) / 4 % kBilateralSliceLevels, s = int(e) % 4;
        const int x = (s & 1) != 0 ? base_x1 : base_x, y = (s & 2) != 0 ? base_y1 : base_y;
        ppisp_atomic_add(p.grad_grid + ((ci * L + z) * H + y) * W + x, total);
    }
}

struct BilateralTvParams {
    device const float* grids;
    device float* output; // stage 1 partials, the loss, or the gradient
    device const float* partials;
    int N, C, L, H, W, norm_n;
    int partial_count;
    float grad_output;
};

kernel void bilateral_tv_partials(constant BilateralTvParams& p [[buffer(0)]], uint group [[threadgroup_position_in_grid]],
                                  uint groups [[threadgroups_per_grid]], uint local [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                                  uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float partials[32];
    const int L = p.L, H = p.H, W = p.W;
    const int total = p.N * L * H * W;
    float sum = 0.0f;
    for (int idx = int(group * kThreadgroupWidth + local); idx < total; idx += int(groups * kThreadgroupWidth)) {
        const int wi = idx % W, hi = (idx / W) % H, li = (idx / (W * H)) % L, ni = idx / (W * H * L);
        for (int ci = 0; ci < p.C; ++ci) {
            const int cell = (ni * p.C + ci) * L * H * W + (li * H + hi) * W + wi;
            const float v = p.grids[cell];
            if (wi > 0) {
                const float d = v - p.grids[cell - 1];
                sum += d * d / float(L * H * (W - 1));
            }
            if (hi > 0) {
                const float d = v - p.grids[cell - W];
                sum += d * d / float(L * (H - 1) * W);
            }
            if (li > 0) {
                const float d = v - p.grids[cell - W * H];
                sum += d * d / float((L - 1) * H * W);
            }
        }
    }
    sum /= float(p.C * p.norm_n);
    const float block = ppisp_group_sum(sum, partials, lane, simd, simds);
    if (local == 0)
        p.output[group] = block;
}

kernel void bilateral_tv_total(constant BilateralTvParams& p [[buffer(0)]], uint local [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                               uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float partials[32];
    float sum = 0.0f;
    for (int i = int(local); i < p.partial_count; i += int(kThreadgroupWidth))
        sum += p.partials[i];
    const float total = ppisp_group_sum(sum, partials, lane, simd, simds);
    if (local == 0)
        p.output[0] = total;
}

kernel void bilateral_tv_backward(constant BilateralTvParams& p [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    const int L = p.L, H = p.H, W = p.W;
    if (int(index) >= p.N * L * H * W)
        return;
    const int wi = int(index) % W, hi = (int(index) / W) % H, li = (int(index) / (W * H)) % L;
    const int ni = int(index) / (W * H * L);
    const float s = 2.0f * p.grad_output / float(p.C * p.norm_n);
    const float sx = s / float(L * H * (W - 1));
    const float sy = s / float(L * (H - 1) * W);
    const float sz = s / float((L - 1) * H * W);
    for (int ci = 0; ci < p.C; ++ci) {
        const int cell = (((ni * p.C + ci) * L + li) * H + hi) * W + wi;
        const float v = p.grids[cell];
        float half_grad = 0.0f;
        if (wi > 0)
            half_grad += (v - p.grids[cell - 1]) * sx;
        if (wi < W - 1)
            half_grad += (v - p.grids[cell + 1]) * sx;
        if (hi > 0)
            half_grad += (v - p.grids[cell - W]) * sy;
        if (hi < H - 1)
            half_grad += (v - p.grids[cell + W]) * sy;
        if (li > 0)
            half_grad += (v - p.grids[cell - W * H]) * sz;
        if (li < L - 1)
            half_grad += (v - p.grids[cell + W * H]) * sz;
        p.output[cell] += half_grad;
    }
}

struct BilateralProjectParams {
    device float* grids;
    device const float* mean;
    device const float* identity;
    int total, C, spatial, per_image;
};

kernel void bilateral_project_mean(constant BilateralProjectParams& p [[buffer(0)]],
                                   uint index [[thread_position_in_grid]]) {
    if (int(index) >= p.total)
        return;
    const int ci = (int(index) / p.spatial) % p.C;
    const int ni = int(index) / (p.spatial * p.C);
    const float m = p.per_image != 0 ? p.mean[ni * p.C + ci] : p.mean[ci];
    p.grids[index] += p.identity[ci] - m;
}

struct BilateralOffsetParams {
    device float* channel_sum;
    device float* shared_offset;
    device const float* identity;
    device const float* old_mean;
    device const float* new_mean;
    int C;
    float spatial, inv_n_spatial;
};

kernel void bilateral_update_offset(constant BilateralOffsetParams& p [[buffer(0)]],
                                    uint index [[thread_position_in_grid]]) {
    if (int(index) >= p.C)
        return;
    p.channel_sum[index] += (p.new_mean[index] - p.old_mean[index]) * p.spatial;
    p.shared_offset[index] = p.identity[index] - p.channel_sum[index] * p.inv_n_spatial;
}

struct BilateralAdamParams {
    device float* grid;
    device float* moment1;
    device float* moment2;
    device const float* gradient;
    uint count;
    float lr, beta1, beta2, bc1_rcp, bc2_sqrt_rcp, eps;
};

kernel void bilateral_adam(constant BilateralAdamParams& p [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index >= p.count)
        return;
    const float g = p.gradient[index];
    const float m = p.beta1 * p.moment1[index] + (1.0f - p.beta1) * g;
    const float v = p.beta2 * p.moment2[index] + (1.0f - p.beta2) * g * g;
    p.moment1[index] = m;
    p.moment2[index] = v;
    const float m_hat = m * p.bc1_rcp;
    const float v_hat = v * p.bc2_sqrt_rcp * p.bc2_sqrt_rcp;
    p.grid[index] -= p.lr * m_hat / (sqrt(v_hat) + p.eps);
}

struct BilateralScaleParams {
    device float* moment1;
    device float* moment2;
    uint count;
    float scale1, scale2;
};

kernel void bilateral_scale_moments(constant BilateralScaleParams& p [[buffer(0)]],
                                    uint index [[thread_position_in_grid]]) {
    if (index >= p.count)
        return;
    p.moment1[index] *= p.scale1;
    p.moment2[index] *= p.scale2;
}
