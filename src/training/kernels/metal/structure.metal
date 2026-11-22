/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
constant uint kStructureOperation [[function_constant(2)]];
struct StructureParams {
    ulong a, b, c, d, e, f, g;
    int width, height, first_row, rows;
    uint stride, blocks, image_byte, mask_byte, radius, first_scale, valid_padding, reserved;
    float gain, sigma_sq;
    float filter_g[21], filter_d1[21], filter_d2[21];
};

inline float st_read(ulong a, int i) { return ((device const float*)a)[i]; }
inline uint st_byte(ulong a, int i) { return uint(((device const uchar*)a)[i]); }
inline void st_write(ulong a, int i, float v) { ((device float*)a)[i] = v; }
inline float st_value(ulong a, int i, uint byte_value) { return byte_value != 0 ? float(st_byte(a, i)) * (1.0f / 255.0f) : st_read(a, i); }
inline float st_luma_at(ulong image, int n, uint image_byte, int i) {
    float r = image_byte != 0 ? float(st_byte(image, i)) : st_read(image, i);
    float g = image_byte != 0 ? float(st_byte(image, n + i)) : st_read(image, n + i);
    float b = image_byte != 0 ? float(st_byte(image, 2 * n + i)) : st_read(image, 2 * n + i);
    float value = fma(0.0722f, b, fma(0.2126f, r, 0.7152f * g));
    return image_byte != 0 ? value * (1.0f / 255.0f) : value;
}
float st_luma(constant StructureParams& p, int i) { return st_luma_at(p.a, p.width * p.height, p.image_byte, i); }
int st_reflect(int i, int n) {
    if (i < 0)
        i = -i - 1;
    i %= 2 * n;
    return i < n ? i : 2 * n - 1 - i;
}
// st_reflect without the division when i lies inside the image.
inline int st_reflect_fast(int i, int n) { return i >= 0 && i < n ? i : st_reflect(i, n); }
bool st_interior(constant StructureParams& p, int x, int y) { return x > 0 && x < p.width - 1 && y > 0 && y < p.height - 1; }
float st_sx(int x, int y) { return float(x) * (y == 0 ? 2.0f : 1.0f) * 0.125f; }
float st_sy(int x, int y) { return float(y) * (x == 0 ? 2.0f : 1.0f) * 0.125f; }
float st_mask(constant StructureParams& p, int i) {
    if (p.c == 0)
        return 1.0f;
    return p.mask_byte != 0 ? (st_byte(p.c, i) != 0 ? 1.0f : 0.0f) : st_read(p.c, i);
}
// mode 0 keeps mask > 0; mode 1 (segment and ignore) keeps the band above 250.
inline bool st_keeps_at(ulong mask, uint mask_byte, uint mode, int i) {
    if (mask == 0)
        return true;
    if (mode == 0)
        return mask_byte != 0 ? st_byte(mask, i) != 0 : st_read(mask, i) > 0.0f;
    return mask_byte != 0 ? st_byte(mask, i) > 250u : st_read(mask, i) > (250.5f / 255.0f);
}
bool st_ridge_keeps(constant StructureParams& p, int i) { return st_keeps_at(p.e, p.mask_byte, p.reserved, i); }
bool st_invalid_response(float value) { return as_type<uint>(value) == 0xff800000u; }
// Ridge strength of one scale from its Hessian sums (hxy, hyy still without the sigma^2 normalization).
inline float st_ridge_response(float hxx, float hxy, float hyy, float sigma_sq) {
    hxy *= sigma_sq;
    hyy *= sigma_sq;
    float difference = fma(hxx, sigma_sq, -hyy);
    float trace = fma(hxx, sigma_sq, hyy);
    float disc = sqrt(max(fma(difference, difference, (4.0f * hxy) * hxy), 0.0f));
    float a = abs(0.5f * (trace + disc)), b = abs(0.5f * (trace - disc));
    float hi = max(a, b), lo = min(a, b);
    return hi * (hi - lo) / (hi + lo + 1e-6f);
}
float st_residual(constant StructureParams& p, int i) {
    int n = p.width * p.height;
    return 0.2126f * (st_read(p.a, i) - st_value(p.b, i, p.image_byte)) +
           0.7152f * (st_read(p.a, n + i) - st_value(p.b, n + i, p.image_byte)) +
           0.0722f * (st_read(p.a, 2 * n + i) - st_value(p.b, 2 * n + i, p.image_byte));
}

kernel void structure_main(constant StructureParams& p [[buffer(0)]],
                           uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint operation = kStructureOperation;
    threadgroup float3 reduction[256];
    threadgroup float count_reduction[256];
    threadgroup float residual[432], confidence[432], coefficient_x[340], coefficient_y[340];
    uint i = group * 256u + lane;
    int n = p.width * p.height;
    // Operations 0 and 1 (the band ridge passes of structure_gpu.hpp) are not used on Metal: structure_ridge below
    // computes the whole map.
    if (operation == 2 || operation == 3 || operation == 6 || operation == 7) {
        float3 value = float3(0.0f, 0.0f, 0.0f);
        float valid_count = 0.0f;
        if (operation == 2 || operation == 3)
            value = float3(0.0f, INFINITY, -INFINITY);
        if (operation == 2) {
            for (uint k = i; k < uint(n); k += p.blocks * 256u) {
                if (!st_ridge_keeps(p, int(k)))
                    continue;
                float l = st_luma(p, int(k));
                value.x += st_read(p.c, k);
                if (p.e != 0)
                    valid_count += 1.0f;
                value.y = min(value.y, l);
                value.z = max(value.z, l);
            }
        } else if (operation == 3) {
            for (uint k = lane; k < p.blocks; k += 256u) {
                value.x += st_read(p.d, k);
                if (p.e != 0)
                    valid_count += st_read(p.d, 3072 + k);
                else {
                    value.y = min(value.y, st_read(p.d, 1024 + k));
                    value.z = max(value.z, st_read(p.d, 2048 + k));
                }
            }
        } else if (operation == 6) {
            for (uint k = i; k < uint(n); k += p.blocks * 256u) {
                int x = int(k) % p.width, y = int(k) / p.width;
                bool valid = st_interior(p, x, y);
                float gx = 0, gy = 0;
                if (valid) {
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            int j = int(k) + dy * p.width + dx;
                            valid = valid && st_mask(p, j) > 0.0f;
                            float r = st_residual(p, j);
                            gx += st_sx(dx, dy) * r;
                            gy += st_sy(dx, dy) * r;
                        }
                }
                float m = valid ? st_mask(p, int(k)) : 0.0f;
                value.x += m * (sqrt(gx * gx + (0.001f * 0.001f)) + sqrt(gy * gy + (0.001f * 0.001f)) - 2.0f * 0.001f);
                value.y += m;
            }
        } else {
            for (uint k = lane; k < p.blocks; k += 256u) {
                value.x += st_read(p.d, k);
                value.y += st_read(p.d, 1024 + k);
            }
        }
        reduction[lane] = value;
        count_reduction[lane] = valid_count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Warp XOR followed by the eight warp totals, matching the CUDA reduction order.
        for (uint offset = 16; offset > 0; offset >>= 1) {
            float3 other = reduction[lane ^ offset];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            value.x += other.x;
            if (operation == 2 || operation == 3) {
                value.y = min(value.y, other.y);
                value.z = max(value.z, other.z);
            } else
                value.y += other.y;
            reduction[lane] = value;
            count_reduction[lane] += count_reduction[lane ^ offset];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        value = lane < 8 ? reduction[lane * 32] : float3(0.0f, 0.0f, 0.0f);
        valid_count = lane < 8 ? count_reduction[lane * 32] : 0.0f;
        if ((operation == 2 || operation == 3) && lane >= 8)
            value = float3(0.0f, INFINITY, -INFINITY);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        reduction[lane] = value;
        count_reduction[lane] = valid_count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 4; offset > 0; offset >>= 1) {
            float3 other = reduction[lane ^ offset];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            value.x += other.x;
            if (operation == 2 || operation == 3) {
                value.y = min(value.y, other.y);
                value.z = max(value.z, other.z);
            } else
                value.y += other.y;
            reduction[lane] = value;
            count_reduction[lane] += count_reduction[lane ^ offset];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        value = reduction[0];
        valid_count = count_reduction[0];
        if (operation == 2) {
            if (lane == 0) {
                st_write(p.d, group, value.x);
                st_write(p.d, 1024 + group, value.y);
                st_write(p.d, 2048 + group, value.z);
                if (p.e != 0)
                    st_write(p.d, 3072 + group, valid_count);
            }
        } else if (operation == 3) {
            if (i < uint(n)) {
                bool range_valid = p.e != 0 ? valid_count > 0.0f : value.z > value.y;
                float count = p.e != 0 ? valid_count : float(n);
                st_write(p.c, i, st_read(p.c, i) * (value.x > 0.0f && range_valid ? count / value.x : 0.0f));
            }
        } else if (operation == 6) {
            if (lane == 0) {
                st_write(p.d, group, value.x);
                st_write(p.d, 1024 + group, value.y);
            }
        } else if (lane == 0) {
            float scale = value.y > 0.0f ? p.gain / (2.0f * value.y) : 0.0f;
            st_write(p.e, 0, scale);
            st_write(p.e, 1, value.y);
            st_write(p.f, 0, value.x * scale);
        }
        return;
    }
    if (operation == 4) {
        if (i >= uint(n))
            return;
        int x = int(i) % p.width, y = int(i) / p.width;
        bool inside = p.valid_padding == 0 || p.width <= 10 || p.height <= 10 || (x >= 5 && x < p.width - 5 && y >= 5 && y < p.height - 5);
        float base = 1.0f;
        if (p.b != 0)
            base = p.mask_byte != 0 ? (st_byte(p.b, i) != 0 ? 1.0f : 0.0f) : st_read(p.b, i);
        st_write(p.c, i, inside ? base * (1.0f + p.gain * min(st_read(p.a, i), 4.0f)) : 0.0f);
        return;
    }
    if (operation == 5) {
        if (i < uint(n))
            st_write(p.a, i, st_read(p.a, i) * (1.0f + p.gain * clamp(st_read(p.b, i), 0.0f, 4.0f)));
        return;
    }
    if (operation == 8) {
        int tiles_x = (p.width + 31) / 32;
        int x0 = int(group) % tiles_x * 32, y0 = int(group) / tiles_x * 8;
        for (uint k = lane; k < 12u * 36u; k += 256u) {
            int y = y0 + int(k) / 36 - 2, x = x0 + int(k) % 36 - 2;
            bool inside = x >= 0 && x < p.width && y >= 0 && y < p.height;
            residual[k] = inside ? st_residual(p, y * p.width + x) : 0.0f;
            confidence[k] = inside ? st_mask(p, y * p.width + x) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = lane; k < 10u * 34u; k += 256u) {
            int ly = int(k) / 34, lx = int(k) % 34;
            bool valid = st_interior(p, x0 + lx - 1, y0 + ly - 1);
            float gx = 0, gy = 0;
            if (valid)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        int j = (ly + 1 + dy) * 36 + lx + 1 + dx;
                        valid = valid && confidence[j] > 0.0f;
                        float r = residual[j];
                        gx += st_sx(dx, dy) * r;
                        gy += st_sy(dx, dy) * r;
                    }
            float m = valid ? confidence[(ly + 1) * 36 + lx + 1] : 0.0f;
            coefficient_x[k] = m * gx / sqrt(gx * gx + (0.001f * 0.001f));
            coefficient_y[k] = m * gy / sqrt(gy * gy + (0.001f * 0.001f));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int tx = int(lane) % 32, ty = int(lane) / 32, x = x0 + tx, y = y0 + ty;
        if (x >= p.width || y >= p.height)
            return;
        int index = y * p.width + x;
        float value = 0;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                if (st_interior(p, x - dx, y - dy)) {
                    int j = (ty - dy + 1) * 34 + tx - dx + 1;
                    value += st_sx(dx, dy) * coefficient_x[j] + st_sy(dx, dy) * coefficient_y[j];
                }
        value *= st_read(p.e, 0);
        st_write(p.g, index, st_read(p.g, index) + 0.2126f * value);
        st_write(p.g, n + index, st_read(p.g, n + index) + 0.7152f * value);
        st_write(p.g, 2 * n + index, st_read(p.g, 2 * n + index) + 0.0722f * value);
    }
}

// ---- Ridge map: the four scales in one tiled pass --------------------------------------------------------------
// A 16x32 tile of output pixels loads its luma and mask keep flags, with a 10-pixel halo reflected at the image border,
// into threadgroup memory once. Each scale then runs its separable Hessian from there: horizontal sums g, d1, d2 over
// the tile rows and that scale's halo rows, then the vertical sums per output pixel, the same fma chains in the same
// order as the band passes of structure_gpu.hpp, whose per-scale filters StructureRidgeParams packs back to back. A
// pixel keeps the largest response of the scales whose window lies on kept pixels, 0 if even the smallest does not:
// scale r's window is kept when the pixel's Chebyshev distance to the nearest hidden span cell exceeds r.
// Threadgroup memory: 52x36 floats and bytes for the span, 3x52x16 floats for the horizontal sums: 19,344 bytes.
constant constexpr int kRidgeTileWidth = 16;
constant constexpr int kRidgeTileHeight = 32;
constant constexpr int kRidgeHalo = 10; // the largest radius, sigma 2.5
constant constexpr int kRidgeSpanWidth = kRidgeTileWidth + 2 * kRidgeHalo;
constant constexpr int kRidgeSpanHeight = kRidgeTileHeight + 2 * kRidgeHalo;
constant constexpr uint kRidgeThreads = kRidgeTileWidth * kRidgeTileHeight; // one thread per output pixel
constant constexpr int kRidgeTaps = 7 + 11 + 15 + 21;

struct StructureRidgeParams {
    ulong image, mask, output;
    int width, height;
    uint image_byte, mask_byte, mask_mode, reserved;
    float sigma_sq[4];
    float filter_g[kRidgeTaps], filter_d1[kRidgeTaps], filter_d2[kRidgeTaps];
};

// One scale: Radius, and Offset of its filters in StructureRidgeParams. Column pairs share their taps' loads.
template <int Radius, int Offset>
static void st_ridge_scale(constant StructureRidgeParams& p, const float sigma_sq, threadgroup const float* luma,
                           threadgroup float* horizontal_g, threadgroup float* horizontal_d1,
                           threadgroup float* horizontal_d2, const uint2 local, const uint rank, const bool kept,
                           const bool first, thread float& best) {
    constexpr int taps = 2 * Radius + 1;
    constexpr uint pairs = uint(kRidgeTileHeight + 2 * Radius) * uint(kRidgeTileWidth / 2);
    if (rank < pairs) {
        const int row = int(rank) / (kRidgeTileWidth / 2), column = 2 * (int(rank) % (kRidgeTileWidth / 2));
        const int start = (kRidgeHalo - Radius + row) * kRidgeSpanWidth + column + kRidgeHalo - Radius;
        float g0 = 0, d10 = 0, d20 = 0, g1 = 0, d11 = 0, d21 = 0;
        float v1 = luma[start];
        for (int j = 0; j < taps; ++j) {
            const float v0 = v1;
            v1 = luma[start + j + 1];
            g0 = fma(v0, p.filter_g[Offset + j], g0);
            d10 = fma(v0, p.filter_d1[Offset + j], d10);
            d20 = fma(v0, p.filter_d2[Offset + j], d20);
            g1 = fma(v1, p.filter_g[Offset + j], g1);
            d11 = fma(v1, p.filter_d1[Offset + j], d11);
            d21 = fma(v1, p.filter_d2[Offset + j], d21);
        }
        const int cell = row * kRidgeTileWidth + column;
        horizontal_g[cell] = g0;
        horizontal_d1[cell] = d10;
        horizontal_d2[cell] = d20;
        horizontal_g[cell + 1] = g1;
        horizontal_d1[cell + 1] = d11;
        horizontal_d2[cell + 1] = d21;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kept) {
        const int column = int(local.y) * kRidgeTileWidth + int(local.x);
        float hxx = 0, hxy = 0, hyy = 0;
        for (int j = 0; j < taps; ++j) {
            const int cell = column + j * kRidgeTileWidth;
            hxx = fma(horizontal_d2[cell], p.filter_g[Offset + j], hxx);
            hxy = fma(horizontal_d1[cell], p.filter_d1[Offset + j], hxy);
            hyy = fma(horizontal_g[cell], p.filter_d2[Offset + j], hyy);
        }
        const float response = st_ridge_response(hxx, hxy, hyy, sigma_sq);
        best = first ? response : max(best, response);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void structure_ridge(constant StructureRidgeParams& p [[buffer(0)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint2 local [[thread_position_in_threadgroup]],
                            uint rank [[thread_index_in_threadgroup]]) {
    threadgroup float luma[kRidgeSpanHeight * kRidgeSpanWidth];
    threadgroup uchar hidden_distance[kRidgeSpanHeight * kRidgeSpanWidth];
    threadgroup float horizontal_g[kRidgeSpanHeight * kRidgeTileWidth];
    threadgroup float horizontal_d1[kRidgeSpanHeight * kRidgeTileWidth];
    threadgroup float horizontal_d2[kRidgeSpanHeight * kRidgeTileWidth];
    constexpr uchar kFar = kRidgeHalo + 1;
    const int x0 = int(group.x) * kRidgeTileWidth, y0 = int(group.y) * kRidgeTileHeight;
    const int n = p.width * p.height;
    for (uint cell = rank; cell < uint(kRidgeSpanHeight * kRidgeSpanWidth); cell += kRidgeThreads) {
        const int row = st_reflect_fast(y0 - kRidgeHalo + int(cell) / kRidgeSpanWidth, p.height);
        const int column = st_reflect_fast(x0 - kRidgeHalo + int(cell) % kRidgeSpanWidth, p.width);
        const int source = row * p.width + column;
        luma[cell] = st_luma_at(p.image, n, p.image_byte, source);
        hidden_distance[cell] = st_keeps_at(p.mask, p.mask_byte, p.mask_mode, source) ? kFar : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Each span row's distance to its nearest hidden cell, capped at kFar: one thread scans a row both ways.
    if (p.mask != 0 && rank < uint(kRidgeSpanHeight)) {
        threadgroup uchar* row = hidden_distance + rank * kRidgeSpanWidth;
        uchar distance = kFar;
        for (int c = 0; c < kRidgeSpanWidth; ++c) {
            distance = row[c] == 0 ? 0 : min(uchar(distance + 1), kFar);
            row[c] = distance;
        }
        distance = kFar;
        for (int c = kRidgeSpanWidth - 1; c >= 0; --c) {
            distance = row[c] == 0 ? 0 : min(uchar(distance + 1), kFar);
            row[c] = min(row[c], distance);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int x = x0 + int(local.x), y = y0 + int(local.y);
    const bool inside = x < p.width && y < p.height;
    // Chebyshev distance of the output pixel to the nearest hidden span cell, capped at kFar.
    int distance = kFar;
    for (int j = -kRidgeHalo; j <= kRidgeHalo; ++j) {
        const int row = int(local.y) + kRidgeHalo + j;
        distance = min(distance, max(abs(j), int(hidden_distance[row * kRidgeSpanWidth + int(local.x) + kRidgeHalo])));
    }
    float best = 0.0f;
    st_ridge_scale<3, 0>(p, p.sigma_sq[0], luma, horizontal_g, horizontal_d1, horizontal_d2, local, rank,
                         inside && distance > 3, true, best);
    st_ridge_scale<5, 7>(p, p.sigma_sq[1], luma, horizontal_g, horizontal_d1, horizontal_d2, local, rank,
                         inside && distance > 5, false, best);
    st_ridge_scale<7, 18>(p, p.sigma_sq[2], luma, horizontal_g, horizontal_d1, horizontal_d2, local, rank,
                          inside && distance > 7, false, best);
    st_ridge_scale<10, 33>(p, p.sigma_sq[3], luma, horizontal_g, horizontal_d1, horizontal_d2, local, rank,
                           inside && distance > 10, false, best);
    if (inside)
        st_write(p.output, y * p.width + x, best);
}
