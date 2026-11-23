/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal BilateralOps; see ops/bilateral_cuda.cpp for the reference behaviour.

#include "metal_families.hpp"
#include "metal_kernels.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <stdexcept>

namespace lfs::training {
    namespace {
        using lfs::gpu_ops::AdamUpdateParams;
        using lfs::gpu_ops::GridSliceParams;
        using lfs::gpu_ops::GridTransform;
        using lfs::gpu_ops::In;
        using lfs::gpu_ops::Layout;
        using lfs::gpu_ops::Out;
        using lfs::gpu_ops::PPISPInputs;
        using lfs::gpu_ops::PPISPRegion;
        namespace mk = metal;

        constexpr uint32_t kTile = 32;
        constexpr uint32_t kTileThreads = 16;

        // ppisp.metal's PpispForwardParams, which the slice kernels read with kBilateralPpispInput.
        struct PpispParams {
            uint64_t exposure, vignetting, color, crf, rgb_in, rgb_out;
            int32_t height, width, y_offset, full_height, camera_index, frame_index, x_offset, full_width;
        };

        struct SliceParams {
            uint64_t grid, rgb, grad_output, offset, output, grad_grid;
            int32_t L, H, W, h, w;
            uint32_t chw, exposure_chroma;
            PpispParams ppisp;
        };
        static_assert(offsetof(SliceParams, ppisp) == 80 && sizeof(SliceParams) == 160);

        // Function constant kBilateralPpispInput of the slice kernels.
        constexpr uint32_t kPpispInputConstant = 80;

        SliceParams slice_params(In grid, In rgb, In shared_offset, const GridSliceParams& params) {
            const auto& shape = grid.shape();
            const bool chw = params.layout == Layout::CHW;
            if (shape.rank() != 5 || rgb.ndim() != 3 || rgb.shape()[chw ? 0 : 2] != 3)
                throw std::invalid_argument(std::format("bilateral slice needs a [N,C,L,H,W] grid and 3-channel {} rgb, got {} and {}",
                                                        chw ? "CHW" : "HWC", shape.str(), rgb.shape().str()));
            const size_t channels = params.transform == GridTransform::ExposureChroma ? 9 : 12;
            if (shape[1] < channels)
                throw std::invalid_argument(std::format("bilateral slice needs {} grid channels, got {}", channels, shape[1]));
            return {.grid = mk::address(grid),
                    .rgb = mk::address(rgb),
                    .offset = mk::address(shared_offset),
                    .L = static_cast<int32_t>(shape[2]),
                    .H = static_cast<int32_t>(shape[3]),
                    .W = static_cast<int32_t>(shape[4]),
                    .h = static_cast<int32_t>(rgb.shape()[chw ? 1 : 0]),
                    .w = static_cast<int32_t>(rgb.shape()[chw ? 2 : 1]),
                    .chw = chw ? 1u : 0u,
                    .exposure_chroma = params.transform == GridTransform::ExposureChroma ? 1u : 0u};
        }

        void slice_forward(In grid, In rgb, In shared_offset, Out output, const GridSliceParams& params) {
            auto p = slice_params(grid, rgb, shared_offset, params);
            p.output = mk::address(output);
            mk::launch_items("bilateral_slice_forward", p, {&grid, &rgb, &shared_offset, &output},
                             static_cast<size_t>(p.h) * static_cast<size_t>(p.w));
        }

        // The slice of PPISP's forward of the CHW raw image: the kernels compute it per pixel.
        SliceParams ppisp_slice_params(In grid, In raw, In shared_offset, const PPISPInputs& ppisp,
                                       const PPISPRegion& region, const GridSliceParams& params) {
            if (params.layout != Layout::CHW)
                throw std::invalid_argument("bilateral slice of a PPISP forward needs a CHW image");
            auto p = slice_params(grid, raw, shared_offset, params);
            const int full_width = region.full_width > 0 ? region.full_width : p.w;
            if (region.x_offset < 0 || region.x_offset + p.w > full_width || region.y_offset < 0 ||
                region.y_offset + p.h > region.full_height)
                throw std::invalid_argument(std::format("PPISP region of {}x{} at ({}, {}) exceeds {}x{}", p.w, p.h,
                                                        region.x_offset, region.y_offset, full_width,
                                                        region.full_height));
            p.ppisp = {.exposure = mk::address(ppisp.exposure),
                       .vignetting = mk::address(ppisp.vignetting),
                       .color = mk::address(ppisp.color),
                       .crf = mk::address(ppisp.crf),
                       .rgb_in = mk::address(raw),
                       .height = p.h,
                       .width = p.w,
                       .y_offset = region.y_offset,
                       .full_height = region.full_height,
                       .camera_index = region.camera_index,
                       .frame_index = region.frame_index,
                       .x_offset = region.x_offset,
                       .full_width = full_width};
            return p;
        }

        void slice_forward_ppisp(In grid, In raw, In shared_offset, const PPISPInputs& ppisp, const PPISPRegion& region,
                                 Out output, const GridSliceParams& params) {
            auto p = ppisp_slice_params(grid, raw, shared_offset, ppisp, region, params);
            p.output = mk::address(output);
            mk::launch_items("bilateral_slice_forward", p,
                             {&grid, &raw, &shared_offset, &ppisp.exposure, &ppisp.vignetting, &ppisp.color, &ppisp.crf,
                              &output},
                             static_cast<size_t>(p.h) * static_cast<size_t>(p.w), {{kPpispInputConstant, 1u}});
        }

        // grad_grid accumulates, as the CUDA atomics do.
        void slice_backward(In grid, In rgb, In grad_output, In shared_offset, Out grad_grid, Out grad_rgb,
                            const GridSliceParams& params) {
            auto p = slice_params(grid, rgb, shared_offset, params);
            p.grad_output = mk::address(grad_output);
            p.output = mk::address(grad_rgb);
            p.grad_grid = mk::address(grad_grid);
            mk::launch_items("bilateral_slice_backward_rgb", p, {&grid, &rgb, &grad_output, &shared_offset, &grad_rgb},
                             static_cast<size_t>(p.h) * static_cast<size_t>(p.w));
            mk::launch_2d("bilateral_slice_backward_grid", p, {&grid, &rgb, &grad_output, &shared_offset, &grad_grid},
                          core::GpuKernelModule::groups_for(p.w, kTile), core::GpuKernelModule::groups_for(p.h, kTile),
                          kTileThreads, kTileThreads);
        }

        struct TvParams {
            uint64_t grids, output, partials;
            int32_t N, C, L, H, W, norm_n, partial_count;
            float grad_output;
        };

        TvParams tv_params(In grids, const int norm_n) {
            const auto& s = grids.shape();
            if (s.rank() != 5)
                throw std::invalid_argument(std::format("bilateral TV needs a [N,C,L,H,W] grid, got {}", s.str()));
            const int n = static_cast<int>(s[0]);
            return {.grids = mk::address(grids),
                    .N = n,
                    .C = static_cast<int32_t>(s[1]),
                    .L = static_cast<int32_t>(s[2]),
                    .H = static_cast<int32_t>(s[3]),
                    .W = static_cast<int32_t>(s[4]),
                    .norm_n = norm_n > 0 ? norm_n : n};
        }

        void tv_forward(In grids, Out loss, Out reduction_temp, const int norm_n) {
            auto p = tv_params(grids, norm_n);
            const size_t cells = static_cast<size_t>(p.N) * p.L * p.H * p.W;
            const uint32_t groups = std::min<uint32_t>(core::GpuKernelModule::groups_for(cells, mk::kGroupWidth), 2048);
            if (reduction_temp.numel() < groups)
                throw std::invalid_argument(std::format("bilateral TV needs {} reduction slots, got {}", groups, reduction_temp.numel()));
            p.output = mk::address(reduction_temp);
            mk::launch("bilateral_tv_partials", p, {&grids, &reduction_temp}, groups);
            p.output = mk::address(loss);
            p.partials = mk::address(reduction_temp);
            p.partial_count = static_cast<int32_t>(groups);
            mk::launch("bilateral_tv_total", p, {&reduction_temp, &loss}, 1);
        }

        void tv_backward(In grids, Out gradients, const float grad_loss, const int norm_n) {
            auto p = tv_params(grids, norm_n);
            p.output = mk::address(gradients);
            p.grad_output = grad_loss;
            mk::launch_items("bilateral_tv_backward", p, {&grids, &gradients},
                             static_cast<size_t>(p.N) * p.L * p.H * p.W);
        }

        struct ProjectParams {
            uint64_t grids, mean, identity;
            int32_t total, C, spatial, per_image;
        };

        void project_mean(Out grids, In mean, In identity, const int per_image) {
            const auto& s = grids.shape();
            const ProjectParams p{mk::address(grids), mk::address(mean), mk::address(identity),
                                  static_cast<int32_t>(grids.numel()), static_cast<int32_t>(s[1]),
                                  static_cast<int32_t>(s[2] * s[3] * s[4]), per_image};
            mk::launch_items("bilateral_project_mean", p, {&grids, &mean, &identity}, grids.numel());
        }

        struct OffsetParams {
            uint64_t channel_sum, shared_offset, identity, old_mean, new_mean;
            int32_t C;
            float spatial, inv_n_spatial;
        };

        void update_offset(Out channel_sum, Out shared_offset, In identity, In old_mean, In new_mean,
                           const float spatial, const float inv_n_spatial) {
            const OffsetParams p{mk::address(channel_sum), mk::address(shared_offset), mk::address(identity),
                                 mk::address(old_mean), mk::address(new_mean),
                                 static_cast<int32_t>(channel_sum.numel()), spatial, inv_n_spatial};
            mk::launch_items("bilateral_update_offset", p, {&channel_sum, &shared_offset, &identity, &old_mean, &new_mean},
                             channel_sum.numel());
        }

        struct AdamParams {
            uint64_t grid, moment1, moment2, gradient;
            uint32_t count;
            float lr, beta1, beta2, bc1_rcp, bc2_sqrt_rcp, eps;
        };

        void adam(Out grid, Out moment1, Out moment2, In gradient, const AdamUpdateParams& a) {
            const AdamParams p{mk::address(grid), mk::address(moment1), mk::address(moment2), mk::address(gradient),
                               mk::count32(grid.numel(), "bilateral grid"), a.lr, a.beta1, a.beta2, a.bc1_rcp, a.bc2_sqrt_rcp, a.eps};
            mk::launch_items("bilateral_adam", p, {&grid, &moment1, &moment2, &gradient}, grid.numel());
        }

        struct ScaleParams {
            uint64_t moment1, moment2;
            uint32_t count;
            float scale1, scale2;
        };

        void scale_moments(Out moment1, Out moment2, const float scale1, const float scale2) {
            if (moment1.numel() == 0)
                return;
            const ScaleParams p{mk::address(moment1), mk::address(moment2), mk::count32(moment1.numel(), "bilateral moments"),
                                scale1, scale2};
            mk::launch_items("bilateral_scale_moments", p, {&moment1, &moment2}, moment1.numel());
        }

        core::Tensor flat_range(Out tensor, const size_t offset, const size_t elements) {
            return tensor.flatten().slice(0, offset, offset + elements);
        }

        void upload_slice(Out host, Out device, const size_t host_offset, const size_t device_offset, const size_t elements) {
            flat_range(device, device_offset, elements).copy_from(flat_range(host, host_offset, elements));
        }

        void download_slice(Out host, Out device, const size_t host_offset, const size_t device_offset, const size_t elements) {
            flat_range(host, host_offset, elements).copy_from(flat_range(device, device_offset, elements));
        }
    } // namespace

    const lfs::gpu_ops::BilateralOps& metal_bilateral_ops() {
        static const lfs::gpu_ops::BilateralOps ops{
            .slice_forward = slice_forward,
            .slice_backward = slice_backward,
            .tv_forward = tv_forward,
            .tv_backward = tv_backward,
            .project_mean = project_mean,
            .update_offset = update_offset,
            .adam = adam,
            .scale_moments = scale_moments,
            .upload_slice = upload_slice,
            .download_slice = download_slice,
            .slice_forward_ppisp = slice_forward_ppisp,
        };
        return ops;
    }
} // namespace lfs::training
