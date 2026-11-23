/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "lfs/training/ops/ppisp.hpp"
#include "lfs/training/ops/types.hpp"

namespace lfs::gpu_ops {
    enum class Layout { HW,
                        HWC,
                        CHW };

    enum class GridTransform { Affine,
                               ExposureChroma };
    struct GridSliceParams {
        Layout layout = Layout::HWC;
        GridTransform transform = GridTransform::Affine;
        bool warp_aggregate = true;
    };
    struct AdamUpdateParams {
        float lr, beta1, beta2, bc1_rcp, bc2_sqrt_rcp, eps;
    };
    // Grid bindings have shape [N, C, L, H, W], including resident slice views.
    struct BilateralOps {
        void (*slice_forward)(In grid, In rgb, In shared_offset, Out output, const GridSliceParams&);
        void (*slice_backward)(In grid, In rgb, In grad_output, In shared_offset,
                               Out grad_grid, Out grad_rgb, const GridSliceParams&);
        void (*tv_forward)(In grids, Out loss, Out reduction_temp, int norm_n);
        void (*tv_backward)(In grids, Out gradients, float grad_loss, int norm_n);
        void (*project_mean)(Out grids, In mean, In identity, int per_image);
        void (*update_offset)(Out channel_sum, Out shared_offset, In identity,
                              In old_mean, In new_mean, float spatial, float inv_n_spatial);
        void (*adam)(Out grid, Out moment1, Out moment2, In gradient, const AdamUpdateParams&);
        void (*scale_moments)(Out moment1, Out moment2, float scale1, float scale2);
        void (*upload_slice)(Out host, Out device, size_t host_offset, size_t device_offset, size_t elements);
        void (*download_slice)(Out host, Out device, size_t host_offset, size_t device_offset, size_t elements);
        // slice_forward may write output over rgb, and slice_backward grad_rgb over grad_output.
        bool in_place = false;
        // Null unless the backend has it: slice_forward of the PPISP forward of a CHW raw image (PPISPOps::forward
        // with these inputs and region), clamped to [0, 1], in one pass.
        void (*slice_forward_ppisp)(In grid, In raw, In shared_offset, const PPISPInputs& ppisp, const PPISPRegion& region,
                                    Out output, const GridSliceParams&) = nullptr;
    };
} // namespace lfs::gpu_ops
