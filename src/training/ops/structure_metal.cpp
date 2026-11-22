/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "metal_families.hpp"
#include "metal_kernels.hpp"
#include "structure_gpu.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>
namespace lfs::training {
    namespace {
        struct Backend {
            static uint64_t address(const core::Tensor& tensor) { return metal::address(tensor); }
            static void launch(uint32_t operation, const structure_detail::Parameters& p,
                               std::initializer_list<const core::Tensor*> reads,
                               std::initializer_list<const core::Tensor*> writes, size_t items) {
                std::vector<const core::Tensor*> uses;
                for (auto* t : reads)
                    if (t->is_valid())
                        uses.push_back(t);
                for (auto* t : writes)
                    if (t->is_valid())
                        uses.push_back(t);
                const std::pair<uint32_t, uint32_t> constant{2, operation};
                metal::kernels().launch({.function = "structure_main", .params = std::as_bytes(std::span(&p, 1)), .uses = uses, .groups = {static_cast<uint32_t>((items + 255) / 256), 1, 1}, .group = {256, 1, 1}, .constants = std::span(&constant, 1)});
            }
        };
        using Shared = structure_detail::Implementation<Backend>;

        // StructureRidgeParams in structure.metal: the filters of the four scales back to back. The kernel unrolls
        // each scale for these radii.
        constexpr std::array<double, 4> kRidgeSigmas{0.8, 1.2, 1.8, 2.5};
        constexpr std::array<uint32_t, 4> kRidgeRadii{3, 5, 7, 10};
        constexpr size_t kRidgeTaps = 7 + 11 + 15 + 21;
        constexpr uint32_t kRidgeTileWidth = 16, kRidgeTileHeight = 32;
        // Operations 2 and 3 of structure_main reduce over at most kRidgeGroups groups and keep their partial sums in
        // runs of kRidgeGroups: the response sum, the luma minimum and maximum, and with a mask the kept-pixel count,
        // the layout RidgeWorkspace::ensure_size allocates for the template.
        constexpr size_t kRidgeGroups = 1024;
        struct RidgeParams {
            uint64_t image = 0, mask = 0, output = 0;
            int32_t width = 0, height = 0;
            uint32_t image_byte = 0, mask_byte = 0, mask_mode = 0, reserved = 0;
            float sigma_sq[4]{};
            float filter_g[kRidgeTaps]{}, filter_d1[kRidgeTaps]{}, filter_d2[kRidgeTaps]{};
        };
        static_assert(offsetof(RidgeParams, width) == 24 && offsetof(RidgeParams, sigma_sq) == 48 &&
                      offsetof(RidgeParams, filter_g) == 64 && sizeof(RidgeParams) == 712);

        // The multi-scale ridge map of the template's ridge pass in one tiled kernel instead of two passes per scale
        // over row bands, so it needs no band buffer; its reduction and normalization are the template's.
        void ridge(const core::Tensor& image, core::Tensor& output, kernels::RidgeWorkspace& workspace,
                   const core::Tensor& photometric_mask, const gpu_ops::MaskPhotoMode mask_mode) {
            LFS_ASSERT(image.device() == core::Device::GPU && image.ndim() == 3 && image.shape()[0] == 3);
            LFS_ASSERT(image.is_contiguous() && image.shape()[1] > 0 && image.shape()[2] > 0);
            LFS_ASSERT(image.dtype() == core::DataType::Float32 || image.dtype() == core::DataType::UInt8);
            const auto h = image.shape()[1], w = image.shape()[2];
            const core::Tensor mask = photometric_mask.is_valid() && photometric_mask.ndim() == 3
                                          ? photometric_mask.squeeze(0)
                                          : photometric_mask;
            const bool masked = mask.is_valid() && mask.numel() > 0;
            if (masked) {
                LFS_ASSERT((mask.device() == core::Device::GPU && mask.shape() == core::TensorShape{h, w}));
                LFS_ASSERT(mask.dtype() == core::DataType::Float32 || mask.dtype() == core::DataType::UInt8 || mask.dtype() == core::DataType::Bool);
            }
            LFS_ASSERT((output.shape() == core::TensorShape{h, w}));
            LFS_ASSERT(output.dtype() == core::DataType::Float32 && output.is_contiguous());
            const size_t reduction = (masked ? 4 : 3) * kRidgeGroups;
            if (!workspace.reduction.is_valid() || workspace.reduction.numel() < reduction)
                workspace.reduction = core::Tensor::empty({reduction}, core::Device::GPU);

            RidgeParams p;
            p.image = Backend::address(image);
            p.mask = masked ? Backend::address(mask) : 0;
            p.output = Backend::address(output);
            p.width = static_cast<int32_t>(w);
            p.height = static_cast<int32_t>(h);
            p.image_byte = image.dtype() == core::DataType::UInt8;
            p.mask_byte = masked && mask.dtype() != core::DataType::Float32;
            p.mask_mode = static_cast<uint32_t>(mask_mode);
            size_t offset = 0;
            for (size_t scale = 0; scale < kRidgeSigmas.size(); ++scale) {
                const double sigma = kRidgeSigmas[scale];
                const auto radius = static_cast<uint32_t>(4.0 * sigma + 0.5);
                LFS_ASSERT(radius == kRidgeRadii[scale]);
                p.sigma_sq[scale] = static_cast<float>(sigma * sigma);
                double sum = 0;
                for (int k = -int(radius); k <= int(radius); ++k)
                    sum += std::exp(-0.5 * k * k / (sigma * sigma));
                for (int k = -int(radius); k <= int(radius); ++k) {
                    const double g = std::exp(-0.5 * k * k / (sigma * sigma)) / sum;
                    const size_t j = offset + static_cast<size_t>(k + int(radius));
                    p.filter_g[j] = static_cast<float>(g);
                    p.filter_d1[j] = static_cast<float>(k * g / (sigma * sigma));
                    p.filter_d2[j] = static_cast<float>((k * k / (sigma * sigma) - 1.0) * g / (sigma * sigma));
                }
                offset += 2 * radius + 1;
            }
            LFS_ASSERT(offset == kRidgeTaps);
            metal::launch_2d("structure_ridge", p, {&image, &mask, &output},
                             core::GpuKernelModule::groups_for(w, kRidgeTileWidth),
                             core::GpuKernelModule::groups_for(h, kRidgeTileHeight), kRidgeTileWidth, kRidgeTileHeight);

            structure_detail::Parameters q;
            q.a = p.image;
            q.c = p.output;
            q.d = Backend::address(workspace.reduction);
            q.e = p.mask;
            q.width = p.width;
            q.height = p.height;
            q.image_byte = p.image_byte;
            q.mask_byte = p.mask_byte;
            q.reserved = p.mask_mode;
            q.blocks = static_cast<uint32_t>(std::min((h * w + 255) / 256, kRidgeGroups));
            Backend::launch(2, q, {&image, &mask, &output}, {&workspace.reduction}, size_t(q.blocks) * 256);
            Backend::launch(3, q, {&workspace.reduction, &output}, {&output}, h * w);
        }
    } // namespace
    const gpu_ops::StructureOps& metal_structure_ops() {
        static const gpu_ops::StructureOps result{ridge, Shared::photometric_weight, Shared::densification_weight,
                                                  Shared::gradient};
        return result;
    }
} // namespace lfs::training
