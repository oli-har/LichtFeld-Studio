/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// MrnfGrowthOps through one Slang program (mrnf_growth.slang) for the tensor backends that run
// GpuKernelModule programs; see ops/mrnf_cuda.cpp for the reference behaviour.

#include "core/assert.hpp"
#include "core/error.hpp"
#include "core/gpu_kernel_module.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/mrnf.hpp"
#include "mrnf_growth_program.hpp"

#include <array>
#include <cstddef>
#include <format>
#include <limits>

namespace lfs::training {
    namespace {
        using core::DataType;
        using core::GpuKernelModule;
        using gpu_ops::Tensor;

        // Parameters in mrnf_growth.slang: nine tensor pointers, then the scalars.
        struct Parameters {
            uint64_t pointers[9]{};
            uint32_t count = 0;
            float x = 0, y = 0, z = 0;
        };
        static_assert(sizeof(Parameters) == 88 && offsetof(Parameters, count) == 72);
        enum Slot { Error,
                    Info,
                    Views,
                    Step,
                    Peak,
                    Renders,
                    Opacity,
                    Scales,
                    Result,
                    SlotCount };
        struct Tensors {
            std::array<const Tensor*, SlotCount> slots{};
            uint32_t writes = 0; // bit per written slot

            void read(Slot s, const Tensor& t) { slots[s] = t.is_valid() && t.numel() > 0 ? &t : nullptr; }
            void write(Slot s, const Tensor& t) {
                read(s, t);
                writes |= 1u << s;
            }
        };

        void run(const char* function, const Tensor& anchor, const Parameters& p, const Tensors& tensors) {
            if (p.count == 0)
                return;
            const auto backend = core::gpu_backend_of(anchor).value_or(core::default_gpu_backend());
            const core::GpuBackendScope scope(backend);
            auto program = GpuKernelModule::load(mrnf_growth_program_entries(), backend);
            if (!program)
                throw lfs::Exception(std::move(program).error());
            std::array<GpuKernelModule::Binding, SlotCount> bindings;
            for (uint32_t s = 0; s < SlotCount; ++s)
                bindings[s] = {s * 8, tensors.slots[s],
                               tensors.writes & (1u << s) ? GpuKernelModule::Access::ReadWrite : GpuKernelModule::Access::Read};
            auto result = (*program)->dispatch({.function = function,
                                                .arguments = {std::as_bytes(std::span(&p, 1)), bindings},
                                                .groups = {GpuKernelModule::groups_for(p.count, 256), 1, 1},
                                                .group = {256, 1, 1}});
            if (!result)
                throw lfs::Exception(std::move(result).error());
        }

        uint32_t rows32(const size_t n, const char* const what) {
            LFS_ASSERT_MSG(n <= std::numeric_limits<uint32_t>::max(),
                           std::format("MRNF {} rows exceed 32-bit indexing (rows={})", what, n));
            return static_cast<uint32_t>(n);
        }

        void require_rows(const Tensor& t, const size_t n, const char* const name) {
            LFS_ASSERT_MSG(t.is_valid() && t.numel() >= n && t.dtype() == DataType::Float32 && t.is_contiguous(),
                           std::format("MRNF {} must be contiguous float with {} values (shape={})", name, n,
                                       t.is_valid() ? t.shape().str() : "none"));
        }

        void fold_view(Tensor& error, Tensor& info, Tensor& views) {
            const size_t n = error.numel();
            if (n == 0)
                return;
            require_rows(error, n, "refine weights");
            require_rows(info, 4 * n, "densification rows");
            if (views.is_valid())
                require_rows(views, n, "coverage views");
            Tensors t;
            t.write(Error, error);
            t.write(Info, info);
            t.write(Views, views);
            run("fold_view", error, {.count = rows32(n, "fold view")}, t);
        }

        void mean_view(Tensor& error, const Tensor& info) {
            const size_t n = error.numel();
            if (n == 0)
                return;
            require_rows(error, n, "refine weights");
            require_rows(info, 4 * n, "densification rows");
            Tensors t;
            t.write(Error, error);
            t.read(Info, info);
            run("mean_view", error, {.count = rows32(n, "mean view")}, t);
        }

        void fold_need(Tensor& error, const float scale, const float maximum, const float power) {
            const size_t n = error.numel();
            if (n == 0)
                return;
            require_rows(error, n, "refine weights");
            LFS_ASSERT_MSG(scale > 0.0f, "need error scale must be positive");
            Tensors t;
            t.write(Error, error);
            run("fold_need", error, {.count = rows32(n, "fold need"), .x = scale, .y = maximum, .z = power}, t);
        }

        void blend_peak(Tensor& step, Tensor& peak, Tensor& renders, const float decay) {
            const size_t n = peak.numel();
            if (n == 0)
                return;
            require_rows(step, n, "step peak");
            require_rows(peak, n, "blend peak");
            require_rows(renders, n, "renders");
            Tensors t;
            t.write(Step, step);
            t.write(Peak, peak);
            t.write(Renders, renders);
            run("blend_peak", peak, {.count = rows32(n, "blend peak"), .x = decay}, t);
        }

        void covered_factor(const Tensor& peak, const Tensor& renders, const Tensor& opacity, const Tensor& scales,
                            const float grace, const float keep, const float floor, Tensor& factor) {
            const size_t n = peak.numel();
            if (n == 0)
                return;
            require_rows(peak, n, "blend peak");
            require_rows(renders, n, "renders");
            require_rows(opacity, n, "raw opacities");
            require_rows(scales, 3 * n, "raw scales");
            require_rows(factor, n, "covered factor");
            Tensors t;
            t.read(Peak, peak);
            t.read(Renders, renders);
            t.read(Opacity, opacity);
            t.read(Scales, scales);
            t.write(Result, factor);
            run("covered_factor", peak, {.count = rows32(n, "covered factor"), .x = grace, .y = keep, .z = floor}, t);
        }

        void retire_key(const Tensor& peak, const Tensor& opacity, const float dead_peak, Tensor& key) {
            const size_t n = peak.numel();
            if (n == 0)
                return;
            require_rows(peak, n, "blend peak");
            require_rows(opacity, n, "raw opacities");
            require_rows(key, n, "retire key");
            Tensors t;
            t.read(Peak, peak);
            t.read(Opacity, opacity);
            t.write(Result, key);
            run("retire_key", peak, {.count = rows32(n, "retire key"), .x = dead_peak}, t);
        }

        // The order of CUDA's one radix sort of lowest_eligible_keys from two stable tensor sorts: eligible rows by
        // ascending value (-0 equal to +0), then eligible NaNs, then ineligible rows, ties by row. The first sort
        // orders the eligible values, the second the three groups.
        void lowest_rows(const Tensor& score, const Tensor& eligible, Tensor& rows) {
            const size_t count = rows.numel();
            if (count == 0)
                return;
            const size_t n = score.numel();
            LFS_ASSERT_MSG(count <= n && eligible.numel() == n && n <= static_cast<size_t>(std::numeric_limits<int>::max()),
                           std::format("Invalid MRNF selection size: {} of {} rows, {} eligibility flags", count, n,
                                       eligible.numel()));
            const auto values = score.reshape({-1});
            const auto chosen = eligible.reshape({-1}).to(DataType::Bool);
            const auto nan = values.isnan();
            const auto number = chosen.logical_and(nan.logical_not());
            const auto group = chosen.logical_not().to(DataType::Float32).mul(2.0f).add(chosen.logical_and(nan).to(DataType::Float32));
            const auto by_value = values.masked_fill(number.logical_not(), 0.0f).sort(0, /*descending=*/false).second;
            const auto by_group = group.index_select(0, by_value).sort(0, /*descending=*/false).second;
            rows.copy_from(by_value.index_select(0, by_group).slice(0, 0, count));
        }
    } // namespace

    const gpu_ops::MrnfGrowthOps& portable_mrnf_growth_ops() {
        static const gpu_ops::MrnfGrowthOps ops{
            .fold_view = fold_view,
            .mean_view = mean_view,
            .fold_need = fold_need,
            .blend_peak = blend_peak,
            .covered_factor = covered_factor,
            .retire_key = retire_key,
            .lowest_rows = lowest_rows,
        };
        return ops;
    }
} // namespace lfs::training
