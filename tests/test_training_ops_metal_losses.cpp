/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The Metal photometric, mask, extra-loss and geometry ops against CPU
// transliterations of their CUDA kernels (ssim.cu, ssim_reduction.cu,
// l1_loss.cu, mask_preprocess.cu, regularization.cu,
// depth_loss.cu, normal_loss.cu,
// normal_consistency_loss.cu), evaluated in double. SSIM partials are rounded
// to half as the CUDA workspace stores them. A finite-difference check covers
// the SSIM gradient itself.

#include "core/gpu_backend_fwd.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/ops/structure.hpp"
#ifdef LFS_TENSOR_METAL
#include "ops/metal_kernels.hpp"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;
    namespace ops = lfs::gpu_ops;
    namespace k = lfs::training::kernels;
    using Field = std::vector<double>;

    // CPU references of the CUDA loss kernels, run on every portable backend.
    class PortableLosses : public testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(GetParam()))
                GTEST_SKIP() << "No " << lfs::core::gpu_backend_name(GetParam()) << " device";
            scope_.emplace(GetParam());
            table_ = &lfs::training::training_ops(GetParam());
            if (!table_->photometric || !table_->masks || !table_->extra_loss || !table_->geometry)
                GTEST_SKIP() << "Loss families are not filled";
        }

        std::optional<lfs::core::GpuBackendScope> scope_;
        const lfs::training::TrainingOps* table_ = nullptr;
    };

    // Deterministic uniform values in [lo, hi).
    std::vector<float> uniform(const size_t count, const uint32_t seed, const float lo = 0.f, const float hi = 1.f) {
        std::vector<float> values(count);
        uint32_t state = seed * 747796405u + 2891336453u;
        for (float& value : values) {
            state = state * 1664525u + 1013904223u;
            value = lo + (hi - lo) * static_cast<float>(state >> 8) / 16777216.f;
        }
        return values;
    }

    Tensor upload(const std::vector<float>& values, const TensorShape& shape) {
        return Tensor::from_vector(values, shape, Device::GPU);
    }

    Tensor upload_bytes(const std::vector<uint8_t>& values, const TensorShape& shape) {
        auto cpu = Tensor::empty(shape, Device::CPU, DataType::UInt8);
        std::copy(values.begin(), values.end(), cpu.ptr<uint8_t>());
        return cpu.to(Device::GPU);
    }

    std::vector<float> download(const Tensor& tensor) {
        Tensor view = tensor.dtype() == DataType::Float16 ? tensor.to(DataType::Float32) : tensor;
        const Tensor cpu = view.cpu().contiguous();
        const float* data = cpu.ptr<float>();
        return {data, data + cpu.numel()};
    }

    Field widen(const std::vector<float>& values) { return {values.begin(), values.end()}; }

    // |got - want| <= abs + rel * max|want| everywhere.
    void expect_field(const std::vector<float>& got, const Field& want, const double abs, const double rel,
                      const std::string& what) {
        ASSERT_EQ(got.size(), want.size()) << what;
        double scale = 0.0;
        for (const double value : want)
            scale = std::max(scale, std::abs(value));
        const double limit = abs + rel * scale;
        size_t worst = 0;
        double worst_error = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double error = std::abs(static_cast<double>(got[i]) - want[i]);
            if (!(error <= worst_error)) {
                worst_error = error;
                worst = i;
            }
        }
        EXPECT_LE(worst_error, limit) << what << ": index " << worst << " got " << got[worst] << " want "
                                      << want[worst] << " (max |want| " << scale << ")";
    }

    void expect_value(const double got, const double want, const double rel, const std::string& what) {
        EXPECT_NEAR(got, want, 1e-7 + rel * std::abs(want)) << what;
    }

    double sign(const double v) { return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0); }
    double to_half(const double v) { return static_cast<double>(static_cast<_Float16>(static_cast<float>(v))); }

    // ---- SSIM reference (ssim.cu) ---------------------------------------------

    constexpr std::array<float, 11> kGauss = {
        0.001028380123898387f, 0.0075987582094967365f, 0.036000773310661316f, 0.10936068743467331f,
        0.21300552785396576f, 0.26601171493530273f, 0.21300552785396576f, 0.10936068743467331f,
        0.036000773310661316f, 0.0075987582094967365f, 0.001028380123898387f};
    constexpr double kC1 = 0.01f * 0.01f;
    constexpr double kC2 = 0.03f * 0.03f;

    // Separable 11x11 gaussian with zero padding over one [H,W] plane.
    Field blur(const Field& f, const int H, const int W) {
        Field horizontal(f.size(), 0.0), out(f.size(), 0.0);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int d = -5; d <= 5; ++d)
                    if (x + d >= 0 && x + d < W)
                        horizontal[y * W + x] += kGauss[d + 5] * f[y * W + x + d];
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int d = -5; d <= 5; ++d)
                    if (y + d >= 0 && y + d < H)
                        out[y * W + x] += kGauss[d + 5] * horizontal[(y + d) * W + x];
        return out;
    }

    struct Shape4 {
        int n, c, h, w;
        [[nodiscard]] size_t plane() const { return static_cast<size_t>(h) * w; }
        [[nodiscard]] size_t numel() const { return static_cast<size_t>(n) * c * plane(); }
        [[nodiscard]] TensorShape tensor() const {
            return TensorShape({static_cast<size_t>(n), static_cast<size_t>(c), static_cast<size_t>(h), static_cast<size_t>(w)});
        }
    };

    Field plane_of(const Field& f, const Shape4& s, const int n, const int c) {
        const size_t base = (static_cast<size_t>(n) * s.c + c) * s.plane();
        return {f.begin() + static_cast<std::ptrdiff_t>(base), f.begin() + static_cast<std::ptrdiff_t>(base + s.plane())};
    }

    Field product(const Field& a, const Field& b) {
        Field out(a.size());
        for (size_t i = 0; i < a.size(); ++i)
            out[i] = a[i] * b[i];
        return out;
    }

    // Per element SSIM, CS and the half-rounded partials the backward reads.
    // Decoupled (raw non-empty): luminance from the corrected image, contrast
    // and structure from the raw render; partials as in decoupledFusedL1SSIMForwardCUDA.
    struct SsimForward {
        Field ssim, cs, dm_mu, dm_s1, dm_s12, raw_dm_mu;
    };

    SsimForward ssim_forward(const Field& x, const Field& raw, const Field& y, const Shape4& s, const double weight,
                             const Field& mask = {}) {
        SsimForward out;
        for (Field* f : {&out.ssim, &out.cs, &out.dm_mu, &out.dm_s1, &out.dm_s12, &out.raw_dm_mu})
            f->assign(s.numel(), 0.0);
        // Outside-image zero padding retains unit weight. Only excluded samples
        // inside the image reduce the Gaussian window's normalization mass.
        Field window_mass(s.plane(), 1.0);
        if (!mask.empty()) {
            Field excluded(mask.size());
            std::transform(mask.begin(), mask.end(), excluded.begin(), [](double m) { return 1.0 - m; });
            const Field missing = blur(excluded, s.h, s.w);
            for (size_t i = 0; i < window_mass.size(); ++i)
                window_mass[i] -= missing[i];
        }
        const auto moment = [&](const Field& values) {
            Field out = blur(mask.empty() ? values : product(values, mask), s.h, s.w);
            for (size_t i = 0; i < out.size(); ++i)
                out[i] = window_mass[i] >= 1e-30 ? out[i] / window_mass[i] : 0.0;
            return out;
        };
        for (int n = 0; n < s.n; ++n) {
            for (int c = 0; c < s.c; ++c) {
                const Field px = plane_of(x, s, n, c), py = plane_of(y, s, n, c);
                const Field pr = raw.empty() ? px : plane_of(raw, s, n, c);
                const Field mx = moment(px), my = moment(py), mr = moment(pr);
                const Field rr = moment(product(pr, pr)), yy = moment(product(py, py));
                const Field ry = moment(product(pr, py));
                const size_t base = (static_cast<size_t>(n) * s.c + c) * s.plane();
                for (size_t i = 0; i < s.plane(); ++i) {
                    // Weight before half rounding, as in the saved GPU partials.
                    const double derivative_weight = mask.empty() ? 1.0 : (window_mass[i] >= 1e-30 ? mask[i] / window_mass[i] : 0.0);
                    const auto partial = [&](double v) { return to_half(v * derivative_weight); };
                    const double s_r = rr[i] - mr[i] * mr[i], s_y = yy[i] - my[i] * my[i], s_ry = ry[i] - mr[i] * my[i];
                    const double B = s_r + s_y + kC2, D = 2 * s_ry + kC2;
                    const size_t e = base + i;
                    if (raw.empty()) {
                        const double A = mx[i] * mx[i] + my[i] * my[i] + kC1, C = 2 * mx[i] * my[i] + kC1;
                        out.ssim[e] = C * D / (A * B);
                        out.cs[e] = D / B;
                        const double mu1 = mx[i], mu2 = my[i];
                        out.dm_mu[e] = partial(mu2 * 2 * D / (A * B) - mu2 * 2 * C / (A * B) - mu1 * 2 * C * D / (A * A * B) +
                                               mu1 * 2 * C * D / (A * B * B));
                        out.dm_s1[e] = partial(-C * D / (A * B * B));
                        out.dm_s12[e] = partial(2 * C / (A * B));
                    } else {
                        const double A = mx[i] * mx[i] + my[i] * my[i] + kC1, C = 2 * mx[i] * my[i] + kC1;
                        const double lum = C / A;
                        out.cs[e] = D / B;
                        out.ssim[e] = lum * out.cs[e];
                        out.dm_mu[e] = partial(out.cs[e] * 2 * (my[i] * A - mx[i] * C) / (A * A));
                        const double ds1 = weight * (-(lum * D) / (B * B)), ds12 = weight * (2 * lum / B);
                        out.dm_s1[e] = partial(ds1);
                        out.dm_s12[e] = partial(ds12);
                        out.raw_dm_mu[e] = partial(ds1 * (-2 * mr[i]) + ds12 * (-my[i]));
                    }
                }
            }
        }
        return out;
    }

    // fusedL1SSIMBackwardCUDA / maskedFusedL1SSIMBackwardCUDA with chain(pixel).
    Field ssim_backward(const Field& x, const Field& y, const Field& dm_mu, const Field* dm_s1, const Field* dm_s12,
                        const Field& chain, const double weight, const Shape4& s, const Field& mask = {}) {
        Field grad(s.numel());
        double mask_sum = 0.0;
        for (const double m : mask)
            mask_sum += m;
        const double masked_chain = 1.0 / (static_cast<float>(mask_sum) * s.n * s.c + 1e-8);
        for (int n = 0; n < s.n; ++n) {
            for (int c = 0; c < s.c; ++c) {
                const size_t base = (static_cast<size_t>(n) * s.c + c) * s.plane();
                Field d0(s.plane()), d1(s.plane(), 0.0), d2(s.plane(), 0.0);
                for (size_t i = 0; i < s.plane(); ++i) {
                    const double derivative_chain = mask.empty() ? chain[i] : masked_chain;
                    d0[i] = -weight * dm_mu[base + i] * derivative_chain;
                    if (dm_s1) {
                        d1[i] = -weight * (*dm_s1)[base + i] * derivative_chain;
                        d2[i] = -weight * (*dm_s12)[base + i] * derivative_chain;
                    }
                }
                const Field s0 = blur(d0, s.h, s.w), s1 = blur(d1, s.h, s.w), s2 = blur(d2, s.h, s.w);
                for (size_t i = 0; i < s.plane(); ++i) {
                    const double p1 = x[base + i], p2 = y[base + i];
                    const double input_mask = mask.empty() ? 1.0 : mask[i];
                    grad[base + i] = input_mask * (s0[i] + 2 * p1 * s1[i] + p2 * s2[i]) + (1 - weight) * sign(p1 - p2) * chain[i];
                }
            }
        }
        return grad;
    }

    Field channel_mean(const Field& f, const Shape4& s) {
        Field out(static_cast<size_t>(s.n) * s.plane(), 0.0);
        for (int n = 0; n < s.n; ++n)
            for (int c = 0; c < s.c; ++c)
                for (size_t i = 0; i < s.plane(); ++i)
                    out[n * s.plane() + i] += f[(static_cast<size_t>(n) * s.c + c) * s.plane() + i] / s.c;
        return out;
    }

    // ssim_reduction.cu crops each axis on its own.
    bool in_mean_crop(const int x, const int y, const Shape4& s, const bool padding) {
        const bool crop_y = padding && s.h > 10, crop_x = padding && s.w > 10;
        return !((crop_y && (y < 5 || y >= s.h - 5)) || (crop_x && (x < 5 || x >= s.w - 5)));
    }

    // The backward crops each axis like the forward.
    Field valid_chain(const Shape4& s, const bool padding) {
        const bool crop_y = padding && s.h > 10, crop_x = padding && s.w > 10;
        const double count = static_cast<double>(s.n) * s.c * (crop_y ? s.h - 10 : s.h) * (crop_x ? s.w - 10 : s.w);
        const float per_pixel = 1.0f / static_cast<float>(count);
        Field chain(s.plane(), 0.0);
        for (int y = 0; y < s.h; ++y)
            for (int x = 0; x < s.w; ++x)
                if (in_mean_crop(x, y, s, padding))
                    chain[y * s.w + x] = per_pixel;
        return chain;
    }

    struct PhotoCase {
        ops::PhotoPath path;
        Shape4 shape;
        bool target_bytes = false;
        bool mask_bytes = false;
        float weight = 0.2f;
    };

    struct PhotoRef {
        double loss = 0.0;
        Field grad, grad_raw, ssim_map, cs_map;
    };

    PhotoRef photo_reference(const PhotoCase& pc, const Field& x, const Field& raw, const Field& y, const Field& mask) {
        const Shape4& s = pc.shape;
        const bool decoupled = pc.path == ops::PhotoPath::Decoupled || pc.path == ops::PhotoPath::MaskedDecoupled;
        const bool masked = !mask.empty();
        PhotoRef ref;
        if (pc.path == ops::PhotoPath::L1) {
            ref.grad.resize(s.numel());
            for (size_t i = 0; i < s.numel(); ++i) {
                ref.loss += std::abs(x[i] - y[i]) / static_cast<double>(s.numel());
                ref.grad[i] = sign(x[i] - y[i]) / static_cast<double>(s.numel());
            }
            return ref;
        }
        const bool pure = pc.path == ops::PhotoPath::SSIM;
        const double w = pure ? 1.0 : pc.weight;
        const SsimForward f = ssim_forward(x, decoupled ? raw : Field{}, y, s, w, mask);
        const Field ssim_mean = channel_mean(f.ssim, s);
        ref.ssim_map = pure ? f.ssim : ssim_mean;
        ref.cs_map = pure ? f.cs : channel_mean(f.cs, s);
        if (masked) {
            for (int n = 0; n < s.n; ++n)
                for (size_t i = 0; i < s.plane(); ++i)
                    if (mask[i] <= 0.0) {
                        ref.ssim_map[n * s.plane() + i] = 1.0;
                        ref.cs_map[n * s.plane() + i] = 1.0;
                    }
        }

        Field chain;
        if (masked) {
            double mask_sum = 0.0;
            for (const double m : mask)
                mask_sum += m;
            const double normalized = static_cast<float>(mask_sum) * s.n * s.c + 1e-8;
            chain.resize(s.plane());
            for (size_t i = 0; i < s.plane(); ++i)
                chain[i] = mask[i] / normalized;
            for (int n = 0; n < s.n; ++n)
                for (int c = 0; c < s.c; ++c)
                    for (size_t i = 0; i < s.plane(); ++i) {
                        const size_t e = (static_cast<size_t>(n) * s.c + c) * s.plane() + i;
                        ref.loss += chain[i] * ((1 - w) * std::abs(x[e] - y[e]) + w * (1 - ssim_mean[n * s.plane() + i]));
                    }
        } else {
            chain = valid_chain(s, true);
            double count = 0.0;
            for (int n = 0; n < s.n; ++n)
                for (int c = 0; c < s.c; ++c)
                    for (int py = 0; py < s.h; ++py)
                        for (int px = 0; px < s.w; ++px) {
                            if (!in_mean_crop(px, py, s, true))
                                continue;
                            const size_t i = static_cast<size_t>(py) * s.w + px;
                            const size_t e = (static_cast<size_t>(n) * s.c + c) * s.plane() + i;
                            count += 1.0;
                            ref.loss += pure ? f.ssim[e]
                                             : (1 - w) * std::abs(x[e] - y[e]) + w * (1 - ssim_mean[n * s.plane() + i]);
                        }
            ref.loss /= count;
            if (pure)
                ref.loss = 1.0 - ref.loss;
        }
        if (decoupled) {
            ref.grad = ssim_backward(x, y, f.dm_mu, nullptr, nullptr, chain, w, s, mask);
            ref.grad_raw = ssim_backward(raw, y, f.raw_dm_mu, &f.dm_s1, &f.dm_s12, chain, 1.0, s, mask);
        } else {
            ref.grad = ssim_backward(x, y, f.dm_mu, &f.dm_s1, &f.dm_s12, chain, w, s, mask);
        }
        return ref;
    }

    struct PhotoInputs {
        Tensor x, raw, y, mask;
        Field hx, hraw, hy, hmask;
    };

    PhotoInputs photo_inputs(const PhotoCase& pc, const uint32_t seed) {
        const Shape4& s = pc.shape;
        PhotoInputs in;
        const auto x = uniform(s.numel(), seed);
        auto noise = uniform(s.numel(), seed + 1, -0.15f, 0.15f);
        std::vector<float> y(s.numel());
        for (size_t i = 0; i < y.size(); ++i)
            y[i] = std::clamp(0.8f * x[i] + 0.1f + noise[i], 0.f, 1.f);
        in.x = upload(x, s.tensor());
        in.hx = widen(x);
        if (pc.target_bytes) {
            std::vector<uint8_t> bytes(y.size());
            for (size_t i = 0; i < y.size(); ++i) {
                bytes[i] = static_cast<uint8_t>(std::lround(y[i] * 255.f));
                y[i] = static_cast<float>(bytes[i]) * (1.0f / 255.0f);
            }
            in.y = upload_bytes(bytes, s.tensor());
        } else {
            in.y = upload(y, s.tensor());
        }
        in.hy = widen(y);
        const auto raw = uniform(s.numel(), seed + 2, 0.1f, 0.9f);
        in.raw = upload(raw, s.tensor());
        in.hraw = widen(raw);
        std::vector<float> mask(s.plane());
        std::vector<uint8_t> mask_bytes(s.plane());
        const auto soft = uniform(s.plane(), seed + 3);
        for (size_t i = 0; i < mask.size(); ++i) {
            mask_bytes[i] = soft[i] < 0.3f ? 0 : static_cast<uint8_t>(1 + (i % 200));
            mask[i] = pc.mask_bytes ? (mask_bytes[i] != 0 ? 1.f : 0.f) : (soft[i] < 0.3f ? 0.f : soft[i]);
        }
        const TensorShape plane({static_cast<size_t>(s.h), static_cast<size_t>(s.w)});
        in.mask = pc.mask_bytes ? upload_bytes(mask_bytes, plane) : upload(mask, plane);
        in.hmask = widen(mask);
        return in;
    }

    void check_photo(const lfs::training::TrainingOps& table, const PhotoCase& pc, const std::string& name,
                     const double map_tolerance = 2e-6) {
        SCOPED_TRACE(name);
        const PhotoInputs in = photo_inputs(pc, 11);
        const bool decoupled = pc.path == ops::PhotoPath::Decoupled || pc.path == ops::PhotoPath::MaskedDecoupled;
        const bool masked = pc.path == ops::PhotoPath::MaskedFused || pc.path == ops::PhotoPath::MaskedDecoupled;
        ops::PhotoSaved saved{.backend = table.photometric->create()};
        Tensor loss, grad, grad_raw;
        table.photometric->evaluate(saved, in.x, decoupled ? in.raw : Tensor{}, in.y, masked ? in.mask : Tensor{},
                                    {.path = pc.path, .ssim_weight = pc.weight, .valid_padding = true}, loss, grad,
                                    grad_raw);
        const PhotoRef ref = photo_reference(pc, in.hx, decoupled ? in.hraw : Field{}, in.hy, masked ? in.hmask : Field{});
        expect_value(download(loss)[0], ref.loss, 1e-5, name + " loss");
        expect_field(download(grad), ref.grad, 1e-9, 2e-4, name + " grad");
        if (decoupled)
            expect_field(download(grad_raw), ref.grad_raw, 1e-9, 2e-4, name + " grad_raw");
        else
            EXPECT_FALSE(grad_raw.is_valid());
        if (pc.path != ops::PhotoPath::L1) {
            expect_field(download(saved.ssim_map), ref.ssim_map, map_tolerance, 0.0, name + " ssim_map");
            expect_field(download(saved.cs_map), ref.cs_map, map_tolerance, 0.0, name + " cs_map");
        }
    }

    TEST_P(PortableLosses, PhotometricPathsMatchReference) {
        const Shape4 fixture{1, 3, 32, 40};
        const Shape4 odd{2, 3, 21, 27};
        check_photo(*table_, {ops::PhotoPath::L1, fixture}, "l1");
        check_photo(*table_, {ops::PhotoPath::L1, odd, true}, "l1_u8");
        check_photo(*table_, {ops::PhotoPath::SSIM, fixture}, "ssim");
        check_photo(*table_, {ops::PhotoPath::Fused, fixture}, "fused");
        check_photo(*table_, {ops::PhotoPath::Fused, odd, true}, "fused_batch_u8");
        check_photo(*table_, {ops::PhotoPath::Fused, {1, 3, 9, 30}}, "fused_thin");
        check_photo(*table_, {ops::PhotoPath::Decoupled, fixture}, "decoupled");
        check_photo(*table_, {ops::PhotoPath::MaskedFused, fixture}, "masked_fused");
        check_photo(*table_, {ops::PhotoPath::MaskedFused, odd, true, true}, "masked_fused_u8");
        check_photo(*table_, {ops::PhotoPath::MaskedDecoupled, fixture, false, true}, "masked_decoupled");
        // Soft normalized moments add float32 rounding before covariance
        // subtraction. Keep the original cases and all gradient/loss bounds;
        // this new map comparison allows 3 ppm against the double reference.
        check_photo(*table_, {ops::PhotoPath::MaskedDecoupled, fixture}, "masked_decoupled_soft", 3e-6);
        check_photo(*table_, {ops::PhotoPath::MaskedDecoupled, odd, true, true}, "masked_decoupled_batch_u8");
    }

#ifdef LFS_TENSOR_METAL
    // photo_ssim_backward_mu, the SSIM backward of the decoupled corrected image, convolves dm_mu alone: the sigma
    // partials' terms it leaves out add zeros. Launched as photometric_metal.cpp launches them for the unmasked and
    // the masked decoupled path, it gives photo_ssim_backward's gradient for zero sigma partials up to float
    // rounding; both also match the CPU reference.
    TEST_P(PortableLosses, SsimBackwardMuMatchesZeroSigmaPartials) {
        if (GetParam() != GpuBackend::Metal)
            GTEST_SKIP() << "Metal kernels";
        namespace mk = lfs::training::metal;
        // PhotoBackwardParams in photometric.metal.
        struct Params {
            uint64_t prediction, target, mask, mask_sum, dm_mu, dm_sigma1, dm_sigma12, grad;
            int32_t height, width, channels;
            uint32_t tiles_x, tiles_y;
            float ssim_weight, grad_per_pixel;
            uint32_t valid_padding, masked;
        };
        struct Variant {
            const char* name;
            bool masked, padding, mask_bytes;
        };
        constexpr float kWeight = 0.2f;
        constexpr uint32_t kTile = 16;
        for (const Shape4& s : {Shape4{1, 3, 37, 45}, Shape4{2, 3, 21, 27}})
            for (const bool target_bytes : {false, true})
                for (const Variant& v : {Variant{"valid padding", false, true, false}, Variant{"no padding", false, false, false},
                                         Variant{"float mask", true, false, false}, Variant{"byte mask", true, false, true}}) {
                    SCOPED_TRACE(std::format("{}x{}x{}, {} target, {}", s.n, s.h, s.w, target_bytes ? "u8" : "f32", v.name));
                    const PhotoInputs in = photo_inputs({ops::PhotoPath::MaskedDecoupled, s, target_bytes, v.mask_bytes}, 29);
                    const Tensor dm_mu = upload(uniform(s.numel(), 41, -1.f, 1.f), s.tensor()).to(DataType::Float16);
                    const Tensor zero_sigma = Tensor::zeros(s.tensor(), Device::GPU, DataType::Float16);
                    const Tensor mask = v.masked ? in.mask : Tensor{};
                    // The masked chain divides by the forward's normalization, mask sum * N * C + eps.
                    double mask_total = 0.0;
                    for (const double m : in.hmask)
                        mask_total += m;
                    const double normalized = static_cast<float>(mask_total) * s.n * s.c + 1e-8;
                    const Tensor mask_sum = v.masked ? upload({static_cast<float>(normalized)}, TensorShape({1})) : Tensor{};
                    Field chain = valid_chain(s, v.padding);
                    if (v.masked)
                        for (size_t i = 0; i < chain.size(); ++i)
                            chain[i] = in.hmask[i] / normalized;
                    const int h = v.padding && s.h > 10 ? s.h - 10 : s.h, w = v.padding && s.w > 10 ? s.w - 10 : s.w;
                    const float grad_per_pixel = v.masked ? 0.f : 1.0f / static_cast<float>(static_cast<double>(s.n) * s.c * h * w);
                    const uint32_t tiles_x = (static_cast<uint32_t>(s.w) + kTile - 1) / kTile;
                    const uint32_t tiles_y = (static_cast<uint32_t>(s.h) + kTile - 1) / kTile;
                    const auto run = [&](const std::string_view function, const Tensor& sigma) {
                        const Tensor grad = Tensor::zeros(s.tensor(), Device::GPU);
                        const Params params{.prediction = mk::address(in.x),
                                            .target = mk::address(in.y),
                                            .mask = mk::address(mask),
                                            .mask_sum = mk::address(mask_sum),
                                            .dm_mu = mk::address(dm_mu),
                                            .dm_sigma1 = mk::address(sigma),
                                            .dm_sigma12 = mk::address(sigma),
                                            .grad = mk::address(grad),
                                            .height = s.h,
                                            .width = s.w,
                                            .channels = s.c,
                                            .tiles_x = tiles_x,
                                            .tiles_y = tiles_y,
                                            .ssim_weight = kWeight,
                                            .grad_per_pixel = grad_per_pixel,
                                            .valid_padding = v.padding ? 1u : 0u,
                                            .masked = v.masked ? 1u : 0u};
                        mk::launch_2d(function, params, {&in.x, &in.y, &mask, &mask_sum, &dm_mu, &sigma, &sigma, &grad},
                                      tiles_x, tiles_y * static_cast<uint32_t>(s.n), kTile, kTile,
                                      {{40, target_bytes ? 1u : 0u}, {41, v.mask_bytes ? 1u : 0u}, {42, 0u}, {43, 0u}});
                        return download(grad);
                    };
                    const std::vector<float> mu = run("photo_ssim_backward_mu", Tensor{});
                    const std::vector<float> full = run("photo_ssim_backward", zero_sigma);
                    // The kernels are two fast-math specializations of one template, so the compiler may order the dm_mu
                    // sums differently in each (on an Apple M3 Ultra they agree bit for bit in all 16 cases). Another
                    // order moves an 11-tap sum by at most 10 float epsilons of its terms' magnitudes, which stay below
                    // 0.2 (the SSIM weight times |dm_mu| < 1) of the chain factor, while the L1 term keeps the largest
                    // gradient above 0.6 of it: both passes and the final multiply-add stay within 8 epsilons of the
                    // largest gradient magnitude.
                    expect_field(mu, widen(full), 0.0, 8.0 * std::numeric_limits<float>::epsilon(),
                                 "photo_ssim_backward_mu against photo_ssim_backward");
                    const Field want = ssim_backward(in.hx, in.hy, widen(download(dm_mu)), nullptr, nullptr, chain, kWeight, s,
                                                     v.masked ? in.hmask : Field{});
                    expect_field(mu, want, 1e-9, 2e-4, "photo_ssim_backward_mu");
                    expect_field(full, want, 1e-9, 2e-4, "photo_ssim_backward");
                }
    }
#endif

    // Central differences of the GPU loss 1 - SSIM against the GPU gradient.
    TEST_P(PortableLosses, SsimGradientMatchesFiniteDifferences) {
        const Shape4 s{1, 3, 16, 20};
        const PhotoCase pc{ops::PhotoPath::SSIM, s};
        const PhotoInputs in = photo_inputs(pc, 5);
        const auto x = download(in.x);
        ops::PhotoSaved saved{.backend = table_->photometric->create()};
        const auto evaluate = [&](const std::vector<float>& image, Tensor& grad) {
            Tensor loss, grad_raw;
            table_->photometric->evaluate(saved, upload(image, s.tensor()), {}, in.y, {},
                                          {.path = ops::PhotoPath::SSIM, .ssim_weight = 1.f, .valid_padding = true}, loss,
                                          grad, grad_raw);
            return static_cast<double>(download(loss)[0]);
        };
        Tensor grad;
        evaluate(x, grad);
        const auto analytic = download(grad);
        constexpr float h = 1e-2f;
        const std::tuple<int, int, int> probes[] = {{0, 0, 0}, {0, 3, 4}, {1, 7, 9}, {2, 15, 19}, {1, 5, 5}, {2, 10, 14}, {0, 12, 2}, {1, 8, 17}};
        for (const auto& [c, py, px] : probes) {
            const size_t i = (static_cast<size_t>(c) * s.h + py) * s.w + px;
            auto plus = x, minus = x;
            plus[i] += h;
            minus[i] -= h;
            Tensor scratch;
            const double numeric = (evaluate(plus, scratch) - evaluate(minus, scratch)) / (2.0 * h);
            EXPECT_NEAR(analytic[i], numeric, 2e-6 + 3e-2 * std::abs(numeric))
                << "channel " << c << " pixel (" << px << ", " << py << ")";
        }
    }

    TEST_P(PortableLosses, PhotometricMetricAndErrorMaps) {
        const Shape4 s{1, 3, 32, 40};
        const PhotoCase pc{ops::PhotoPath::SSIM, s};
        const PhotoInputs in = photo_inputs(pc, 23);
        const SsimForward f = ssim_forward(in.hx, {}, in.hy, s, 1.0);
        ops::PhotoSaved saved{.backend = table_->photometric->create()};
        for (const bool padding : {true, false}) {
            double sum = 0.0, count = 0.0;
            for (int c = 0; c < s.c; ++c)
                for (int y = 0; y < s.h; ++y)
                    for (int x = 0; x < s.w; ++x)
                        if (in_mean_crop(x, y, s, padding)) {
                            sum += f.ssim[(static_cast<size_t>(c) * s.h + y) * s.w + x];
                            count += 1.0;
                        }
            for (const bool maps : {false, true}) {
                const Tensor value = table_->photometric->metric(saved, in.x, in.y, maps, padding);
                expect_value(download(value)[0], sum / count, 1e-5, std::format("metric maps={} padding={}", maps, padding));
            }
        }
        expect_field(download(saved.ssim_map), f.ssim, 2e-6, 0.0, "metric ssim_map");
        expect_field(download(saved.cs_map), f.cs, 2e-6, 0.0, "metric cs_map");

        for (const bool cs_only : {false, true}) {
            Tensor error;
            table_->photometric->error_map(saved, in.x, in.y, error, cs_only);
            const Field mean = channel_mean(cs_only ? f.cs : f.ssim, s);
            Field want(mean.size());
            for (size_t i = 0; i < mean.size(); ++i)
                want[i] = std::max(1.0 - mean[i], 0.0);
            expect_field(download(error), want, 2e-6, 0.0, std::format("error_map cs_only={}", cs_only));
        }
        Tensor from_map = Tensor::empty({static_cast<size_t>(s.h), static_cast<size_t>(s.w)}, Device::GPU);
        table_->photometric->map_to_error(saved.ssim_map, from_map);
        Field want(s.plane());
        const Field mean = channel_mean(f.ssim, s);
        for (size_t i = 0; i < want.size(); ++i)
            want[i] = std::max(1.0 - mean[i], 0.0);
        expect_field(download(from_map), want, 2e-6, 0.0, "map_to_error");
    }

    // LossWorkspaceArena accounting: exact on variant switches, a same-variant
    // high-water on shape changes, shrink to the active layout, reset to zero.
    TEST_P(PortableLosses, PhotometricWorkspaceBytesFollowTheArena) {
        const auto layout = [](const std::vector<size_t>& fields) {
            size_t total = 0;
            for (const size_t bytes : fields)
                total = ((total + 255) & ~size_t{255}) + bytes;
            return (total + 255) & ~size_t{255};
        };
        const auto fused_bytes = [&](const Shape4& s) {
            return layout({s.n * s.plane() * 4, s.numel() * 2, s.numel() * 2, s.numel() * 2, s.numel() * 4, 4096, 4});
        };
        // As LossWorkspaceArena::masked_decoupled_layout_bytes: the raw-render gradient is not an arena field.
        const auto masked_decoupled_bytes = [&](const Shape4& s) {
            return layout({s.n * s.plane() * 4, s.numel() * 2, s.numel() * 2, s.numel() * 2, s.numel() * 2,
                           s.numel() * 4, 8192, 4, 4});
        };
        ops::PhotoSaved saved{.backend = table_->photometric->create()};
        const auto run = [&](const ops::PhotoPath path, const Shape4& s) {
            const PhotoInputs in = photo_inputs({path, s}, 3);
            Tensor loss, grad, grad_raw;
            table_->photometric->evaluate(saved, in.x, in.raw, in.y, in.mask,
                                          {.path = path, .ssim_weight = 0.2f, .valid_padding = true}, loss, grad, grad_raw);
            download(loss);
        };
        const Shape4 big{1, 3, 32, 40}, small{1, 3, 16, 20};
        run(ops::PhotoPath::Fused, big);
        auto bytes = table_->photometric->workspace_bytes(saved);
        EXPECT_EQ(bytes.required, fused_bytes(big));
        EXPECT_EQ(bytes.allocated, fused_bytes(big));
        run(ops::PhotoPath::Fused, small);
        bytes = table_->photometric->workspace_bytes(saved);
        EXPECT_EQ(bytes.required, fused_bytes(small));
        EXPECT_EQ(bytes.allocated, fused_bytes(big));
        table_->photometric->shrink_to_required(saved);
        EXPECT_EQ(table_->photometric->workspace_bytes(saved).allocated, fused_bytes(small));
        run(ops::PhotoPath::MaskedDecoupled, small);
        bytes = table_->photometric->workspace_bytes(saved);
        EXPECT_EQ(bytes.required, masked_decoupled_bytes(small));
        EXPECT_EQ(bytes.allocated, masked_decoupled_bytes(small));
        EXPECT_EQ(bytes.error_map, 0u);
        Tensor error;
        table_->photometric->error_map(saved, photo_inputs({ops::PhotoPath::SSIM, big}, 1).x,
                                       photo_inputs({ops::PhotoPath::SSIM, big}, 2).x, error, false);
        EXPECT_EQ(table_->photometric->workspace_bytes(saved).error_map, big.numel() * 4);
        table_->photometric->reset(saved);
        bytes = table_->photometric->workspace_bytes(saved);
        EXPECT_EQ(bytes.required, 0u);
        EXPECT_EQ(bytes.allocated, 0u);
    }

    // SSIM forward + backward at training resolution. Run with
    // --gtest_also_run_disabled_tests.
    TEST_P(PortableLosses, DISABLED_FusedSsimTiming) {
        const Shape4 s{1, 3, 700, 1000};
        const PhotoInputs in = photo_inputs({ops::PhotoPath::Fused, s}, 9);
        ops::PhotoSaved saved{.backend = table_->photometric->create()};
        Tensor loss, grad, grad_raw;
        const auto step = [&] {
            table_->photometric->evaluate(saved, in.x, {}, in.y, {},
                                          {.path = ops::PhotoPath::Fused, .ssim_weight = 0.2f, .valid_padding = true}, loss,
                                          grad, grad_raw);
        };
        for (int i = 0; i < 10; ++i)
            step();
        download(loss);
        constexpr int kSteps = 200;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < kSteps; ++i)
            step();
        download(loss);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::printf("fused L1+SSIM forward+backward 1000x700x3: %.3f ms/step\n", ms / kSteps);
    }

    // ---- Masks (mask_preprocess.cu) --------------------------------------------

    TEST_P(PortableLosses, MasksMatchReference) {
        constexpr size_t h = 33, w = 35, n = h * w;
        const auto soft = uniform(n, 7);
        std::vector<uint8_t> bytes(n);
        std::vector<float> floats(n);
        for (size_t i = 0; i < n; ++i) {
            bytes[i] = static_cast<uint8_t>((i * 79 + 31) % 256);
            floats[i] = static_cast<float>(bytes[i]) / 255.f;
        }
        const auto alpha_h = uniform(n, 8);
        const auto roi_h = uniform(n, 9, 0.2f, 1.f);
        const TensorShape shape({h, w});
        const Tensor alpha = upload(alpha_h, shape), roi = upload(roi_h, shape);
        for (const bool use_bytes : {false, true}) {
            const Tensor mask = use_bytes ? upload_bytes(bytes, shape) : upload(floats, shape);
            for (const bool use_roi : {false, true}) {
                for (const int mode : {0, 1}) {
                    const std::string tag = std::format("bytes={} roi={} mode={}", use_bytes, use_roi, mode);
                    const auto normalized = [&](size_t i) { return use_bytes ? bytes[i] / 255.0 : double(floats[i]); };
                    const auto as_float = [&](size_t i) { return use_bytes ? (bytes[i] != 0 ? 1.0 : 0.0) : double(floats[i]); };
                    const auto roi_at = [&](size_t i) { return use_roi ? double(roi_h[i]) : 1.0; };

                    Tensor weight = Tensor::full({h, w}, -9.f, Device::GPU);
                    table_->masks->photometric_weight(mask, use_roi ? roi : Tensor{}, weight, static_cast<ops::MaskPhotoMode>(mode));
                    Field want(n);
                    for (size_t i = 0; i < n; ++i)
                        want[i] = (mode == 1 ? (normalized(i) > k::kMaskKeepMin ? 1.0 : 0.0) : as_float(i)) * roi_at(i);
                    expect_field(download(weight), want, 1e-7, 0.0, "photometric weight " + tag);

                    constexpr float power = 1.7f, scale = 0.13f;
                    Tensor grad = Tensor::full({h, w}, -9.f, Device::GPU), temp = Tensor::zeros({1024}, Device::GPU);
                    Tensor loss = Tensor::full({1}, -9.f, Device::GPU);
                    table_->masks->opacity_penalty(alpha, mask, use_roi ? roi : Tensor{}, grad, temp, loss,
                                                   static_cast<ops::MaskOpacityMode>(mode), power, scale);
                    double sum = 0.0;
                    for (size_t i = 0; i < n; ++i) {
                        const double bg = mode == 1 ? (normalized(i) >= k::kMaskSegmentMin && normalized(i) <= k::kMaskKeepMin ? 1.0 : 0.0)
                                                    : 1.0 - as_float(i);
                        const double pen = (bg <= 0 ? 0.0 : (bg >= 1 ? 1.0 : std::pow(bg, double(power)))) * roi_at(i);
                        want[i] = pen * (scale / static_cast<float>(n));
                        sum += alpha_h[i] * pen;
                    }
                    expect_field(download(grad), want, 1e-9, 1e-5, "opacity grad " + tag);
                    expect_value(download(loss)[0], scale * sum / n, 1e-5, "opacity loss " + tag);

                    table_->masks->alpha_consistency(alpha, mask, use_roi ? roi : Tensor{}, grad, temp, loss, 10.f);
                    sum = 0.0;
                    for (size_t i = 0; i < n; ++i) {
                        const double d = alpha_h[i] - as_float(i);
                        want[i] = sign(d) * roi_at(i) * (10.0f / static_cast<float>(n));
                        sum += std::abs(d) * roi_at(i);
                    }
                    expect_field(download(grad), want, 1e-9, 1e-6, "alpha grad " + tag);
                    expect_value(download(loss)[0], 10.0 * sum / n, 1e-5, "alpha loss " + tag);
                }
            }
        }
        Tensor grad = Tensor::full({h, w}, -9.f, Device::GPU), temp = Tensor::zeros({1024}, Device::GPU);
        Tensor loss = Tensor::full({1}, -9.f, Device::GPU);
        table_->masks->opacity_penalty(alpha, upload(floats, shape), roi, grad, temp, loss, ops::MaskOpacityMode::BinaryGt0,
                                       1.f, 0.f);
        expect_field(download(grad), Field(n, 0.0), 0.0, 0.0, "zero-scale grad");
        EXPECT_EQ(download(loss)[0], 0.f);
    }

    // ---- Extra loss (regularization.cu) ------------------------------------------

    TEST_P(PortableLosses, ExtraLossMatchesReference) {
        constexpr size_t n = 1027;
        for (const auto kind : {ops::Regularizer::Scale, ops::Regularizer::Opacity}) {
            const size_t attrs = kind == ops::Regularizer::Scale ? 3 : 1;
            const auto raw_h = uniform(n * attrs, 4, -2.f, 2.f);
            const auto grad_h = uniform(n * attrs, 5, -0.5f, 0.5f);
            const Tensor raw = upload(raw_h, {n, attrs});
            for (const bool with_grad : {true, false}) {
                Tensor gradient = with_grad ? upload(grad_h, {n, attrs}) : Tensor{};
                Tensor loss = Tensor::full({1}, -9.f, Device::GPU), temp = Tensor::zeros({1024}, Device::GPU);
                constexpr float weight = 0.013f;
                table_->extra_loss->regularize(raw, gradient, loss, temp, kind, weight);
                const size_t count = n * attrs;
                Field want(count);
                double sum = 0.0;
                for (size_t i = 0; i < count; ++i) {
                    const double value = kind == ops::Regularizer::Scale ? std::exp(double(raw_h[i])) : 1.0 / (1.0 + std::exp(-double(raw_h[i])));
                    const double derivative = kind == ops::Regularizer::Scale ? value : value * (1.0 - value);
                    sum += value;
                    want[i] = grad_h[i] + weight / static_cast<float>(count) * derivative;
                }
                const std::string tag = std::format("regularizer={} gradient={}", static_cast<int>(kind), with_grad);
                expect_value(download(loss)[0], weight * sum / count, 1e-5, "loss " + tag);
                if (with_grad)
                    expect_field(download(gradient), want, 1e-7, 1e-6, "grad " + tag);
            }
        }
        Tensor loss = Tensor::full({1}, -9.f, Device::GPU), temp = Tensor::zeros({1024}, Device::GPU), none;
        table_->extra_loss->regularize(upload(uniform(n, 1), {n, 1}), none, loss, temp, ops::Regularizer::Scale, 0.f);
        EXPECT_EQ(download(loss)[0], -9.f) << "zero weight leaves the loss untouched";
    }

    // ---- Geometry (depth_loss.cu, normal_loss.cu, normal_consistency_loss.cu) --

    bool finite(const double v) { return std::isfinite(v); }

    struct GeomInputs {
        int w = 0, h = 0;
        std::vector<float> depth, alpha, target, weight, normal, prior;
    };

    GeomInputs geom_inputs(const int w, const int h, const bool sparse_weight) {
        GeomInputs in{w, h};
        const size_t n = static_cast<size_t>(w) * h;
        in.alpha = uniform(n, 31, 0.55f, 1.f);
        in.depth.resize(n);
        in.target = uniform(n, 32, 0.2f, 0.7f);
        in.weight = uniform(n, 33, 0.f, 1.5f);
        in.normal.resize(3 * n);
        in.prior.resize(3 * n);
        const auto jitter = uniform(3 * n, 34, -0.3f, 0.3f);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const size_t i = static_cast<size_t>(y) * w + x;
                const float e = 2.f + 0.3f * std::sin(x / 7.f) + 0.2f * std::cos(y / 5.f) + 0.01f * jitter[i];
                in.depth[i] = e * in.alpha[i];
                if (sparse_weight && (x + y) % 7 == 0)
                    in.weight[i] = 0.f;
                in.normal[i] = 0.2f + jitter[i];
                in.normal[n + i] = -0.1f + jitter[n + i];
                in.normal[2 * n + i] = -0.9f + 0.5f * jitter[2 * n + i];
                in.prior[i] = 0.1f - jitter[2 * n + i];
                in.prior[n + i] = 0.3f + jitter[i];
                in.prior[2 * n + i] = -0.8f + jitter[n + i];
            }
        in.alpha[5] = 0.f; // inactive pixels
        in.target[9] = -1.f;
        return in;
    }

    struct DepthRef {
        std::array<double, 10> finals{};
        double loss = 0.0;
        Field grad_depth, grad_alpha;
    };

    DepthRef depth_reference(const GeomInputs& in, const bool weighted, const ops::DepthParams& p) {
        namespace slots = k::depth_loss_slots;
        const size_t n = static_cast<size_t>(in.w) * in.h;
        const auto active = [&](const size_t i) {
            const double t = in.target[i], d = in.depth[i], a = in.alpha[i];
            return t > 0 && a > k::kDepthLossMinAlpha && finite(t) && finite(d) && finite(a);
        };
        const auto pixel_weight = [&](const size_t i) {
            return !weighted ? 1.0 : (finite(in.weight[i]) && in.weight[i] > 0 ? double(in.weight[i]) : 0.0);
        };
        const auto expected = [&](const size_t i) { return std::max(double(in.depth[i]), 0.0) / in.alpha[i]; };
        DepthRef ref;
        double sum_alpha = 0, sum_e = 0, count = 0;
        for (size_t i = 0; i < n; ++i)
            if (active(i) && pixel_weight(i) != 0) {
                const double aw = in.alpha[i] * pixel_weight(i);
                sum_alpha += aw;
                sum_e += aw * expected(i);
                count += 1;
            }
        bool valid = sum_alpha > 0;
        const double mean_e = valid ? sum_e / sum_alpha : 0;
        ref.finals[slots::kFloor] = std::max(1e-8, k::kDepthLossFloorFraction * mean_e);
        const bool anchor = p.anchor && p.anchor->valid;
        const double floor_f = anchor && p.anchor->floor > 0 ? p.anchor->floor : ref.finals[slots::kFloor];
        double p1 = 0, p2 = 0;
        if (valid)
            for (size_t i = 0; i < n; ++i)
                if (active(i) && pixel_weight(i) != 0) {
                    const double inv = 1.0 / (expected(i) + floor_f), aw = in.alpha[i] * pixel_weight(i);
                    p1 += aw * inv;
                    p2 += aw * inv * inv;
                }
        const double a_floor = anchor ? p.anchor->floor : 0, a_scale = anchor ? p.anchor->scale : 0;
        const double a_shift = anchor ? p.anchor->shift : 0;
        valid = valid && a_floor > 0;
        double sigma = 0;
        if (valid) {
            const double mean = p1 / sum_alpha;
            sigma = std::sqrt(std::max(std::max(p2 / sum_alpha - mean * mean, 0.0), 1e-20));
            ref.finals[slots::kFloor] = a_floor;
        }
        ref.finals[slots::kValid] = valid ? 1 : 0;
        ref.finals[slots::kModel] = anchor ? p.anchor->model : 0;
        ref.finals[slots::kScale] = a_scale;
        ref.finals[slots::kShift] = a_shift;
        ref.finals[slots::kSigmaP] = sigma;
        ref.finals[slots::kInvNorm] = valid ? 1.0 / sum_alpha : 0;
        ref.finals[slots::kSumAlpha] = sum_alpha;
        ref.finals[slots::kCount] = count;
        ref.finals[slots::kMeanExpectedDepth] = mean_e;

        const double fl = ref.finals[slots::kFloor], inv_norm = ref.finals[slots::kInvNorm];
        const double inv_sigma = valid && sigma > 0 ? 1.0 / (k::kDepthLossResidualScale * sigma) : 0;
        const double p_max = 1.0 / fl, half_step = 0.5 * std::max(p.prior_quantization_step, 0.f) * std::abs(a_scale);
        const int model = anchor ? p.anchor->model : 0;
        struct S {
            bool ok = false;
            double alpha = 0, wa = 0, e = 0, p = 0, d = 0, delta = 0;
        };
        const auto sample = [&](const size_t i) {
            S s;
            if (!active(i) || pixel_weight(i) == 0)
                return s;
            s.alpha = in.alpha[i];
            s.wa = in.alpha[i] * pixel_weight(i);
            s.e = expected(i);
            s.p = 1.0 / (s.e + fl);
            const double fit = a_scale * in.target[i] + a_shift;
            if (!(fit > 0))
                return s;
            if (model == 0) {
                s.d = std::min(fit, p_max);
                s.delta = half_step;
            } else {
                s.d = 1.0 / (fit + fl);
                s.delta = half_step * s.d * s.d;
            }
            s.ok = true;
            return s;
        };
        const auto rho = [](const double x) { return 0.5 * x * x / (1 + x * x); };
        const auto psi = [](const double x) { return x / ((1 + x * x) * (1 + x * x)); };
        const auto deadband = [](const double r, const double delta) {
            const double excess = std::abs(r) - delta;
            return excess > 0 ? sign(r) * excess : 0.0;
        };
        ref.grad_depth.assign(n, 0.0);
        ref.grad_alpha.assign(n, 0.0);
        double l0 = 0, l1 = 0;
        const double lambda = p.gradient_weight;
        for (size_t i = 0; valid && i < n; ++i) {
            const int x = static_cast<int>(i % in.w), y = static_cast<int>(i / in.w);
            const S c = sample(i);
            if (!c.ok)
                continue;
            double gp = 0;
            const double xr = deadband(c.p - c.d, c.delta) * inv_sigma;
            if (xr != 0) {
                l0 += c.wa * rho(xr);
                gp += c.wa * psi(xr) * inv_sigma;
            }
            const auto forward = [&](const size_t j) {
                const S nb = sample(j);
                if (!nb.ok)
                    return;
                const double xh = deadband((nb.p - c.p) - (nb.d - c.d), c.delta + nb.delta) * inv_sigma;
                if (xh != 0) {
                    const double w2 = std::min(c.wa, nb.wa);
                    l1 += w2 * rho(xh);
                    gp -= lambda * w2 * psi(xh) * inv_sigma;
                }
            };
            const auto backward = [&](const size_t j) {
                const S nb = sample(j);
                if (!nb.ok)
                    return;
                const double xh = deadband((c.p - nb.p) - (c.d - nb.d), c.delta + nb.delta) * inv_sigma;
                if (xh != 0)
                    gp += lambda * std::min(nb.wa, c.wa) * psi(xh) * inv_sigma;
            };
            if (x + 1 < in.w)
                forward(i + 1);
            if (y + 1 < in.h)
                forward(i + in.w);
            if (x > 0)
                backward(i - 1);
            if (y > 0)
                backward(i - in.w);
            if (gp != 0) {
                const double g = p.weight * inv_norm * gp * (-c.p * c.p);
                ref.grad_depth[i] = g / c.alpha;
                ref.grad_alpha[i] = -g * c.e / c.alpha;
            }
        }
        ref.loss = valid ? p.weight * inv_norm * (l0 + lambda * l1) : 0;
        return ref;
    }

    void expect_finals(const std::vector<float>& partials, const double* want, const size_t count, const std::string& what) {
        for (size_t i = 0; i < count; ++i)
            EXPECT_NEAR(partials[i], want[i], 1e-6 + 2e-5 * std::abs(want[i])) << what << " slot " << i;
    }

    TEST_P(PortableLosses, DepthLossMatchesReference) {
        const GeomInputs in = geom_inputs(41, 37, true);
        const size_t n = static_cast<size_t>(in.w) * in.h;
        const TensorShape plane({static_cast<size_t>(in.h), static_cast<size_t>(in.w)});
        k::DepthAnchor anchor;
        anchor.valid = true;
        anchor.scale = 1.1f;
        anchor.shift = 0.05f;
        anchor.floor = 0.01f;
        const std::tuple<int, float, bool> cases[] = {{0, 0.f, true}, {0, 1.f / 255.f, false}, {1, 1.f / 255.f, true}};
        for (const auto& [model, step, weighted] : cases) {
            anchor.model = model;
            const ops::DepthParams params{0.5f, 0.25f, step, &anchor};
            Tensor gd = Tensor::full(plane, 3.f, Device::GPU), ga = Tensor::full(plane, 3.f, Device::GPU);
            Tensor loss = Tensor::zeros({1}, Device::GPU);
            Tensor partials = Tensor::zeros({k::depth_loss_partial_count(n)}, Device::GPU);
            table_->geometry->depth(upload(in.depth, plane), upload(in.alpha, plane), upload(in.target, plane),
                                    weighted ? upload(in.weight, plane) : Tensor{}, gd, ga, loss, partials, params);
            const DepthRef ref = depth_reference(in, weighted, params);
            const std::string tag = std::format("model={} step={} weighted={}", model, step, weighted);
            expect_finals(download(partials), ref.finals.data(), ref.finals.size(), "depth finals " + tag);
            expect_value(download(loss)[0], ref.loss, 2e-5, "depth loss " + tag);
            expect_field(download(gd), ref.grad_depth, 1e-12, 1e-4, "depth grad_depth " + tag);
            expect_field(download(ga), ref.grad_alpha, 1e-12, 1e-4, "depth grad_alpha " + tag);
        }
        // An invalid anchor zeroes the gradients and the loss.
        anchor.valid = false;
        Tensor gd = Tensor::full(plane, 3.f, Device::GPU), ga = Tensor::full(plane, 3.f, Device::GPU);
        Tensor loss = Tensor::full({1}, 3.f, Device::GPU);
        Tensor partials = Tensor::zeros({k::depth_loss_partial_count(n)}, Device::GPU);
        table_->geometry->depth(upload(in.depth, plane), upload(in.alpha, plane), upload(in.target, plane), {}, gd, ga,
                                loss, partials, {0.5f, 0.25f, 0.f, &anchor});
        expect_field(download(gd), Field(n, 0.0), 0.0, 0.0, "invalid anchor grad_depth");
        expect_field(download(ga), Field(n, 0.0), 0.0, 0.0, "invalid anchor grad_alpha");
        EXPECT_EQ(download(loss)[0], 0.f);
    }

    struct Vec3 {
        double x, y, z;
    };
    Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    Vec3 operator*(double s, Vec3 a) { return {s * a.x, s * a.y, s * a.z}; }
    double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
    double norm(Vec3 a) { return std::sqrt(dot(a, a)); }

    Vec3 vec_at(const std::vector<float>& f, const size_t n, const size_t i) { return {f[i], f[n + i], f[2 * n + i]}; }

    struct NormalRef {
        std::array<double, 5> finals{};
        double loss = 0.0;
        Field grad_normal, grad_depth, grad_alpha;
    };

    // Shared finalize of normal_loss.cu and normal_consistency_loss.cu.
    void normal_finals(NormalRef& ref, const double sum_alpha, const double count, const double sum_cos,
                       const double weight, const double min_count, const double min_weight) {
        const bool valid = count >= min_count && sum_alpha >= min_weight;
        ref.finals = {valid ? 1.0 : 0.0, sum_alpha, count, sum_alpha > 0 ? sum_cos / sum_alpha : 0.0,
                      valid ? weight / std::max(sum_alpha, 1.0) : 0.0};
    }

    NormalRef normal_reference(const GeomInputs& in, const std::vector<float>& target, const bool weighted,
                               const double weight) {
        const size_t n = static_cast<size_t>(in.w) * in.h;
        struct S {
            bool active = false;
            double alpha = 0, cos = 0, norm = 0;
            Vec3 t{}, r{};
        };
        const auto sample = [&](const size_t i) {
            S s;
            const double a = in.alpha[i];
            const Vec3 t = vec_at(target, n, i), r = vec_at(in.normal, n, i);
            if (a <= k::kNormalLossMinAlpha)
                return s;
            double eff = a;
            if (weighted) {
                if (!(in.weight[i] > 0))
                    return s;
                eff *= in.weight[i];
            }
            if (norm(t) < k::kNormalLossMinPriorNorm || norm(r) < k::kNormalLossMinRenderNorm)
                return s;
            s.t = (1 / norm(t)) * t;
            s.r = (1 / norm(r)) * r;
            s.cos = dot(s.t, s.r);
            s.alpha = eff;
            s.norm = norm(r);
            s.active = true;
            return s;
        };
        NormalRef ref;
        double sa = 0, cnt = 0, sc = 0;
        for (size_t i = 0; i < n; ++i)
            if (const S s = sample(i); s.active) {
                sa += s.alpha;
                cnt += 1;
                sc += s.alpha * s.cos;
            }
        normal_finals(ref, sa, cnt, sc, weight, k::kNormalLossMinValidCount, k::kNormalLossMinValidWeight);
        ref.grad_normal.assign(3 * n, 0.0);
        const double inv = ref.finals[4];
        double sum = 0;
        for (size_t i = 0; i < n; ++i) {
            const S s = sample(i);
            if (!s.active || inv == 0)
                continue;
            sum += s.alpha * (1 - s.cos);
            const Vec3 g = (-inv * s.alpha / s.norm) * (s.t - s.cos * s.r);
            ref.grad_normal[i] = g.x;
            ref.grad_normal[n + i] = g.y;
            ref.grad_normal[2 * n + i] = g.z;
        }
        ref.loss = inv * sum;
        return ref;
    }

    // Consistency (prior=false) and prior depth (prior=true), added onto the
    // given starting gradients.
    // rays: optional [H,W,3] camera rays in place of the pinhole; wrap: the image wraps around horizontally (a
    // panorama), so the edge columns are each other's neighbours.
    NormalRef depth_normal_reference(const GeomInputs& in, const bool prior, const bool weighted, const ops::Intrinsics& kk,
                                     const double weight, const Field& g_normal0, const Field& g_depth0, const Field& g_alpha0,
                                     const std::vector<float>& rays = {}, const bool wrap = false) {
        const int W = in.w, H = in.h;
        const size_t n = static_cast<size_t>(W) * H;
        const auto ray = [&](const int x, const int y) {
            if (!rays.empty()) {
                const size_t j = 3 * (static_cast<size_t>(y) * W + (wrap ? (x + W) % W : x));
                return Vec3{rays[j], rays[j + 1], rays[j + 2]};
            }
            return Vec3{(x + 0.5 - kk.cx) / kk.fx, (y + 0.5 - kk.cy) / kk.fy, 1.0};
        };
        const auto neighbours = [&](const int x, const int y) {
            const size_t row = static_cast<size_t>(y) * W;
            return std::array<size_t, 4>{row + (x + 1) % W, row + (x + W - 1) % W, row + W + x, row - W + x};
        };
        const auto expected = [&](const size_t i, double& e, double& a) {
            a = in.alpha[i];
            if (a < k::kNormalConsistencyMinAlpha)
                return false;
            e = std::max(double(in.depth[i]), 0.0) / a;
            return e >= 1e-6;
        };
        struct S {
            bool active = false;
            double alpha = 0, sign = 0, raw_norm = 0, cos = 0, aw = 0, render_norm = 0;
            Vec3 nd{}, tx{}, ty{}, hat{};
            std::array<double, 4> e{}, a{};
        };
        const auto sample = [&](const int x, const int y) {
            S s;
            if ((!wrap && (x <= 0 || x >= W - 1)) || y <= 0 || y >= H - 1)
                return s;
            const size_t i = static_cast<size_t>(y) * W + x;
            double ec, ac;
            if (!expected(i, ec, ac))
                return s;
            const auto nb = neighbours(x, y);
            for (int q = 0; q < 4; ++q)
                if (!expected(nb[q], s.e[q], s.a[q]) || std::abs(s.e[q] - ec) > k::kNormalConsistencyMaxRelDepthJump * ec)
                    return s;
            s.tx = s.e[0] * ray(x + 1, y) - s.e[1] * ray(x - 1, y);
            s.ty = s.e[2] * ray(x, y + 1) - s.e[3] * ray(x, y - 1);
            const Vec3 raw = cross(s.tx, s.ty);
            if (dot(raw, raw) < 1e-24)
                return s;
            s.sign = dot(raw, ray(x, y)) > 0 ? -1.0 : 1.0;
            s.raw_norm = norm(raw);
            s.nd = (s.sign / s.raw_norm) * raw;
            s.alpha = ac;
            const Vec3 other = vec_at(prior ? in.prior : in.normal, n, i);
            const double other_norm = norm(other);
            if (other_norm < (prior ? k::kNormalLossMinPriorNorm : k::kNormalLossMinRenderNorm))
                return s;
            s.hat = (1 / other_norm) * other;
            s.render_norm = other_norm;
            s.cos = dot(s.nd, s.hat);
            s.aw = s.alpha * (!weighted ? 1.0 : (in.weight[i] > 0 ? double(in.weight[i]) : 0.0));
            s.active = s.aw != 0;
            return s;
        };
        NormalRef ref;
        double sa = 0, cnt = 0, sc = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (const S s = sample(x, y); s.active) {
                    sa += s.aw;
                    cnt += 1;
                    sc += s.aw * s.cos;
                }
        normal_finals(ref, sa, cnt, sc, weight, k::kNormalConsistencyMinValidCount, k::kNormalConsistencyMinValidWeight);
        ref.grad_normal = g_normal0;
        ref.grad_depth = g_depth0;
        ref.grad_alpha = g_alpha0;
        const double inv = ref.finals[4];
        double sum = 0;
        for (int y = 0; y < H && inv != 0; ++y)
            for (int x = 0; x < W; ++x) {
                const S s = sample(x, y);
                if (!s.active)
                    continue;
                const size_t i = static_cast<size_t>(y) * W + x;
                sum += s.aw * (1 - s.cos);
                const double gw = inv * s.aw;
                if (!prior) {
                    const Vec3 g = (-gw / s.render_norm) * (s.nd - s.cos * s.hat);
                    ref.grad_normal[i] += g.x;
                    ref.grad_normal[n + i] += g.y;
                    ref.grad_normal[2 * n + i] += g.z;
                }
                const Vec3 g_raw = (-gw * s.sign / s.raw_norm) * (s.hat - s.cos * s.nd);
                const Vec3 g_tx = cross(s.ty, g_raw), g_ty = cross(g_raw, s.tx);
                const double g_e[4] = {dot(g_tx, ray(x + 1, y)), -dot(g_tx, ray(x - 1, y)), dot(g_ty, ray(x, y + 1)),
                                       -dot(g_ty, ray(x, y - 1))};
                const auto nb = neighbours(x, y);
                for (int q = 0; q < 4; ++q) {
                    ref.grad_depth[nb[q]] += g_e[q] / s.a[q];
                    ref.grad_alpha[nb[q]] += -g_e[q] * s.e[q] / s.a[q];
                }
            }
        ref.loss = inv * sum;
        return ref;
    }

    TEST_P(PortableLosses, NormalLossesMatchReference) {
        const GeomInputs in = geom_inputs(38, 29, true);
        const size_t n = static_cast<size_t>(in.w) * in.h;
        const TensorShape plane({static_cast<size_t>(in.h), static_cast<size_t>(in.w)});
        const TensorShape planes({3, static_cast<size_t>(in.h), static_cast<size_t>(in.w)});
        const ops::Intrinsics kk{45.f, 43.f, 19.5f, 14.f};
        for (const bool weighted : {true, false}) {
            const std::string tag = std::format("weighted={}", weighted);
            const Tensor weight = weighted ? upload(in.weight, plane) : Tensor{};
            {
                Tensor gn = Tensor::full(planes, 3.f, Device::GPU), loss = Tensor::zeros({1}, Device::GPU);
                Tensor partials = Tensor::zeros({k::normal_loss_partial_count(n)}, Device::GPU);
                table_->geometry->normal(upload(in.normal, planes), upload(in.alpha, plane), upload(in.prior, planes), weight,
                                         gn, loss, partials, 0.5f);
                const NormalRef ref = normal_reference(in, in.prior, weighted, 0.5);
                expect_finals(download(partials), ref.finals.data(), ref.finals.size(), "normal finals " + tag);
                expect_value(download(loss)[0], ref.loss, 2e-5, "normal loss " + tag);
                expect_field(download(gn), ref.grad_normal, 1e-12, 1e-5, "normal grad " + tag);
            }
            const auto g_normal0 = uniform(3 * n, 40, -0.01f, 0.01f), g_depth0 = uniform(n, 41, -0.01f, 0.01f);
            const auto g_alpha0 = uniform(n, 42, -0.01f, 0.01f);
            for (const bool prior : {false, true}) {
                const std::string ptag = std::format("{} {}", prior ? "prior" : "consistency", tag);
                Tensor gn = upload(g_normal0, planes), gd = upload(g_depth0, plane), ga = upload(g_alpha0, plane);
                Tensor loss = Tensor::zeros({1}, Device::GPU);
                Tensor partials = Tensor::zeros({k::normal_consistency_partial_count(n)}, Device::GPU);
                if (prior)
                    table_->geometry->prior_depth(upload(in.prior, planes), upload(in.depth, plane), upload(in.alpha, plane),
                                                  weight, gd, ga, loss, partials, kk, 0.5f);
                else
                    table_->geometry->consistency(upload(in.normal, planes), upload(in.depth, plane), upload(in.alpha, plane),
                                                  weight, gn, gd, ga, loss, partials, kk, 0.5f);
                const NormalRef ref = depth_normal_reference(in, prior, weighted, kk, 0.5, widen(g_normal0), widen(g_depth0),
                                                             widen(g_alpha0));
                expect_finals(download(partials), ref.finals.data(), ref.finals.size(), ptag + " finals");
                EXPECT_GT(ref.finals[0], 0.5) << ptag << " fixture must be valid";
                expect_value(download(loss)[0], ref.loss, 2e-5, ptag + " loss");
                expect_field(download(gn), ref.grad_normal, 1e-12, 1e-4, ptag + " grad_normal");
                expect_field(download(gd), ref.grad_depth, 1e-12, 1e-4, ptag + " grad_depth");
                expect_field(download(ga), ref.grad_alpha, 1e-12, 1e-4, ptag + " grad_alpha");
            }
        }
    }

    // GeometryLossOps::prior_depth_and_consistency against the references of the two losses one after the other,
    // which the separate calls compute: each loss's finals and value, and the gradients of both. Pinhole cameras with
    // and without pixel weights, and a wrapping panorama with per-pixel rays.
    TEST_P(PortableLosses, PriorDepthAndConsistencyMatchReference) {
        if (table_->geometry->prior_depth_and_consistency == nullptr)
            GTEST_SKIP() << "No prior_depth_and_consistency on " << lfs::core::gpu_backend_name(GetParam());
        const ops::Intrinsics kk{45.f, 43.f, 19.5f, 14.f};
        for (const int c : {0, 1, 2}) {
            const bool weighted = c != 1, panorama = c == 2;
            const std::string tag = std::format("weighted={} panorama={}", weighted, panorama);
            GeomInputs in = geom_inputs(38, 29, true);
            const int W = in.w, H = in.h;
            const size_t n = static_cast<size_t>(W) * H;
            std::vector<float> rays;
            if (panorama) {
                // Depth periodic across the seam, so the edge columns keep their depth normals.
                rays.resize(3 * n);
                const auto jitter = uniform(n, 43, -1.f, 1.f);
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        const size_t i = static_cast<size_t>(y) * W + x;
                        const float e = 2.f + 0.3f * std::sin(6.2831853f * static_cast<float>(x) / W) +
                                        0.2f * std::cos(y / 5.f) + 0.01f * jitter[i];
                        in.depth[i] = e * in.alpha[i];
                        const float longitude = 6.2831853f * (x + 0.5f) / W - 3.1415927f;
                        const float latitude = 3.1415927f * (y + 0.5f) / H - 1.5707964f;
                        rays[3 * i] = std::cos(latitude) * std::sin(longitude);
                        rays[3 * i + 1] = std::sin(latitude);
                        rays[3 * i + 2] = std::cos(latitude) * std::cos(longitude);
                    }
            }
            const TensorShape plane({static_cast<size_t>(H), static_cast<size_t>(W)});
            const TensorShape planes({3, static_cast<size_t>(H), static_cast<size_t>(W)});
            const ops::GeometryCamera camera =
                panorama ? ops::GeometryCamera{kk.fx, kk.fy, kk.cx, kk.cy,
                                               upload(rays, TensorShape({static_cast<size_t>(H), static_cast<size_t>(W), 3})),
                                               true}
                         : ops::GeometryCamera{kk};
            const auto g_normal0 = uniform(3 * n, 40, -0.01f, 0.01f), g_depth0 = uniform(n, 41, -0.01f, 0.01f);
            const auto g_alpha0 = uniform(n, 42, -0.01f, 0.01f);
            Tensor gn = upload(g_normal0, planes), gd = upload(g_depth0, plane), ga = upload(g_alpha0, plane);
            Tensor prior_loss = Tensor::zeros({1}, Device::GPU), consistency_loss = Tensor::zeros({1}, Device::GPU);
            Tensor prior_partials = Tensor::zeros({k::normal_consistency_partial_count(n)}, Device::GPU);
            Tensor consistency_partials = Tensor::zeros({k::normal_consistency_partial_count(n)}, Device::GPU);
            table_->geometry->prior_depth_and_consistency(upload(in.prior, planes), upload(in.normal, planes),
                                                          upload(in.depth, plane), upload(in.alpha, plane),
                                                          weighted ? upload(in.weight, plane) : Tensor{}, gn, gd, ga,
                                                          prior_loss, prior_partials, consistency_loss,
                                                          consistency_partials, camera, 0.5f, 0.25f);
            const NormalRef prior = depth_normal_reference(in, true, weighted, kk, 0.5, widen(g_normal0), widen(g_depth0),
                                                           widen(g_alpha0), rays, panorama);
            const NormalRef both = depth_normal_reference(in, false, weighted, kk, 0.25, prior.grad_normal,
                                                          prior.grad_depth, prior.grad_alpha, rays, panorama);
            EXPECT_GT(prior.finals[0], 0.5) << tag << " fixture must be valid";
            EXPECT_GT(both.finals[0], 0.5) << tag << " fixture must be valid";
            expect_finals(download(prior_partials), prior.finals.data(), prior.finals.size(), "prior finals " + tag);
            expect_finals(download(consistency_partials), both.finals.data(), both.finals.size(),
                          "consistency finals " + tag);
            expect_value(download(prior_loss)[0], prior.loss, 2e-5, "prior loss " + tag);
            expect_value(download(consistency_loss)[0], both.loss, 2e-5, "consistency loss " + tag);
            expect_field(download(gn), both.grad_normal, 1e-12, 1e-4, "grad_normal " + tag);
            expect_field(download(gd), both.grad_depth, 1e-12, 1e-4, "grad_depth " + tag);
            expect_field(download(ga), both.grad_alpha, 1e-12, 1e-4, "grad_alpha " + tag);
        }
    }

    TEST_P(PortableLosses, AnchorSamplesMatchProjection) {
        constexpr int W = 40, H = 30;
        const auto prior_h = uniform(static_cast<size_t>(W) * H, 50, 0.1f, 2.f);
        std::vector<float> xyz;
        for (int i = 0; i < 900; ++i) {
            // Quarter-pixel offsets keep every projection off a pixel boundary.
            const float u = static_cast<float>(i % 45) - 2.25f, v = static_cast<float>(i / 45) * 1.5f - 1.75f;
            const float z = 1.5f + 0.01f * static_cast<float>(i % 13);
            xyz.insert(xyz.end(), {(u - 20.f) * z / 40.f, (v - 15.f) * z / 40.f, z - 0.5f});
        }
        xyz[0] = std::numeric_limits<float>::quiet_NaN();
        const std::vector<float> view{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0.5f, 0, 0, 0, 1};
        const ops::AnchorParams params{{40, 40, 20, 15}, 0.01f, {-10, -10, -10}, {10, 10, 1.9f}};
        auto got = table_->geometry->collect_anchor_samples(upload(xyz, {xyz.size() / 3, 3}), upload(view, {4, 4}),
                                                            upload(prior_h, {H, W}), params);
        std::vector<std::pair<float, float>> want;
        for (size_t i = 0; i < xyz.size() / 3; ++i) {
            const double x = xyz[3 * i], y = xyz[3 * i + 1], zw = xyz[3 * i + 2];
            if (!finite(x) || zw > params.aabb_hi[2])
                continue;
            const double z = zw + 0.5;
            const int u = static_cast<int>(std::floor(40 * x / z + 20)), v = static_cast<int>(std::floor(40 * y / z + 15));
            if (u < 0 || u >= W || v < 0 || v >= H)
                continue;
            want.emplace_back(prior_h[static_cast<size_t>(v) * W + u], static_cast<float>(z));
        }
        ASSERT_GE(want.size(), static_cast<size_t>(k::kMinAnchorSamples));
        ASSERT_EQ(got.size(), want.size());
        std::sort(got.begin(), got.end(), [](auto a, auto b) { return std::tie(a.x, a.y) < std::tie(b.x, b.y); });
        std::sort(want.begin(), want.end());
        for (size_t i = 0; i < want.size(); ++i) {
            EXPECT_EQ(got[i].x, want[i].first) << i;
            EXPECT_NEAR(got[i].y, want[i].second, 1e-6) << i;
        }
    }

    // The thin-structure denominator is read back once per camera and kept while what its weight was made from
    // stays the same; new inputs, another size or a cleared cache read the weight again.
    TEST_P(PortableLosses, StructureDenominatorCacheReadsEachCameraOnce) {
        const Tensor half = Tensor::full({8, 8}, 0.5f, Device::GPU);
        const Tensor ones = Tensor::ones({8, 8}, Device::GPU);
        const float half_sum = k::structure_base_denominator(half, 8, 8, false);
        const float ones_sum = k::structure_base_denominator(ones, 8, 8, false);
        ASSERT_NE(half_sum, ones_sum);
        k::StructureDenominatorCache<int> cache;
        EXPECT_EQ(cache.get(3, 7, half, 8, 8, false), half_sum);
        EXPECT_EQ(cache.get(3, 7, ones, 8, 8, false), half_sum) << "same camera and inputs: not read again";
        EXPECT_EQ(cache.get(4, 7, ones, 8, 8, false), ones_sum) << "another camera";
        EXPECT_EQ(cache.get(3, 8, ones, 8, 8, false), ones_sum) << "new inputs";
        EXPECT_EQ(cache.get(3, 8, half, 8, 8, true), half_sum) << "another padding";
        EXPECT_EQ(cache.get(3, 8, ones, 8, 8, true), half_sum);
        cache.clear();
        EXPECT_EQ(cache.get(3, 8, ones, 8, 8, true), ones_sum) << "cleared";
    }

    // ---- Thin-structure ridge map (structure.metal) ----------------------------------------------------------

    struct RidgeCase {
        size_t h, w;
        DataType image;
        std::optional<DataType> mask; // none, or the mask's dtype
        ops::MaskPhotoMode mode = ops::MaskPhotoMode::BinaryGt0;
    };

    std::string ridge_case_name(const RidgeCase& c) {
        const char* const mask = !c.mask                      ? "nomask"
                                 : *c.mask == DataType::UInt8 ? "u8mask"
                                                              : "f32mask";
        return std::format("{}x{}_{}_{}{}", c.h, c.w, c.image == DataType::UInt8 ? "u8" : "f32", mask,
                           c.mask && c.mode == ops::MaskPhotoMode::SegmentAndIgnore ? "_segment" : "");
    }

    // Stripes with thin bright columns and dark rows, so every scale responds. Masks hide a block and two single
    // pixels; the segment mode also gets a band just below its keep threshold (250, hidden) and one just above (251).
    struct RidgeHost {
        std::vector<float> image, mask;
        std::vector<uint8_t> image_bytes, mask_bytes;
    };

    RidgeHost ridge_host(const RidgeCase& c) {
        const size_t n = c.h * c.w;
        RidgeHost host;
        host.image.resize(3 * n);
        for (size_t ch = 0; ch < 3; ++ch)
            for (size_t y = 0; y < c.h; ++y)
                for (size_t x = 0; x < c.w; ++x) {
                    float v = 0.2f + 0.5f * static_cast<float>((x * 7 + y * 3 + ch * 11) % 23) / 23.0f;
                    v += x % 9 == 4 ? 0.25f : 0.0f;
                    v -= y % 13 == 6 ? 0.15f : 0.0f;
                    host.image[(ch * c.h + y) * c.w + x] = std::clamp(v, 0.0f, 1.0f);
                }
        host.image_bytes.resize(host.image.size());
        for (size_t i = 0; i < host.image.size(); ++i)
            host.image_bytes[i] = static_cast<uint8_t>(std::lround(host.image[i] * 255.0f));
        if (!c.mask)
            return host;
        const bool segment = c.mode == ops::MaskPhotoMode::SegmentAndIgnore;
        host.mask.resize(n);
        host.mask_bytes.resize(n);
        for (size_t y = 0; y < c.h; ++y)
            for (size_t x = 0; x < c.w; ++x) {
                const size_t i = y * c.w + x;
                const bool hidden = (x >= c.w / 3 && x < c.w / 2 && y >= c.h / 4 && y < c.h / 2) ||
                                    (x == c.w - 3 && y == c.h / 5) || (x == 2 && y == c.h - 3);
                if (segment) {
                    host.mask_bytes[i] = hidden ? 0 : x < c.w / 6 ? 250
                                                  : y < c.h / 6   ? 251
                                                                  : 255;
                    host.mask[i] = static_cast<float>(host.mask_bytes[i]) / 255.0f;
                } else {
                    host.mask_bytes[i] = hidden ? 0 : static_cast<uint8_t>(1 + i % 200);
                    host.mask[i] = hidden ? 0.0f : i % 11 == 0 ? 1.0e-6f
                                                               : 0.25f + 0.75f * static_cast<float>(i % 13) / 12.0f;
                }
            }
        return host;
    }

    std::pair<Tensor, Tensor> ridge_inputs(const RidgeCase& c) {
        const RidgeHost host = ridge_host(c);
        const Tensor image = c.image == DataType::UInt8 ? upload_bytes(host.image_bytes, TensorShape({3, c.h, c.w}))
                                                        : upload(host.image, TensorShape({3, c.h, c.w}));
        if (!c.mask)
            return {image, Tensor{}};
        const TensorShape plane({c.h, c.w});
        return {image, *c.mask == DataType::UInt8 ? upload_bytes(host.mask_bytes, plane) : upload(host.mask, plane)};
    }

    // The ridge map on the CPU in double: the filters of structure_gpu.hpp (rounded to float, as the kernels get
    // them), its separable passes as plain loops over reflected rows and columns, then the normalization of
    // structure_main operations 2 and 3.
    std::vector<double> ridge_reference(const RidgeCase& c, const RidgeHost& host) {
        const int h = static_cast<int>(c.h), w = static_cast<int>(c.w);
        const size_t n = c.h * c.w;
        const bool bytes = c.image == DataType::UInt8, segment = c.mode == ops::MaskPhotoMode::SegmentAndIgnore;
        std::vector<double> luma(n);
        std::vector<char> keep(n, 1);
        for (size_t i = 0; i < n; ++i) {
            const auto channel = [&](const size_t ch) {
                return bytes ? static_cast<double>(host.image_bytes[ch * n + i]) / 255.0
                             : static_cast<double>(host.image[ch * n + i]);
            };
            luma[i] = 0.2126 * channel(0) + 0.7152 * channel(1) + 0.0722 * channel(2);
            if (c.mask) {
                const bool mask_bytes = *c.mask == DataType::UInt8;
                keep[i] = segment ? (mask_bytes ? host.mask_bytes[i] > 250 : host.mask[i] > 250.5f / 255.0f)
                                  : (mask_bytes ? host.mask_bytes[i] != 0 : host.mask[i] > 0.0f);
            }
        }
        const auto reflect = [](int i, const int size) {
            if (i < 0)
                i = -i - 1;
            i %= 2 * size;
            return i < size ? i : 2 * size - 1 - i;
        };
        std::vector<double> best(n, 0.0);
        bool first = true;
        for (const double sigma : {0.8, 1.2, 1.8, 2.5}) {
            const int radius = static_cast<int>(4.0 * sigma + 0.5);
            std::vector<double> fg, fd1, fd2;
            double sum = 0;
            for (int k = -radius; k <= radius; ++k)
                sum += std::exp(-0.5 * k * k / (sigma * sigma));
            for (int k = -radius; k <= radius; ++k) {
                const double g = std::exp(-0.5 * k * k / (sigma * sigma)) / sum;
                fg.push_back(static_cast<float>(g));
                fd1.push_back(static_cast<float>(k * g / (sigma * sigma)));
                fd2.push_back(static_cast<float>((k * k / (sigma * sigma) - 1.0) * g / (sigma * sigma)));
            }
            const double sigma_sq = static_cast<float>(sigma * sigma);
            std::vector<double> hg(n), hd1(n), hd2(n);
            std::vector<char> hvalid(n);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    double g = 0, d1 = 0, d2 = 0;
                    bool valid = true;
                    for (int k = -radius; k <= radius; ++k) {
                        const size_t source = static_cast<size_t>(y) * w + reflect(x + k, w);
                        valid = valid && keep[source];
                        g += luma[source] * fg[k + radius];
                        d1 += luma[source] * fd1[k + radius];
                        d2 += luma[source] * fd2[k + radius];
                    }
                    const size_t i = static_cast<size_t>(y) * w + x;
                    hg[i] = g;
                    hd1[i] = d1;
                    hd2[i] = d2;
                    hvalid[i] = valid;
                }
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    bool valid = true;
                    double hxx = 0, hxy = 0, hyy = 0;
                    for (int k = -radius; k <= radius; ++k) {
                        const size_t source = static_cast<size_t>(reflect(y + k, h)) * w + x;
                        valid = valid && hvalid[source];
                        hxx += hd2[source] * fg[k + radius];
                        hxy += hd1[source] * fd1[k + radius];
                        hyy += hg[source] * fd2[k + radius];
                    }
                    if (!valid)
                        continue;
                    hxy *= sigma_sq;
                    hyy *= sigma_sq;
                    const double difference = hxx * sigma_sq - hyy, trace = hxx * sigma_sq + hyy;
                    const double disc = std::sqrt(std::max(difference * difference + 4.0 * hxy * hxy, 0.0));
                    const double a = std::abs(0.5 * (trace + disc)), b = std::abs(0.5 * (trace - disc));
                    const double hi = std::max(a, b), lo = std::min(a, b);
                    const size_t i = static_cast<size_t>(y) * w + x;
                    const double response = hi * (hi - lo) / (hi + lo + 1e-6);
                    best[i] = first ? response : std::max(best[i], response);
                }
            first = false;
        }
        double total = 0, count = 0, lowest = std::numeric_limits<double>::infinity(), highest = -lowest;
        for (size_t i = 0; i < n; ++i) {
            if (!keep[i])
                continue;
            total += best[i];
            count += 1;
            lowest = std::min(lowest, luma[i]);
            highest = std::max(highest, luma[i]);
        }
        const bool range = c.mask ? count > 0 : highest > lowest;
        const double scale = total > 0 && range ? (c.mask ? count : static_cast<double>(n)) / total : 0.0;
        for (double& value : best)
            value *= scale;
        return best;
    }

    // The ridge map against ridge_reference, within float rounding: sizes below the 10-pixel halo (reflected more
    // than once) and sizes that are not multiples of the 16x32 tile, u8 and float images, no mask, a float mask, and
    // a byte mask in the segment-and-ignore mode.
    TEST_P(PortableLosses, RidgeMapMatchesCpuReference) {
        for (const auto& [h, w] : std::array<std::pair<size_t, size_t>, 4>{{{2, 3}, {5, 7}, {29, 37}, {33, 129}}})
            for (const DataType image : {DataType::UInt8, DataType::Float32})
                for (const RidgeCase& c : {RidgeCase{h, w, image, std::nullopt},
                                           RidgeCase{h, w, image, DataType::Float32},
                                           RidgeCase{h, w, image, DataType::UInt8, ops::MaskPhotoMode::SegmentAndIgnore}}) {
                    SCOPED_TRACE(ridge_case_name(c));
                    const auto [input, mask] = ridge_inputs(c);
                    Tensor map = Tensor::empty({h, w}, Device::GPU);
                    k::RidgeWorkspace workspace;
                    table_->structure->ridge(input, map, workspace, mask, c.mode);
                    expect_field(download(map), ridge_reference(c, ridge_host(c)), 1e-4, 1e-5, "ridge map");
                }
    }

    // PhotoParams::loss_only, which the crop-box report sets: the loss keeps its bits, and on Metal the gradients,
    // the maps and their workspace are left out and the next call reuses the small arena. A backend that ignores the
    // flag computes the full evaluate, so the loss check holds there too.
    TEST_P(PortableLosses, PhotometricLossOnlyKeepsTheLoss) {
        const Shape4 fixture{1, 3, 32, 40};
        const Shape4 odd{2, 3, 21, 27};
        const std::array<std::pair<PhotoCase, const char*>, 7> cases{{
            {{ops::PhotoPath::SSIM, fixture}, "ssim"},
            {{ops::PhotoPath::Fused, odd, true}, "fused_batch_u8"},
            {{ops::PhotoPath::Decoupled, fixture}, "decoupled"},
            {{ops::PhotoPath::MaskedFused, odd, true, true}, "masked_fused_u8"},
            {{ops::PhotoPath::MaskedDecoupled, fixture, false, true}, "masked_decoupled"},
            {{ops::PhotoPath::MaskedDecoupled, fixture}, "masked_decoupled_soft"},
            {{ops::PhotoPath::MaskedDecoupled, odd, true, true}, "masked_decoupled_batch_u8"},
        }};
        for (const auto& [pc, name] : cases) {
            SCOPED_TRACE(name);
            const PhotoInputs in = photo_inputs(pc, 11);
            const bool decoupled = pc.path == ops::PhotoPath::Decoupled || pc.path == ops::PhotoPath::MaskedDecoupled;
            const bool masked = pc.path == ops::PhotoPath::MaskedFused || pc.path == ops::PhotoPath::MaskedDecoupled;
            const auto run = [&](ops::PhotoSaved& saved, const bool loss_only, Tensor& grad, Tensor& grad_raw) {
                Tensor loss;
                table_->photometric->evaluate(saved, in.x, decoupled ? in.raw : Tensor{}, in.y, masked ? in.mask : Tensor{},
                                              {.path = pc.path, .ssim_weight = pc.weight, .valid_padding = true, .denominator = 0.f, .loss_only = loss_only},
                                              loss, grad, grad_raw);
                return std::bit_cast<uint32_t>(download(loss)[0]);
            };
            ops::PhotoSaved full{.backend = table_->photometric->create()};
            ops::PhotoSaved only{.backend = table_->photometric->create()};
            Tensor grad, grad_raw, no_grad, no_grad_raw;
            const uint32_t expected = run(full, false, grad, grad_raw);
            EXPECT_EQ(run(only, true, no_grad, no_grad_raw), expected);
            if (GetParam() != GpuBackend::Metal)
                continue;
            EXPECT_FALSE(no_grad.is_valid());
            EXPECT_FALSE(no_grad_raw.is_valid());
            EXPECT_FALSE(only.ssim_map.is_valid());
            EXPECT_FALSE(only.cs_map.is_valid());
            // The partial sums and the result: a few KB whatever the image size.
            const size_t allocated = table_->photometric->workspace_bytes(only).allocated;
            EXPECT_LE(allocated, 2048 * sizeof(float) + 2 * 256);
            EXPECT_LT(allocated, table_->photometric->workspace_bytes(full).allocated);
            EXPECT_EQ(run(only, true, no_grad, no_grad_raw), expected);
            EXPECT_EQ(table_->photometric->workspace_bytes(only).allocated, allocated);
        }
    }

    // got and want equal bit for bit; reports the number of differing values and the first one.
    void expect_same_bits(const std::vector<float>& got, const std::vector<float>& want, const std::string& what) {
        ASSERT_EQ(got.size(), want.size()) << what;
        size_t differing = 0, first = 0;
        for (size_t i = 0; i < got.size(); ++i)
            if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0 && differing++ == 0)
                first = i;
        EXPECT_EQ(differing, 0u) << what << ": first at " << first << " got " << got[first] << " want " << want[first];
    }

    // |got - want| <= rel * |want| for every value: each within rel of its own size, zeros exact.
    void expect_each_relative(const std::vector<float>& got, const Field& want, const double rel,
                              const std::string& what) {
        ASSERT_EQ(got.size(), want.size()) << what;
        size_t outside = 0, first = 0;
        for (size_t i = 0; i < got.size(); ++i)
            if (!(std::abs(got[i] - want[i]) <= rel * std::abs(want[i])) && outside++ == 0)
                first = i;
        EXPECT_EQ(outside, 0u) << what << ": first at " << first << " got " << got[first] << " want " << want[first];
    }

    // PhotometricOps::weighted_error_map against the trainer's sequence it replaces (map_to_error, then the mask, ROI
    // and structure weights as separate passes) and against the same products in double on the CPU, for each kind of
    // mask weight.
    TEST_P(PortableLosses, WeightedErrorMapMatchesTheSequence) {
        if (table_->photometric->weighted_error_map == nullptr)
            GTEST_SKIP() << "No weighted_error_map on " << lfs::core::gpu_backend_name(GetParam());
        constexpr size_t H = 37, W = 53, n = H * W;
        const TensorShape plane({H, W});
        // SSIM maps run past [0, 1] at the edges of their range; structure values past [0, 4].
        const auto map_values = uniform(n, 3, -0.4f, 1.3f);
        auto soft = uniform(n, 5);
        for (size_t i = 0; i < n; i += 7)
            soft[i] = i % 2 == 0 ? 0.0f : 1.0f;
        std::vector<uint8_t> bytes(n);
        for (size_t i = 0; i < n; ++i)
            bytes[i] = static_cast<uint8_t>(i % 9 == 0 ? 255 - i % 4 : (i * 37) % 256);
        const auto roi_values = uniform(n, 7, 0.1f, 1.0f);
        const auto structure_values = uniform(n, 9, -0.5f, 5.0f);
        // The masks' values as the weights read them: bytes as 0 to 255, bools as 0 or 1.
        const std::vector<float> byte_values(bytes.begin(), bytes.end());
        std::vector<float> bool_values(n);
        for (size_t i = 0; i < n; ++i)
            bool_values[i] = soft[i] > 0.5f ? 1.0f : 0.0f;

        const Tensor map = upload(map_values, TensorShape({1, 1, H, W}));
        const Tensor float_mask = upload(soft, plane);
        const Tensor byte_mask = upload_bytes(bytes, plane);
        const Tensor bool_mask = float_mask.gt(0.5f);
        const Tensor roi = upload(roi_values, plane);
        const Tensor structure = upload(structure_values, plane);

        struct Case {
            const char* name;
            Tensor mask;
            const std::vector<float>* mask_values;
            bool keep_band;
            float keep_min;
            bool roi;
            float gain;
        };
        const std::array<Case, 9> cases{{
            {"no weights", {}, nullptr, false, 0.f, false, 0.f},
            {"float mask, roi, structure", float_mask, &soft, false, 0.f, true, 1.0f},
            {"float keep band, roi, structure", float_mask, &soft, true, k::kMaskKeepMin, true, 0.35f},
            {"byte mask, structure", byte_mask, &byte_values, false, 0.f, false, 1.0f},
            {"byte keep band, roi", byte_mask, &byte_values, true, 250.f, true, 0.f},
            {"bool mask, roi, structure", bool_mask, &bool_values, false, 0.f, true, 0.35f},
            {"bool keep band", bool_mask, &bool_values, true, 250.f, false, 0.f},
            {"roi only", {}, nullptr, false, 0.f, true, 0.f},
            {"structure only", {}, nullptr, false, 0.f, false, 2.5f},
        }};
        // Both GPU sides round the same product of float factors once per operation: the subtraction, up to three
        // multiplies and the structure weight's multiply and add, at most six roundings of epsilon / 2. Each value so
        // lies within 3 epsilons of the exact product, which the CPU computes in double, and within 5 of the other
        // side (the subtraction rounds alike on both), whatever order of the multiplies or contraction of the weight
        // a compiler picks. Each value is compared with its own size, so zeros stay exact. On an Apple M3 Ultra the
        // map equals the sequence bit for bit in all nine cases.
        constexpr double rel = 8.0 * std::numeric_limits<float>::epsilon();
        for (const Case& c : cases) {
            SCOPED_TRACE(c.name);
            Field reference(n);
            for (size_t i = 0; i < n; ++i) {
                double weight = 1.0;
                if (c.mask_values != nullptr) {
                    const float value = (*c.mask_values)[i];
                    weight = c.keep_band ? (value > c.keep_min ? 1.0 : 0.0) : value;
                }
                if (c.roi)
                    weight *= roi_values[i];
                if (c.gain > 0.f)
                    weight *= 1.0 + c.gain * std::clamp(static_cast<double>(structure_values[i]), 0.0, 4.0);
                reference[i] = std::max(1.0 - map_values[i], 0.0) * weight;
            }

            // The trainer's sequence, in place on a copy of the map.
            Tensor sequence_map = map.clone();
            Tensor sequence = sequence_map.reshape({static_cast<int>(H), static_cast<int>(W)});
            table_->photometric->map_to_error(sequence_map, sequence);
            if (c.keep_band)
                sequence.mul_(c.mask.gt(c.keep_min).to(DataType::Float32));
            else if (c.mask.is_valid())
                sequence.mul_(c.mask.dtype() == DataType::Float32 ? c.mask : c.mask.to(DataType::Float32));
            if (c.roi)
                sequence.mul_(roi);
            if (c.gain > 0.f)
                k::structure_densification_weight(sequence, structure, c.gain);

            const ops::ErrorMapWeights weights{.mask = c.mask,
                                               .mask_keep_band = c.keep_band,
                                               .mask_keep_min = c.keep_min,
                                               .roi = c.roi ? roi : Tensor{},
                                               .structure = structure,
                                               .structure_gain = c.gain};
            // In place, as the trainer calls it, and into a separate map: one kernel on the same values, so the two
            // agree bit for bit.
            Tensor fused_map = map.clone();
            Tensor fused = fused_map.reshape({static_cast<int>(H), static_cast<int>(W)});
            table_->photometric->weighted_error_map(fused_map, fused, weights);
            Tensor separate = Tensor::empty(plane, Device::GPU);
            table_->photometric->weighted_error_map(map, separate, weights);
            const auto fused_values = download(fused);
            expect_same_bits(download(separate), fused_values, "separate against in place");
            expect_each_relative(fused_values, reference, rel, "against the CPU");
            expect_each_relative(fused_values, widen(download(sequence)), rel, "against the sequence");
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, PortableLosses, testing::Values(GpuBackend::Metal, GpuBackend::Vulkan),
                             [](const auto& info) { return std::string(lfs::core::gpu_backend_name(info.param)); });

} // namespace
