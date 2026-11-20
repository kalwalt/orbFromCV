#include "image_utils.hpp"
#include "simd_utils.hpp"
#include "thread_pool.hpp"
#include <cmath>
#include <algorithm>
#include <utility>
#include <vector>

namespace {

// Fixed-point scale of the interpolation coefficients: OpenCV's
// INTER_RESIZE_COEF_BITS, 8 of them used here (ufixedpoint16's fixedShift).
// Two passes at 8 fractional bits each give a single rounding over 16 bits.
const int kCoefBits = 8;
const int kCoefOne = 1 << kCoefBits;

// Floor division, for the negative source coordinates that the centre-aligned
// mapping produces on the first destination pixel of an upscale.
long long floorDiv(long long n, long long d) {
    long long q = n / d;
    if (n % d != 0 && ((n < 0) != (d < 0))) --q;
    return q;
}

// round-half-to-even of num / den (both positive), matching cvRound on the
// softdouble OpenCV feeds it while building its coefficients.
long long roundHalfEven(long long num, long long den) {
    long long q = floorDiv(num, den);
    long long r = num - q * den;
    if (2 * r > den) return q + 1;
    if (2 * r == den) return q + (q & 1);
    return q;
}

// One axis of the resize mapping, precomputed once per call as OpenCV's
// INTER_LINEAR_EXACT does it: destination pixel centres are aligned to source
// pixel centres, srcpos = scale * (d + 0.5) - 0.5 with scale = srcsize/dstsize.
// Outside the source the nearest edge pixel is replicated (offset clamped to
// the last column/row with a full weight), which is what OpenCV's min_x/max_x
// clamping amounts to.
struct AxisMapping {
    std::vector<int> offset;   // first source index
    std::vector<int> weight0;  // weight of `offset`, in 1/256 units, sum = 256
    std::vector<int> weight1;  // weight of `offset + 1`
};

AxisMapping buildAxisMapping(int srcSize, int dstSize) {
    AxisMapping m;
    m.offset.resize(dstSize);
    m.weight0.resize(dstSize);
    m.weight1.resize(dstSize);

    const long long den = 2LL * dstSize;
    for (int d = 0; d < dstSize; ++d) {
        // srcpos = (srcSize * (2d + 1) - dstSize) / (2 * dstSize), kept exact so
        // the fraction rounds the same way OpenCV's softdouble does.
        const long long num = static_cast<long long>(srcSize) * (2LL * d + 1) - dstSize;
        long long base = floorDiv(num, den);
        long long frac = num - base * den; // 0 <= frac < den

        int w0, w1;
        if (base < 0) {
            base = 0;
            w0 = kCoefOne;
            w1 = 0;
        } else if (base >= srcSize - 1) {
            base = srcSize - 1;
            w0 = kCoefOne;
            w1 = 0;
        } else {
            w1 = static_cast<int>(roundHalfEven(static_cast<long long>(kCoefOne) * frac, den));
            w0 = kCoefOne - w1;
        }
        m.offset[d] = static_cast<int>(base);
        m.weight0[d] = w0;
        m.weight1[d] = w1;
    }
    return m;
}

// Horizontal pass of one source row: a0 + a1 = 256 and samples are <= 255,
// so the sum peaks at 65280 and fits uint16 without saturation (OpenCV
// saturates its uint16 intermediate here too).
void resizeHorizontalPass(const ImageView8U& src, const AxisMapping& x, int newWidth, int srcRow, uint16_t* h) {
    const uint8_t* s = src.ptr(srcRow);
    const int lastCol = src.cols - 1;
    for (int col = 0; col < newWidth; ++col) {
        const int x0 = x.offset[col];
        const int x1 = std::min(x0 + 1, lastCol);
        h[col] = static_cast<uint16_t>(x.weight0[col] * s[x0] + x.weight1[col] * s[x1]);
    }
}

// Destination rows [rowBegin, rowEnd) of resizeBilinear. Both passes are
// fixed point, as in OpenCV's INTER_LINEAR_EXACT: the horizontal pass carries
// 8 fractional bits per sample, the vertical pass accumulates those and rounds
// once over 16 bits, so the output is byte-identical to
// cv2.resize(..., INTER_LINEAR_EXACT), the interpolation ORB_Impl builds its
// pyramid with (#40). The horizontal pass runs once per source row into a
// two-slot cache local to the band: a band's first row computes both of its
// source rows, after that consecutive rows share one whenever the scale is
// below 2.
void resizeRows(const ImageView8U& src, Image8U& dst, const AxisMapping& x, const AxisMapping& y,
                bool vec, int rowBegin, int rowEnd) {
    const int newWidth = dst.cols;
    std::vector<uint16_t> hrow[2] = {std::vector<uint16_t>(newWidth), std::vector<uint16_t>(newWidth)};
    int cachedRow[2] = {-1, -1};
    auto cachedPass = [&](int srcRow, int otherRow) -> const uint16_t* {
        for (int i = 0; i < 2; ++i)
            if (cachedRow[i] == srcRow) return hrow[i].data();
        // Evict the slot that does not hold the other row this output needs.
        const int slot = (cachedRow[0] == otherRow) ? 1 : 0;
        resizeHorizontalPass(src, x, newWidth, srcRow, hrow[slot].data());
        cachedRow[slot] = srcRow;
        return hrow[slot].data();
    };
    (void)vec;

    for (int row = rowBegin; row < rowEnd; ++row) {
        const int y0 = y.offset[row];
        const int y1 = std::min(y0 + 1, src.rows - 1);
        const int b0 = y.weight0[row], b1 = y.weight1[row];
        const uint16_t* h0 = cachedPass(y0, y1);
        const uint16_t* h1 = cachedPass(y1, y0);
        uint8_t* out = dst.ptr(row);

        // Vertical pass: 16 fractional bits in total, rounded once. b0 + b1 =
        // 256 and h <= 65280 keep the sum below 2^24 + 2^15, and the result in
        // 0..255, so the clamp only documents the range.
        int col = 0;
#if ORB_SIMD
        if (vec) {
            // Four columns per step, the same integer expression lane-wise;
            // the saturating store is the clamp. The horizontal pass stays
            // scalar: its x0 lookup is a gather, which SSE2 does not have.
            const simd::i32x4 vb0 = simd::splat_i32(b0), vb1 = simd::splat_i32(b1);
            const simd::i32x4 half = simd::splat_i32(1 << 15);
            for (; col + 4 <= newWidth; col += 4) {
                simd::i32x4 acc = simd::mul(vb0, simd::load_u16x4_as_i32x4(h0 + col));
                acc = simd::add(acc, simd::mul(vb1, simd::load_u16x4_as_i32x4(h1 + col)));
                acc = simd::shr_arith<16>(simd::add(acc, half));
                simd::store_u8x4_saturated(out + col, acc);
            }
        }
#endif
        for (; col < newWidth; ++col) {
            const int value = (b0 * h0[col] + b1 * h1[col] + (1 << 15)) >> 16;
            out[col] = static_cast<uint8_t>(std::min(255, std::max(0, value)));
        }
    }
}

} // namespace

void resizeBilinear(const ImageView8U& src, Image8U& dst, int new_width, int new_height,
                    bool useSimd, ThreadPool* pool) {
    if (new_width == 0 || new_height == 0) return;
#if ORB_SIMD
    const bool vec = useSimd;
#else
    (void)useSimd;
    const bool vec = false;
#endif

    dst = Image8U(new_width, new_height);

    const AxisMapping x = buildAxisMapping(src.cols, new_width);
    const AxisMapping y = buildAxisMapping(src.rows, new_height);

    // Plain functions rather than nested lambdas: MSVC did not inline the
    // captured-by-reference lambdas and the single-thread pyramid slowed by
    // 45%; g++ and clang were indifferent.
    if (pool) {
        pool->parallelFor(0, new_height, [&](int, int rowBegin, int rowEnd) {
            resizeRows(src, dst, x, y, vec, rowBegin, rowEnd);
        });
    } else {
        resizeRows(src, dst, x, y, vec, 0, new_height);
    }
}

void gaussianBlur7x7(const ImageView8U& src, Image8U& dst, bool useSimd, ThreadPool* pool) {
    if (src.cols == 0 || src.rows == 0) return;
#if ORB_SIMD
    const bool vec = useSimd;   // read only by the vector blocks below
#else
    (void)useSimd;
#endif

    const int cols = src.cols, rows = src.rows;

    // This is the blur cv::ORB applies to every pyramid level, reproduced bit
    // for bit (issue #39). ORB blurs each level as a submatrix of its padded
    // buffer, so cv::GaussianBlur skips its 8-bit fixed-point path and runs
    // the generic float separable filter: a 32-bit float kernel
    // (getGaussianKernel(7, 2, CV_32F)), a float row pass (RowFilter: taps
    // accumulated in order 0..6), a float column pass (SymmColumnFilter:
    // centre tap first, then each symmetric pair added in float and
    // accumulated), and one rounding to 8 bits at the end
    // (saturate_cast<uchar>: round half to even).
    // Verified identical to OpenCV's generic scalar path (cv2.setUseOptimized
    // off, and opencv.js) on 2.17 million pixels over 36 images, including
    // 1-3 px sizes and widths that are not SIMD multiples.
    static const float kernel[7] = {0.0701593235f, 0.131074876f, 0.190712824f, 0.216105938f,
                                    0.190712824f, 0.131074876f, 0.0701593235f};
    const int kRadius = 3;

    // a * b + c in float32 with two roundings, i.e. no fused multiply-add.
    // GCC contracts this into an FMA on targets that have one even across the
    // explicit cast, so CMakeLists.txt builds with -ffp-contract=off (GCC and
    // Clang; MSVC does not contract under its default /fp:precise). Every
    // build then computes the same bytes. This is what OpenCV computes on
    // targets without FMA, opencv.js included; OpenCV builds that dispatch to
    // FMA hardware differ from it on the rare pixels whose exact value sits on
    // a rounding tie (about 3 per million on noise), by 1 gray level.
    auto mulAdd = [](float a, float b, float c) {
        return static_cast<float>(static_cast<float>(a * b) + c);
    };

    // Round half to even to 0..255 (saturate_cast<uchar>(float) == cvRound).
    // Adding 1.5 * 2^23 forces the float to the integer grid under the default
    // round-to-nearest-even mode; valid for |v| < 2^22, far beyond 0..255 blur
    // sums. Much cheaper than std::lrint in the inner loop.
    auto roundToByte = [](float v) {
        const int r = static_cast<int>(v + 12582912.0f) - 12582912;
        return static_cast<uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
    };
    // BORDER_REFLECT_101: index -1 -> 1, n -> n-2. Reflects repeatedly (as
    // OpenCV's borderInterpolate does) so dimensions below 4 px stay in bounds;
    // a 1-pixel dimension always maps to 0. Only border strips and per-row
    // lookups call this, never the interior loop.
    auto reflect = [](int i, int n) {
        if (n == 1) return 0;
        while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - i - 2;
        return i;
    };

    // Row pass: float intermediate, as OpenCV's. Reflection only on the two
    // 3-pixel border strips; the interior reads the taps directly.
    std::vector<float> temp(static_cast<size_t>(rows) * cols);
    const int leftEnd = std::min(kRadius, cols);
    const int rightStart = std::max(kRadius, cols - kRadius);
    // Rows [yBegin, yEnd) of the row pass: each writes only its own rows of
    // temp, so the bands are independent.
    auto rowPass = [&](int, int yBegin, int yEnd) {
        for (int y = yBegin; y < yEnd; ++y) {
            const uint8_t* s = src.ptr(y);
            float* t = temp.data() + static_cast<size_t>(y) * cols;

            auto borderPixel = [&](int x) {
                float acc = kernel[0] * static_cast<float>(s[reflect(x - 3, cols)]);
                for (int k = 1; k < 7; ++k) acc = mulAdd(kernel[k], static_cast<float>(s[reflect(x - 3 + k, cols)]), acc);
                t[x] = acc;
            };
            for (int x = 0; x < leftEnd; ++x) borderPixel(x);
            // Interior, four pixels at a time: the same seven products summed in
            // the same order as the scalar loop below, so the bytes are identical.
            int x = kRadius;
    #if ORB_SIMD
            if (vec) {
                for (; x + 4 <= cols - kRadius; x += 4) {
                    const uint8_t* p = s + x - 3;
                    simd::f32x4 acc = simd::mul(simd::splat(kernel[0]), simd::load_u8x4_as_f32x4(p));
                    for (int k = 1; k < 7; ++k)
                        acc = simd::add(simd::mul(simd::splat(kernel[k]), simd::load_u8x4_as_f32x4(p + k)), acc);
                    simd::store(t + x, acc);
                }
            }
    #endif
            // Scalar interior (the whole row without SIMD, the <4 px tail with it).
            for (; x < cols - kRadius; ++x) {
                const uint8_t* p = s + x - 3;
                float acc = kernel[0] * static_cast<float>(p[0]);
                acc = mulAdd(kernel[1], static_cast<float>(p[1]), acc);
                acc = mulAdd(kernel[2], static_cast<float>(p[2]), acc);
                acc = mulAdd(kernel[3], static_cast<float>(p[3]), acc);
                acc = mulAdd(kernel[4], static_cast<float>(p[4]), acc);
                acc = mulAdd(kernel[5], static_cast<float>(p[5]), acc);
                acc = mulAdd(kernel[6], static_cast<float>(p[6]), acc);
                t[x] = acc;
            }
            for (int x = rightStart; x < cols; ++x) borderPixel(x);
        }

    };
    if (pool) pool->parallelFor(0, rows, rowPass);
    else rowPass(0, 0, rows);
    // Column pass as row accumulation: the seven source rows are resolved
    // (with reflection) once per output row, then read contiguously. Centre
    // tap first, then the symmetric pairs, as SymmColumnFilter does.
    Image8U result(cols, rows);
    // Rows [yBegin, yEnd) of the column pass: reads up to 3 rows of temp
    // outside the band (complete by now), writes only its own rows of result.
    auto colPass = [&](int, int yBegin, int yEnd) {
        for (int y = yBegin; y < yEnd; ++y) {
            const float* r[7];
            for (int k = 0; k < 7; ++k) r[k] = temp.data() + static_cast<size_t>(reflect(y + k - kRadius, rows)) * cols;
            uint8_t* out = result.ptr(y);
            int x = 0;
    #if ORB_SIMD
            if (vec) {
                const simd::f32x4 k0 = simd::splat(kernel[0]), k1 = simd::splat(kernel[1]),
                                  k2 = simd::splat(kernel[2]), k3 = simd::splat(kernel[3]);
                const simd::f32x4 magic = simd::splat(12582912.0f);
                const simd::i32x4 magicInt = simd::splat_i32(12582912);
                for (; x + 4 <= cols; x += 4) {
                    simd::f32x4 acc = simd::mul(k3, simd::load(r[3] + x));
                    acc = simd::add(simd::mul(k2, simd::add(simd::load(r[2] + x), simd::load(r[4] + x))), acc);
                    acc = simd::add(simd::mul(k1, simd::add(simd::load(r[1] + x), simd::load(r[5] + x))), acc);
                    acc = simd::add(simd::mul(k0, simd::add(simd::load(r[0] + x), simd::load(r[6] + x))), acc);
                    // roundToByte lane-wise: magic-number rounding, truncating
                    // convert, integer subtract, saturating narrow to 0..255.
                    simd::i32x4 r32 = simd::sub(simd::trunc_to_i32x4(simd::add(acc, magic)), magicInt);
                    simd::store_u8x4_saturated(out + x, r32);
                }
            }
    #endif
            for (; x < cols; ++x) {
                float acc = kernel[3] * r[3][x];
                acc = mulAdd(kernel[2], static_cast<float>(r[2][x] + r[4][x]), acc);
                acc = mulAdd(kernel[1], static_cast<float>(r[1][x] + r[5][x]), acc);
                acc = mulAdd(kernel[0], static_cast<float>(r[0][x] + r[6][x]), acc);
                out[x] = roundToByte(acc);
            }
        }

    };
    if (pool) pool->parallelFor(0, rows, colPass);
    else colPass(0, 0, rows);
    // Deferred until every read of `src` is done, so gaussianBlur7x7(x, x)
    // (dst aliasing src - exactly how ORB::detectAndCompute calls this) is
    // safe: reassigning dst any earlier would overwrite src's data out from
    // under the row pass before it could read it.
    dst = std::move(result);
}
