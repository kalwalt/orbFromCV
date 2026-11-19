#include "image_utils.hpp"
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

} // namespace

void resizeBilinear(const Image8U& src, Image8U& dst, int new_width, int new_height) {
    if (new_width == 0 || new_height == 0) return;

    dst = Image8U(new_width, new_height);

    const AxisMapping x = buildAxisMapping(src.cols, new_width);
    const AxisMapping y = buildAxisMapping(src.rows, new_height);

    // Both passes are fixed point, as in OpenCV's INTER_LINEAR_EXACT: the
    // horizontal pass carries 8 fractional bits per sample, the vertical pass
    // accumulates those and rounds once over 16 bits. Output is therefore
    // byte-identical to cv2.resize(..., INTER_LINEAR_EXACT), the interpolation
    // ORB_Impl builds its pyramid with (#40).
    for (int row = 0; row < new_height; ++row) {
        const int y0 = y.offset[row], y1 = y0 + 1;
        const int b0 = y.weight0[row], b1 = y.weight1[row];
        const uint8_t* row0 = src.ptr(y0);
        const uint8_t* row1 = src.ptr(std::min(y1, src.rows - 1));
        uint8_t* out = dst.ptr(row);

        for (int col = 0; col < new_width; ++col) {
            const int x0 = x.offset[col], x1 = x0 + 1;
            const int a0 = x.weight0[col], a1 = x.weight1[col];

            // Horizontal pass: 8 fractional bits, since a0 + a1 = 256 and
            // samples are <= 255, so the sum peaks at 65280 and needs no
            // saturation (OpenCV saturates its uint16 intermediate here too).
            const int h0 = a0 * row0[x0] + a1 * row0[std::min(x1, src.cols - 1)];
            const int h1 = a0 * row1[x0] + a1 * row1[std::min(x1, src.cols - 1)];

            // Vertical pass: 16 fractional bits in total, rounded once.
            const int value = (b0 * h0 + b1 * h1 + (1 << 15)) >> 16;
            out[col] = static_cast<uint8_t>(std::min(255, std::max(0, value)));
        }
    }
}

void gaussianBlur7x7(const Image8U& src, Image8U& dst) {
    if (src.cols == 0 || src.rows == 0) return;

    const int cols = src.cols, rows = src.rows;
    Image8U temp(cols, rows);   // horizontal-pass intermediate
    Image8U result(cols, rows); // vertical-pass output

    // 1D Gaussian kernel for sigma = 2.0 (approx integer weights, sum = 256),
    // symmetric so each tap pair shares one multiply. Integer math only.
    const int kernel[7] = {18, 33, 49, 56, 49, 33, 18};
    const int kRadius = 3;
    const int k0 = kernel[0], k1 = kernel[1], k2 = kernel[2], k3 = kernel[3];

    // BORDER_REFLECT_101: index -1 -> 1, n -> n-2. Reflects repeatedly (as
    // OpenCV's borderInterpolate does) so dimensions below 4 px stay in bounds;
    // a 1-pixel dimension always maps to 0. Only border strips and per-row
    // lookups call this, never the interior loop.
    auto reflect = [](int i, int n) {
        if (n == 1) return 0;
        while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - i - 2;
        return i;
    };

    // Horizontal pass: reflection only on the two 3-pixel border strips, the
    // interior runs branch-free on the symmetric kernel.
    const int leftEnd = std::min(kRadius, cols);
    const int rightStart = std::max(kRadius, cols - kRadius);
    for (int y = 0; y < rows; ++y) {
        const uint8_t* s = src.ptr(y);
        uint8_t* t = temp.ptr(y);

        auto borderPixel = [&](int x) {
            int sum = 0;
            for (int k = -kRadius; k <= kRadius; ++k) sum += s[reflect(x + k, cols)] * kernel[k + kRadius];
            t[x] = static_cast<uint8_t>(sum >> 8);
        };
        for (int x = 0; x < leftEnd; ++x) borderPixel(x);
        for (int x = kRadius; x < cols - kRadius; ++x) {
            int sum = k0 * (s[x - 3] + s[x + 3]) + k1 * (s[x - 2] + s[x + 2]) +
                      k2 * (s[x - 1] + s[x + 1]) + k3 * s[x];
            t[x] = static_cast<uint8_t>(sum >> 8);
        }
        for (int x = rightStart; x < cols; ++x) borderPixel(x);
    }

    // Vertical pass as row accumulation: the seven source rows are resolved
    // (with reflection) once per output row, then read contiguously.
    for (int y = 0; y < rows; ++y) {
        const uint8_t* r[7];
        for (int k = 0; k < 7; ++k) r[k] = temp.ptr(reflect(y + k - kRadius, rows));
        uint8_t* out = result.ptr(y);
        for (int x = 0; x < cols; ++x) {
            int sum = k0 * (r[0][x] + r[6][x]) + k1 * (r[1][x] + r[5][x]) +
                      k2 * (r[2][x] + r[4][x]) + k3 * r[3][x];
            out[x] = static_cast<uint8_t>(sum >> 8);
        }
    }

    // Deferred until every read of `src` is done, so gaussianBlur7x7(x, x)
    // (dst aliasing src - exactly how ORB::detectAndCompute calls this) is
    // safe: reassigning dst any earlier would overwrite src's data out from
    // under the horizontal pass before it could read it.
    dst = std::move(result);
}