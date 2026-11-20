// Internal header: compile-time detection of the SIMD target (issue #19).
// Not installed; the public API only exposes orbSimdActive().
//
// ORB_ENABLE_SIMD comes from CMake (option ORB_ENABLE_SIMD, PRIVATE define on
// orb_lib). ORB_SIMD_SSE2 / ORB_SIMD_NEON / ORB_SIMD_WASM say which 128-bit
// target the compiler offers; ORB_SIMD is 1 when the option is on and one of
// them is available. Every kernel keeps a scalar path, so a build with
// ORB_SIMD == 0 is a complete library, just slower.
#ifndef ORB_SIMD_UTILS_HPP
#define ORB_SIMD_UTILS_HPP

#ifndef ORB_ENABLE_SIMD
#define ORB_ENABLE_SIMD 0
#endif

// MSVC never defines __SSE2__; on x64 SSE2 is the baseline (_M_X64) and on
// 32-bit x86 /arch:SSE2 or higher sets _M_IX86_FP >= 2.
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define ORB_SIMD_SSE2 1
#else
#define ORB_SIMD_SSE2 0
#endif

#if defined(__ARM_NEON) || defined(_M_ARM64)
#define ORB_SIMD_NEON 1
#else
#define ORB_SIMD_NEON 0
#endif

#if defined(__wasm_simd128__)
#define ORB_SIMD_WASM 1
#else
#define ORB_SIMD_WASM 0
#endif

#if ORB_ENABLE_SIMD && (ORB_SIMD_SSE2 || ORB_SIMD_NEON || ORB_SIMD_WASM)
#define ORB_SIMD 1
#else
#define ORB_SIMD 0
#endif

// ---------------------------------------------------------------------------
// 128-bit wrappers. One struct per lane type, the same free functions on every
// target, and nothing beyond what the kernels use. Float arithmetic is mul and
// add as separate operations: no fused multiply-add anywhere, since the blur
// has to reproduce OpenCV's two-rounding float sums (see image_utils.cpp).
// ---------------------------------------------------------------------------
#if ORB_SIMD

#include <cstdint>
#include <cstring>

#if ORB_SIMD_SSE2
#include <emmintrin.h>
#elif ORB_SIMD_NEON
#include <arm_neon.h>
#elif ORB_SIMD_WASM
#include <wasm_simd128.h>
#endif

namespace simd {

#if ORB_SIMD_SSE2
struct f32x4 { __m128 v; };
struct i32x4 { __m128i v; };

inline f32x4 load(const float* p) { return {_mm_loadu_ps(p)}; }
inline void store(float* p, f32x4 a) { _mm_storeu_ps(p, a.v); }
inline f32x4 splat(float x) { return {_mm_set1_ps(x)}; }
inline f32x4 mul(f32x4 a, f32x4 b) { return {_mm_mul_ps(a.v, b.v)}; }
inline f32x4 add(f32x4 a, f32x4 b) { return {_mm_add_ps(a.v, b.v)}; }
inline i32x4 splat_i32(int32_t x) { return {_mm_set1_epi32(x)}; }
inline i32x4 sub(i32x4 a, i32x4 b) { return {_mm_sub_epi32(a.v, b.v)}; }
inline i32x4 add(i32x4 a, i32x4 b) { return {_mm_add_epi32(a.v, b.v)}; }
// Low 32 bits of each lane product. SSE2 has no 32-bit mullo (that is
// SSE4.1), so two 32x32->64 multiplies on the even and odd lanes are merged.
inline i32x4 mul(i32x4 a, i32x4 b) {
    __m128i even = _mm_mul_epu32(a.v, b.v);
    __m128i odd = _mm_mul_epu32(_mm_srli_si128(a.v, 4), _mm_srli_si128(b.v, 4));
    return {_mm_unpacklo_epi32(_mm_shuffle_epi32(even, _MM_SHUFFLE(0, 0, 2, 0)),
                               _mm_shuffle_epi32(odd, _MM_SHUFFLE(0, 0, 2, 0)))};
}
template <int N> inline i32x4 shr_arith(i32x4 a) { return {_mm_srai_epi32(a.v, N)}; }
// Four uint16 zero-extended to four int32. Exactly 8 bytes are read.
inline i32x4 load_u16x4_as_i32x4(const uint16_t* p) {
    __m128i h = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    return {_mm_unpacklo_epi16(h, _mm_setzero_si128())};
}

// Four bytes, zero-extended to four floats. Exactly 4 bytes are read.
inline f32x4 load_u8x4_as_f32x4(const uint8_t* p) {
    int32_t w;
    std::memcpy(&w, p, 4);
    const __m128i zero = _mm_setzero_si128();
    __m128i b = _mm_cvtsi32_si128(w);
    b = _mm_unpacklo_epi8(b, zero);
    b = _mm_unpacklo_epi16(b, zero);
    return {_mm_cvtepi32_ps(b)};
}
// Truncation toward zero, as static_cast<int>(float).
inline i32x4 trunc_to_i32x4(f32x4 a) { return {_mm_cvttps_epi32(a.v)}; }
// Clamp each lane to 0..255 and write 4 bytes. Exactly 4 bytes are written.
inline void store_u8x4_saturated(uint8_t* p, i32x4 a) {
    __m128i h = _mm_packs_epi32(a.v, a.v);   // -> int16, saturating
    __m128i b = _mm_packus_epi16(h, h);      // -> uint8, saturating (negatives -> 0)
    int32_t w = _mm_cvtsi128_si32(b);
    std::memcpy(p, &w, 4);
}

#elif ORB_SIMD_NEON
struct f32x4 { float32x4_t v; };
struct i32x4 { int32x4_t v; };

inline f32x4 load(const float* p) { return {vld1q_f32(p)}; }
inline void store(float* p, f32x4 a) { vst1q_f32(p, a.v); }
inline f32x4 splat(float x) { return {vdupq_n_f32(x)}; }
inline f32x4 mul(f32x4 a, f32x4 b) { return {vmulq_f32(a.v, b.v)}; }
inline f32x4 add(f32x4 a, f32x4 b) { return {vaddq_f32(a.v, b.v)}; }
inline i32x4 splat_i32(int32_t x) { return {vdupq_n_s32(x)}; }
inline i32x4 sub(i32x4 a, i32x4 b) { return {vsubq_s32(a.v, b.v)}; }
inline i32x4 add(i32x4 a, i32x4 b) { return {vaddq_s32(a.v, b.v)}; }
inline i32x4 mul(i32x4 a, i32x4 b) { return {vmulq_s32(a.v, b.v)}; }
template <int N> inline i32x4 shr_arith(i32x4 a) { return {vshrq_n_s32(a.v, N)}; }
inline i32x4 load_u16x4_as_i32x4(const uint16_t* p) {
    return {vreinterpretq_s32_u32(vmovl_u16(vld1_u16(p)))};
}

inline f32x4 load_u8x4_as_f32x4(const uint8_t* p) {
    uint32_t w;
    std::memcpy(&w, p, 4);
    uint8x8_t b = vreinterpret_u8_u32(vdup_n_u32(w));
    uint16x8_t h = vmovl_u8(b);
    uint32x4_t i = vmovl_u16(vget_low_u16(h));
    return {vcvtq_f32_u32(i)};
}
inline i32x4 trunc_to_i32x4(f32x4 a) { return {vcvtq_s32_f32(a.v)}; }
inline void store_u8x4_saturated(uint8_t* p, i32x4 a) {
    int16x4_t h = vqmovn_s32(a.v);
    uint8x8_t b = vqmovun_s16(vcombine_s16(h, h));
    uint32_t w = vget_lane_u32(vreinterpret_u32_u8(b), 0);
    std::memcpy(p, &w, 4);
}

#elif ORB_SIMD_WASM
struct f32x4 { v128_t v; };
struct i32x4 { v128_t v; };

inline f32x4 load(const float* p) { return {wasm_v128_load(p)}; }
inline void store(float* p, f32x4 a) { wasm_v128_store(p, a.v); }
inline f32x4 splat(float x) { return {wasm_f32x4_splat(x)}; }
inline f32x4 mul(f32x4 a, f32x4 b) { return {wasm_f32x4_mul(a.v, b.v)}; }
inline f32x4 add(f32x4 a, f32x4 b) { return {wasm_f32x4_add(a.v, b.v)}; }
inline i32x4 splat_i32(int32_t x) { return {wasm_i32x4_splat(x)}; }
inline i32x4 sub(i32x4 a, i32x4 b) { return {wasm_i32x4_sub(a.v, b.v)}; }
inline i32x4 add(i32x4 a, i32x4 b) { return {wasm_i32x4_add(a.v, b.v)}; }
inline i32x4 mul(i32x4 a, i32x4 b) { return {wasm_i32x4_mul(a.v, b.v)}; }
template <int N> inline i32x4 shr_arith(i32x4 a) { return {wasm_i32x4_shr(a.v, N)}; }
inline i32x4 load_u16x4_as_i32x4(const uint16_t* p) {
    return {wasm_u32x4_extend_low_u16x8(wasm_v128_load64_zero(p))};
}

inline f32x4 load_u8x4_as_f32x4(const uint8_t* p) {
    v128_t b = wasm_v128_load32_zero(p);
    v128_t h = wasm_u16x8_extend_low_u8x16(b);
    v128_t i = wasm_u32x4_extend_low_u16x8(h);
    return {wasm_f32x4_convert_u32x4(i)};
}
inline i32x4 trunc_to_i32x4(f32x4 a) { return {wasm_i32x4_trunc_sat_f32x4(a.v)}; }
inline void store_u8x4_saturated(uint8_t* p, i32x4 a) {
    v128_t h = wasm_i16x8_narrow_i32x4(a.v, a.v);
    v128_t b = wasm_u8x16_narrow_i16x8(h, h);
    wasm_v128_store32_lane(p, b, 0);
}
#endif

} // namespace simd

#endif // ORB_SIMD

#endif // ORB_SIMD_UTILS_HPP
