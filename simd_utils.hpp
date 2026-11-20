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

#endif // ORB_SIMD_UTILS_HPP
