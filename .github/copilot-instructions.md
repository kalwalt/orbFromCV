# Copilot Instructions

See `AGENTS.md` at the repo root for full project context, build commands,
and conventions — this file exists only because GitHub Copilot looks here
specifically.

## Quick Reference
- Standalone C++14 ORB (Oriented FAST and Rotated BRIEF) implementation, zero OpenCV dependency.
- Build: `mkdir build && cd build && cmake .. && cmake --build .`
- Run: `./orb_test <path_to_image>`
- No external deps beyond bundled `stb_image.h` — don't introduce new ones.
- Keep header (`.hpp`) / implementation (`.cpp`) split consistent with existing files.
- Don't hand-edit `orb_pattern.hpp` (generated BRIEF sampling pattern).
