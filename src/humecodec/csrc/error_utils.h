#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace humecodec {

namespace detail {

// Helper to format error messages
template <typename... Args>
std::string format_error_impl(std::ostringstream& oss) {
  return oss.str();
}

template <typename T, typename... Args>
std::string format_error_impl(std::ostringstream& oss, T&& first, Args&&... rest) {
  oss << std::forward<T>(first);
  return format_error_impl(oss, std::forward<Args>(rest)...);
}

template <typename... Args>
std::string format_error(Args&&... args) {
  std::ostringstream oss;
  return format_error_impl(oss, std::forward<Args>(args)...);
}

}  // namespace detail

}  // namespace humecodec

// HCODEC_CHECK - Runtime check that throws std::runtime_error on failure
// Use for user-facing errors (invalid input, configuration errors, etc.)
#define HCODEC_CHECK(cond, ...)                                          \
  do {                                                                   \
    if (!(cond)) {                                                       \
      throw std::runtime_error(                                          \
          ::humecodec::detail::format_error(__VA_ARGS__));             \
    }                                                                    \
  } while (false)

// HCODEC_INTERNAL_ASSERT - Internal assertion that throws std::logic_error
// Use for internal invariants that should never fail in correct code
#define HCODEC_INTERNAL_ASSERT(cond, ...)                                \
  do {                                                                   \
    if (!(cond)) {                                                       \
      throw std::logic_error(                                            \
          ::humecodec::detail::format_error(__VA_ARGS__));             \
    }                                                                    \
  } while (false)

// Debug-only assertion (no-op in release builds)
#ifdef NDEBUG
#define HCODEC_INTERNAL_ASSERT_DEBUG_ONLY(cond, ...) ((void)0)
#else
#define HCODEC_INTERNAL_ASSERT_DEBUG_ONLY(cond, ...) \
  HCODEC_INTERNAL_ASSERT(cond, __VA_ARGS__)
#endif

// Warning macro - prints to stderr but doesn't throw
#define HCODEC_WARN(...)                                                 \
  do {                                                                   \
    std::cerr << "Warning: "                                             \
              << ::humecodec::detail::format_error(__VA_ARGS__)        \
              << std::endl;                                              \
  } while (false)

// Warning that only prints once
#define HCODEC_WARN_ONCE(...)                                            \
  do {                                                                   \
    static bool warned = false;                                          \
    if (!warned) {                                                       \
      warned = true;                                                     \
      HCODEC_WARN(__VA_ARGS__);                                          \
    }                                                                    \
  } while (false)
