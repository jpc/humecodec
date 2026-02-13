// Runtime loader for libavdevice — avoids compile-time linking so that
// only one copy of the library is ever loaded (prevents macOS ObjC class
// conflicts when the user already has FFmpeg installed).
//
// Search order:
//   1. System / user-installed copy (standard dynamic-linker search)
//   2. Bundled fallback next to the extension module
//
// If neither is found, the functions gracefully degrade (register_all is
// a no-op and version returns -1).
#pragma once

#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace humecodec {
namespace avdevice_loader {

namespace detail {

using register_all_fn = void (*)();
using version_fn = unsigned (*)();

#ifdef _WIN32

inline void* open_lib(const char* name) {
  return reinterpret_cast<void*>(LoadLibraryA(name));
}
inline void* get_sym(void* h, const char* name) {
  return reinterpret_cast<void*>(
      GetProcAddress(reinterpret_cast<HMODULE>(h), name));
}

inline std::string get_module_dir() {
  HMODULE hm = nullptr;
  GetModuleHandleExA(
      GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
          GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
      reinterpret_cast<LPCSTR>(&get_module_dir), &hm);
  char buf[MAX_PATH];
  GetModuleFileNameA(hm, buf, sizeof(buf));
  std::string path(buf);
  auto pos = path.rfind('\\');
  return (pos != std::string::npos) ? path.substr(0, pos) : ".";
}

#else // Unix

inline void* open_lib(const char* name) {
  return dlopen(name, RTLD_LAZY | RTLD_LOCAL);
}
inline void* get_sym(void* h, const char* name) { return dlsym(h, name); }

inline std::string get_module_dir() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&get_module_dir), &info) &&
      info.dli_fname) {
    std::string path(info.dli_fname);
    auto pos = path.rfind('/');
    return (pos != std::string::npos) ? path.substr(0, pos) : ".";
  }
  return ".";
}

#endif // _WIN32

struct State {
  register_all_fn p_register_all = nullptr;
  version_fn p_version = nullptr;
  bool loaded = false;
};

inline State& load() {
  static State s = []() {
    State st;
    void* handle = nullptr;

    // --- 1. Try system / user-installed copy first ---
#ifdef __APPLE__
    // On macOS, try versioned names that match common FFmpeg soversions
    for (const char* name :
         {"libavdevice.dylib", "libavdevice.62.dylib", "libavdevice.61.dylib",
          "libavdevice.60.dylib", "libavdevice.59.dylib",
          "libavdevice.58.dylib"}) {
      handle = open_lib(name);
      if (handle)
        break;
    }
#elif defined(_WIN32)
    for (const char* name :
         {"avdevice-62.dll", "avdevice-61.dll", "avdevice-60.dll",
          "avdevice-59.dll", "avdevice-58.dll", "avdevice.dll"}) {
      handle = open_lib(name);
      if (handle)
        break;
    }
#else // Linux
    for (const char* name :
         {"libavdevice.so", "libavdevice.so.62", "libavdevice.so.61",
          "libavdevice.so.60", "libavdevice.so.59", "libavdevice.so.58"}) {
      handle = open_lib(name);
      if (handle)
        break;
    }
#endif

    // --- 2. Fall back to the bundled copy next to the extension ---
    if (!handle) {
      std::string dir = get_module_dir();
#ifdef __APPLE__
      // delocate puts bundled dylibs in <pkg>/.dylibs/
      for (const char* name :
           {"libavdevice.62.dylib", "libavdevice.61.dylib",
            "libavdevice.60.dylib", "libavdevice.59.dylib",
            "libavdevice.58.dylib", "libavdevice.dylib"}) {
        handle = open_lib((dir + "/.dylibs/" + name).c_str());
        if (handle)
          break;
      }
#elif defined(_WIN32)
      for (const char* name :
           {"avdevice-62.dll", "avdevice-61.dll", "avdevice-60.dll",
            "avdevice-59.dll", "avdevice-58.dll"}) {
        handle = open_lib((dir + "\\" + name).c_str());
        if (handle)
          break;
      }
#else // Linux — auditwheel puts bundled .so in <pkg>.libs/
      for (const char* name :
           {"libavdevice.so.62", "libavdevice.so.61", "libavdevice.so.60",
            "libavdevice.so.59", "libavdevice.so.58"}) {
        handle = open_lib((dir + "/../humecodec.libs/" + name).c_str());
        if (handle)
          break;
      }
#endif
    }

    if (handle) {
      st.p_register_all = reinterpret_cast<register_all_fn>(
          get_sym(handle, "avdevice_register_all"));
      st.p_version =
          reinterpret_cast<version_fn>(get_sym(handle, "avdevice_version"));
      st.loaded = (st.p_register_all != nullptr);
    }
    return st;
  }();
  return s;
}

} // namespace detail

/// Call avdevice_register_all() if available; no-op otherwise.
inline void register_all() {
  auto& st = detail::load();
  if (st.p_register_all)
    st.p_register_all();
}

/// Return the avdevice version, or -1 if the library is not available.
inline int version() {
  auto& st = detail::load();
  return st.p_version ? static_cast<int>(st.p_version()) : -1;
}

/// Whether libavdevice was found and loaded.
inline bool available() { return detail::load().loaded; }

} // namespace avdevice_loader
} // namespace humecodec
