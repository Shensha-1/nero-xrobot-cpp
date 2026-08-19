#pragma once

#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

#include "xrobot_cpp/control_core.hpp"
#include "xrobot_cpp/nero_can.hpp"

namespace xrobot_cpp {

// One clock domain for XR, CAN, planning, and externally captured RGB frame references.
class SessionRecorder {
 public:
  explicit SessionRecorder(const std::filesystem::path& output_jsonl);
  void record_xr(const XrFrame& frame, std::uint64_t monotonic_ns);
  void record_robot(const NeroFeedback& feedback);
  void record_action(const Vec7& action, std::uint64_t monotonic_ns);
  void record_rgb_reference(std::string_view camera, std::string_view path,
                            std::uint64_t monotonic_ns);

 private:
  void write_line(const std::string& line);
  static std::string vector_json(const Vec7& values);
  static std::string escape_json(std::string_view value);

  std::ofstream output_;
  std::mutex mutex_;
};

}  // namespace xrobot_cpp
