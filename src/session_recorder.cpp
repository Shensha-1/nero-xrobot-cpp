#include "xrobot_cpp/session_recorder.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace xrobot_cpp {
SessionRecorder::SessionRecorder(const std::filesystem::path& output_jsonl) {
  std::filesystem::create_directories(output_jsonl.parent_path());
  output_.open(output_jsonl, std::ios::out | std::ios::app);
  if (!output_) throw std::runtime_error("failed to open session recording: " + output_jsonl.string());
}

void SessionRecorder::record_xr(const XrFrame& frame, std::uint64_t monotonic_ns) {
  std::ostringstream out;
  out << "{\"event\":\"xr\",\"monotonic_ns\":" << monotonic_ns
      << ",\"xr_timestamp_ns\":" << frame.timestamp_ns << ",\"sequence\":" << frame.sequence
      << ",\"deadman\":" << (frame.deadman ? "true" : "false")
      << ",\"trigger\":" << frame.trigger << ",\"grip\":" << frame.grip << "}";
  write_line(out.str());
}

void SessionRecorder::record_robot(const NeroFeedback& feedback) {
  write_line("{\"event\":\"robot_state\",\"monotonic_ns\":" +
             std::to_string(feedback.monotonic_ns) + ",\"joints_rad\":" + vector_json(feedback.joints) + "}");
}

void SessionRecorder::record_action(const Vec7& action, std::uint64_t monotonic_ns) {
  write_line("{\"event\":\"action\",\"monotonic_ns\":" + std::to_string(monotonic_ns) +
             ",\"joint_target_rad\":" + vector_json(action) + "}");
}

void SessionRecorder::record_rgb_reference(std::string_view camera, std::string_view path,
                                           std::uint64_t monotonic_ns) {
  write_line("{\"event\":\"rgb\",\"monotonic_ns\":" + std::to_string(monotonic_ns) +
             ",\"camera\":\"" + escape_json(camera) + "\",\"path\":\"" +
             escape_json(path) + "\"}");
}

void SessionRecorder::write_line(const std::string& line) {
  std::lock_guard lock(mutex_);
  output_ << line << '\n';
  output_.flush();
}

std::string SessionRecorder::vector_json(const Vec7& values) {
  std::ostringstream out;
  out << std::setprecision(17) << '[';
  for (Eigen::Index index = 0; index < values.size(); ++index) {
    if (index > 0) out << ',';
    out << values[index];
  }
  return out.str() + ']';
}

std::string SessionRecorder::escape_json(std::string_view value) {
  std::string result;
  for (const char character : value) {
    if (character == '"' || character == '\\') result.push_back('\\');
    result.push_back(character);
  }
  return result;
}
}  // namespace xrobot_cpp
