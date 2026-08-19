#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include <Eigen/Geometry>

namespace xrobot_cpp {

using Clock = std::chrono::steady_clock;
using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec7 = Eigen::Matrix<double, 7, 1>;

struct Pose {
  Vec3 position{Vec3::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
};

struct XrFrame {
  std::uint64_t sequence{};
  std::uint64_t timestamp_ns{};
  std::string device_id;
  Pose controller;
  Pose headset;
  double trigger{};
  double grip{};
  bool deadman{};
};

struct TcpTarget {
  Pose pose;
  Vec3 prediction_translation{Vec3::Zero()};
  Vec3 prediction_rotation{Vec3::Zero()};
};

class OneEuroFilter6 {
 public:
  OneEuroFilter6(double min_cutoff_hz, double beta, double derivative_cutoff_hz);
  void anchor(const Vec6& value, Clock::time_point now);
  [[nodiscard]] Vec6 filter(const Vec6& value, Clock::time_point now);
  [[nodiscard]] const Vec6& derivative() const { return derivative_; }

 private:
  [[nodiscard]] static double alpha(double cutoff_hz, double dt_s);
  double min_cutoff_hz_;
  double beta_;
  double derivative_cutoff_hz_;
  bool initialized_{false};
  Vec6 raw_{Vec6::Zero()};
  Vec6 value_{Vec6::Zero()};
  Vec6 derivative_{Vec6::Zero()};
  Clock::time_point last_{};
};

class RelativeMapper {
 public:
  RelativeMapper(double translation_scale, double rotation_scale,
                 double position_deadband_m, double rotation_deadband_rad,
                 Eigen::Matrix3d base_from_xr);
  void rebase(const XrFrame& frame, const Pose& tcp, Clock::time_point now);
  [[nodiscard]] TcpTarget target(const XrFrame& frame, Clock::time_point now);

 private:
  [[nodiscard]] static Vec3 rotation_vector(const Eigen::Quaterniond& rotation);
  [[nodiscard]] static Eigen::Quaterniond exp_rotation(const Vec3& rotation_vector);

  double translation_scale_;
  double rotation_scale_;
  double position_deadband_m_;
  double rotation_deadband_rad_;
  Eigen::Matrix3d base_from_xr_;
  OneEuroFilter6 filter_;
  bool active_{false};
  Pose controller_ref_;
  Pose tcp_ref_;
};

enum class GateState { kIdle, kActivate, kTrack, kRelease, kTimeout };

class SafetyGate {
 public:
  explicit SafetyGate(double timeout_s);
  [[nodiscard]] GateState observe(const XrFrame& frame, Clock::time_point now);
  [[nodiscard]] GateState poll(Clock::time_point now);
  [[nodiscard]] bool active() const { return active_; }

 private:
  double timeout_s_;
  bool active_{false};
  bool release_seen_{false};
  std::optional<std::uint64_t> sequence_;
  std::optional<std::uint64_t> timestamp_ns_;
  std::optional<Clock::time_point> last_fresh_;
};

}  // namespace xrobot_cpp
