#pragma once
#include <cstdint>
#include <string>
#include "xrobot_cpp/dynamics_controller.hpp"
#include "xrobot_cpp/nero_can.hpp"
namespace xrobot_cpp {
enum class NeroSafetyState { kAwaitFeedback, kAwaitEnable, kReady, kFault, kWatchdogTimeout };
class NeroCommandInterlock { public: NeroCommandInterlock(Vec7 lower, Vec7 upper, double timeout_s, double max_step); void notify_control_input(std::uint64_t now); NeroSafetyState observe(const NeroFeedback& feedback, std::uint64_t now); bool permits(const Vec7& target, const NeroFeedback& feedback, std::uint64_t now, std::string* reason); [[nodiscard]] NeroSafetyState state() const; private: Vec7 lower_, upper_; std::uint64_t timeout_ns_{}; double max_step_{}; std::uint64_t last_input_ns_{}; NeroSafetyState state_{NeroSafetyState::kAwaitFeedback}; };

struct NeroTorqueSafetyLimits {
  Vec7 max_velocity_rad_s{Vec7::Constant(1.0)};
  Vec7 max_torque_nm{Vec7::Constant(1.0)};
  double max_tracking_error_rad{0.05};
};

class NeroTorqueSafetyGate {
 public:
  NeroTorqueSafetyGate(Vec7 lower, Vec7 upper, double feedback_timeout_s, NeroTorqueSafetyLimits limits);
  void notify_keepalive(std::uint64_t now_ns);
  [[nodiscard]] NeroSafetyState observe(const NeroFeedback& feedback, std::uint64_t now_ns);
  bool permits(const JointTrajectoryState& desired, const Vec7& measured_velocity, const Vec7& torque_nm,
               const NeroFeedback& feedback, std::uint64_t now_ns, std::string* reason);

 private:
  NeroCommandInterlock position_interlock_;
  NeroTorqueSafetyLimits limits_;
};
}  // namespace xrobot_cpp
