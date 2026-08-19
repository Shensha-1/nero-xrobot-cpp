#include "xrobot_cpp/nero_safety.hpp"
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <utility>
namespace xrobot_cpp {
NeroCommandInterlock::NeroCommandInterlock(Vec7 lower, Vec7 upper, double timeout_s, double max_step) : lower_(std::move(lower)), upper_(std::move(upper)), timeout_ns_(static_cast<std::uint64_t>(timeout_s * 1e9)), max_step_(max_step) { if (!(timeout_s > 0.0) || !(max_step > 0.0) || !(lower_.array() < upper_.array()).all()) throw std::invalid_argument("invalid interlock limits"); }
void NeroCommandInterlock::notify_control_input(std::uint64_t now) { last_input_ns_ = now; }
NeroSafetyState NeroCommandInterlock::observe(const NeroFeedback& feedback, std::uint64_t now) { const bool fresh = feedback.monotonic_ns && (now <= feedback.monotonic_ns || now - feedback.monotonic_ns <= timeout_ns_); if (!fresh) state_ = NeroSafetyState::kAwaitFeedback; else if (!std::all_of(feedback.joint_valid.begin(), feedback.joint_valid.end(), [](bool value){return value;}) || !std::all_of(feedback.driver_valid.begin(), feedback.driver_valid.end(), [](bool value){return value;}) || !std::all_of(feedback.joint_enabled.begin(), feedback.joint_enabled.end(), [](bool value){return value;})) state_ = NeroSafetyState::kAwaitEnable; else if (std::any_of(feedback.joint_fault.begin(), feedback.joint_fault.end(), [](bool value){return value;})) state_ = NeroSafetyState::kFault; else if (!last_input_ns_ || now < last_input_ns_ || now - last_input_ns_ > timeout_ns_) state_ = NeroSafetyState::kWatchdogTimeout; else state_ = NeroSafetyState::kReady; return state_; }
bool NeroCommandInterlock::permits(const Vec7& target, const NeroFeedback& feedback, std::uint64_t now, std::string* reason) { const NeroSafetyState observed = observe(feedback, now); if (observed != NeroSafetyState::kReady) { if (reason) { switch (observed) { case NeroSafetyState::kAwaitFeedback: *reason = "feedback is stale or incomplete"; break; case NeroSafetyState::kAwaitEnable: *reason = "one or more joints are not enabled or driver feedback is incomplete"; break; case NeroSafetyState::kFault: *reason = "a joint driver reports a fault"; break; case NeroSafetyState::kWatchdogTimeout: *reason = "control watchdog expired"; break; case NeroSafetyState::kReady: break; } } return false; } if (!target.allFinite()) { if (reason) *reason = "target is non-finite"; return false; } if ((target.array() < lower_.array()).any() || (target.array() > upper_.array()).any()) { if (reason) { Eigen::Index joint = 0; for (; joint < target.size(); ++joint) if (target[joint] < lower_[joint] || target[joint] > upper_[joint]) break; std::ostringstream message; message << "target exceeds URDF joint limits: J" << (joint + 1) << " target=" << target[joint] << "rad, range=[" << lower_[joint] << ", " << upper_[joint] << "]rad"; *reason = message.str(); } return false; } if ((target - feedback.joints).cwiseAbs().maxCoeff() > max_step_) { if (reason) *reason = "target exceeds maximum measured-joint step"; return false; } if (reason) reason->clear(); return true; }
NeroSafetyState NeroCommandInterlock::state() const { return state_; }

NeroTorqueSafetyGate::NeroTorqueSafetyGate(Vec7 lower, Vec7 upper, double feedback_timeout_s, NeroTorqueSafetyLimits limits)
    : position_interlock_(std::move(lower), std::move(upper), feedback_timeout_s, limits.max_tracking_error_rad),
      limits_(std::move(limits)) {
  if (!limits_.max_velocity_rad_s.allFinite() || !limits_.max_torque_nm.allFinite() ||
      (limits_.max_velocity_rad_s.array() <= 0.0).any() || (limits_.max_torque_nm.array() <= 0.0).any() ||
      !std::isfinite(limits_.max_tracking_error_rad) || limits_.max_tracking_error_rad <= 0.0) {
    throw std::invalid_argument("invalid torque safety limits");
  }
}

void NeroTorqueSafetyGate::notify_keepalive(std::uint64_t now_ns) { position_interlock_.notify_control_input(now_ns); }

NeroSafetyState NeroTorqueSafetyGate::observe(const NeroFeedback& feedback, std::uint64_t now_ns) {
  return position_interlock_.observe(feedback, now_ns);
}

bool NeroTorqueSafetyGate::permits(const JointTrajectoryState& desired, const Vec7& measured_velocity,
                                   const Vec7& torque_nm, const NeroFeedback& feedback,
                                   std::uint64_t now_ns, std::string* reason) {
  if (!measured_velocity.allFinite() || !torque_nm.allFinite() || !desired.velocity.allFinite() ||
      !desired.acceleration.allFinite()) {
    if (reason) *reason = "velocity, acceleration, or torque is non-finite";
    return false;
  }
  if (!position_interlock_.permits(desired.position, feedback, now_ns, reason)) return false;
  if ((measured_velocity.array().abs() > limits_.max_velocity_rad_s.array()).any()) {
    if (reason) {
      Eigen::Index joint = 0;
      (measured_velocity.array().abs() - limits_.max_velocity_rad_s.array()).maxCoeff(&joint);
      std::ostringstream message;
      message << "measured joint velocity exceeds safety limit: J" << (joint + 1)
              << "=" << measured_velocity[joint] << "rad/s, limit=" << limits_.max_velocity_rad_s[joint] << "rad/s";
      *reason = message.str();
    }
    return false;
  }
  if ((desired.velocity.array().abs() > limits_.max_velocity_rad_s.array()).any()) {
    if (reason) *reason = "desired joint velocity exceeds safety limit";
    return false;
  }
  if ((torque_nm.array().abs() > limits_.max_torque_nm.array()).any()) {
    if (reason) *reason = "computed torque exceeds safety limit";
    return false;
  }
  if (reason) reason->clear();
  return true;
}
}  // namespace xrobot_cpp
