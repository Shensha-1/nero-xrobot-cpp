#include "xrobot_cpp/control_core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace xrobot_cpp {
namespace {
constexpr double kPi = 3.14159265358979323846;

double elapsed_seconds(Clock::time_point from, Clock::time_point to) {
  return std::chrono::duration<double>(to - from).count();
}

void require_positive(double value, const char* name) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string(name) + " must be finite and positive");
  }
}
}  // namespace

OneEuroFilter6::OneEuroFilter6(double min_cutoff_hz, double beta,
                               double derivative_cutoff_hz)
    : min_cutoff_hz_(min_cutoff_hz), beta_(beta),
      derivative_cutoff_hz_(derivative_cutoff_hz) {
  require_positive(min_cutoff_hz_, "min cutoff");
  require_positive(derivative_cutoff_hz_, "derivative cutoff");
  if (!std::isfinite(beta_) || beta_ < 0.0) {
    throw std::invalid_argument("beta must be finite and non-negative");
  }
}

double OneEuroFilter6::alpha(double cutoff_hz, double dt_s) {
  return 1.0 / (1.0 + 1.0 / (2.0 * kPi * cutoff_hz * dt_s));
}

void OneEuroFilter6::anchor(const Vec6& value, Clock::time_point now) {
  raw_ = value;
  value_ = value;
  derivative_.setZero();
  last_ = now;
  initialized_ = true;
}

Vec6 OneEuroFilter6::filter(const Vec6& raw, Clock::time_point now) {
  if (!initialized_) {
    anchor(raw, now);
    return raw;
  }
  const double dt_s = std::clamp(elapsed_seconds(last_, now), 0.0, 0.10);
  if (dt_s <= 0.0) {
    return value_;
  }
  const double derivative_alpha = alpha(derivative_cutoff_hz_, dt_s);
  derivative_ = derivative_alpha * ((raw - raw_) / dt_s) +
                (1.0 - derivative_alpha) * derivative_;
  for (int index = 0; index < 6; ++index) {
    const double value_alpha = alpha(min_cutoff_hz_ + beta_ * std::abs(derivative_[index]), dt_s);
    value_[index] = value_alpha * raw[index] + (1.0 - value_alpha) * value_[index];
  }
  raw_ = raw;
  last_ = now;
  return value_;
}

RelativeMapper::RelativeMapper(double translation_scale, double rotation_scale,
                               double position_deadband_m, double rotation_deadband_rad,
                               Eigen::Matrix3d base_from_xr)
    : translation_scale_(translation_scale), rotation_scale_(rotation_scale),
      position_deadband_m_(position_deadband_m), rotation_deadband_rad_(rotation_deadband_rad),
      base_from_xr_(std::move(base_from_xr)), filter_(2.0, 0.25, 1.0) {
  require_positive(translation_scale_, "translation scale");
  require_positive(rotation_scale_, "rotation scale");
  if (!std::isfinite(position_deadband_m_) || !std::isfinite(rotation_deadband_rad_) ||
      position_deadband_m_ < 0.0 || rotation_deadband_rad_ < 0.0 ||
      !base_from_xr_.allFinite() ||
      !(base_from_xr_.transpose() * base_from_xr_).isApprox(Eigen::Matrix3d::Identity(), 1e-8) ||
      std::abs(base_from_xr_.determinant() - 1.0) > 1e-8) {
    throw std::invalid_argument("invalid mapper calibration");
  }
}

void RelativeMapper::rebase(const XrFrame& frame, const Pose& tcp, Clock::time_point now) {
  controller_ref_ = frame.controller;
  controller_ref_.orientation.normalize();
  tcp_ref_ = tcp;
  tcp_ref_.orientation.normalize();
  filter_.anchor(Vec6::Zero(), now);
  active_ = true;
}

Vec3 RelativeMapper::rotation_vector(const Eigen::Quaterniond& rotation) {
  Eigen::Quaterniond normalized = rotation.normalized();
  if (normalized.w() < 0.0) {
    normalized.coeffs() *= -1.0;
  }
  const Eigen::AngleAxisd angle_axis(normalized);
  if (angle_axis.angle() < 1e-9) {
    return Vec3::Zero();
  }
  return angle_axis.axis() * angle_axis.angle();
}

Eigen::Quaterniond RelativeMapper::exp_rotation(const Vec3& rotation_vector) {
  const double angle = rotation_vector.norm();
  if (angle < 1e-9) {
    return Eigen::Quaterniond::Identity();
  }
  return Eigen::Quaterniond(Eigen::AngleAxisd(angle, rotation_vector / angle));
}

TcpTarget RelativeMapper::target(const XrFrame& frame, Clock::time_point now) {
  if (!active_) {
    throw std::logic_error("mapper must be rebased before target generation");
  }
  const Vec3 raw_position = frame.controller.position - controller_ref_.position;
  const Eigen::Quaterniond raw_delta = frame.controller.orientation.normalized() *
                                       controller_ref_.orientation.conjugate();
  const Vec3 raw_rotation = rotation_vector(raw_delta);
  Vec6 raw;
  raw << raw_position, raw_rotation;
  const Vec6 filtered = filter_.filter(raw, now);
  const Vec6 velocity = filter_.derivative();

  Vec3 translation_prediction = base_from_xr_ * velocity.head<3>() * translation_scale_ * 0.012;
  if (translation_prediction.norm() > 0.004) {
    translation_prediction *= 0.004 / translation_prediction.norm();
  }
  Vec3 delta = base_from_xr_ * filtered.head<3>() * translation_scale_ + translation_prediction;
  if (delta.norm() < position_deadband_m_) {
    delta.setZero();
  }

  Vec3 rotation_prediction = velocity.tail<3>() * 0.012;
  if (rotation_prediction.norm() > 0.015) {
    rotation_prediction *= 0.015 / rotation_prediction.norm();
  }
  Vec3 rotation = (filtered.tail<3>() + rotation_prediction) * rotation_scale_;
  if (rotation.norm() < rotation_deadband_rad_) {
    rotation.setZero();
  }

  TcpTarget result;
  result.pose.position = tcp_ref_.position + delta;
  result.pose.orientation = Eigen::Quaterniond(base_from_xr_) * exp_rotation(rotation) *
                            Eigen::Quaterniond(base_from_xr_).conjugate() * tcp_ref_.orientation;
  result.pose.orientation.normalize();
  result.prediction_translation = translation_prediction;
  result.prediction_rotation = rotation_prediction;
  return result;
}

SafetyGate::SafetyGate(double timeout_s) : timeout_s_(timeout_s) {
  require_positive(timeout_s_, "XR timeout");
}

GateState SafetyGate::observe(const XrFrame& frame, Clock::time_point now) {
  if (sequence_ && frame.sequence <= *sequence_) {
    return poll(now);
  }
  sequence_ = frame.sequence;
  timestamp_ns_ = frame.timestamp_ns;
  last_fresh_ = now;
  if (!frame.deadman) {
    const bool was_active = active_;
    active_ = false;
    release_seen_ = true;
    return was_active ? GateState::kRelease : GateState::kIdle;
  }
  if (!active_) {
    active_ = true;
    return GateState::kActivate;
  }
  return GateState::kTrack;
}

GateState SafetyGate::poll(Clock::time_point now) {
  if (active_ && last_fresh_ && elapsed_seconds(*last_fresh_, now) > timeout_s_) {
    active_ = false;
    return GateState::kTimeout;
  }
  return GateState::kIdle;
}

}  // namespace xrobot_cpp
