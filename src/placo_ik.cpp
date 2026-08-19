#include "xrobot_cpp/placo_ik.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iterator>
#include <regex>
#include <stdexcept>

#include <placo/kinematics/kinematics_solver.h>
#include <placo/model/robot_wrapper.h>

namespace xrobot_cpp {
namespace {
constexpr std::array<const char*, 7> kJointNames{"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"};

Eigen::Affine3d to_affine(const Pose& pose) {
  if (!pose.position.allFinite() || !pose.orientation.coeffs().allFinite()) throw std::invalid_argument("Placo IK target must be finite");
  Eigen::Affine3d result = Eigen::Affine3d::Identity();
  result.translation() = pose.position;
  result.linear() = pose.orientation.normalized().toRotationMatrix();
  return result;
}

double orientation_error(const Eigen::Matrix3d& target, const Eigen::Matrix3d& actual) {
  return Eigen::AngleAxisd(target * actual.transpose()).angle();
}

std::string load_kinematic_urdf(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot open Placo URDF: " + path);
  const std::string urdf((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  // Placo builds geometry even though this IK backend has no collision task.
  return std::regex_replace(urdf, std::regex(R"(<(visual|collision)(\s[^>]*)?>[\s\S]*?</\1>)"), "");
}
}  // namespace

struct PlacoNeroIkSolver::Impl {
  placo::model::RobotWrapper robot;
  placo::kinematics::KinematicsSolver solver;
  placo::kinematics::FrameTask task;
  placo::kinematics::JointsTask* posture{};
  placo::kinematics::ManipulabilityTask* manipulability{};
  Vec7 hard_lower{Vec7::Zero()};
  Vec7 hard_upper{Vec7::Zero()};
  std::string tip_link;

  Impl(const std::string& urdf_path, const std::string& tcp_link)
      : robot(urdf_path, placo::model::RobotWrapper::Flags::IGNORE_COLLISIONS, load_kinematic_urdf(urdf_path)),
        solver(robot), task(solver.add_frame_task(tcp_link)), tip_link(tcp_link) {
    solver.mask_fbase(true);
    solver.enable_joint_limits(true);
    solver.enable_velocity_limits(true);
    task.configure("nero_tcp", "soft", 1.0, 1.0);
    posture = &solver.add_joints_task();
    posture->configure("nero_posture", "soft", 0.01);
    for (const char* name : kJointNames) posture->set_joint(name, 0.0);
    manipulability = &solver.add_manipulability_task(tcp_link, "both", 0.01);
    manipulability->configure("nero_manipulability", "soft", 0.001);
    for (Eigen::Index index = 0; index < 7; ++index) {
      const auto [lower, upper] = robot.get_joint_limits(kJointNames[static_cast<std::size_t>(index)]);
      hard_lower[index] = lower;
      hard_upper[index] = upper;
    }
    solver.add_regularization_task(1e-7);
  }
};

PlacoNeroIkSolver::PlacoNeroIkSolver(const std::string& urdf_path, std::string tip_link) : impl_(std::make_unique<Impl>(urdf_path, tip_link)) {}
PlacoNeroIkSolver::~PlacoNeroIkSolver() = default;
PlacoNeroIkSolver::PlacoNeroIkSolver(PlacoNeroIkSolver&&) noexcept = default;
PlacoNeroIkSolver& PlacoNeroIkSolver::operator=(PlacoNeroIkSolver&&) noexcept = default;

ContinuousIkResult PlacoNeroIkSolver::solve(const Pose& target, const Vec7& measured, const ContinuousIkOptions& options, double dt_s, double max_joint_velocity_rad_s) {
  if (!measured.allFinite() || !std::isfinite(dt_s) || dt_s <= 0.0 || !std::isfinite(max_joint_velocity_rad_s) || max_joint_velocity_rad_s <= 0.0) throw std::invalid_argument("invalid Placo IK state or timestep");
  if (!std::isfinite(options.position_weight) || !std::isfinite(options.orientation_weight) || options.position_weight <= 0.0 || options.orientation_weight <= 0.0) throw std::invalid_argument("invalid Placo IK task weights");
  const double soft_margin = std::max(0.0, options.soft_limit_margin_rad);
  for (Eigen::Index index = 0; index < 7; ++index) {
    const char* name = kJointNames[static_cast<std::size_t>(index)];
    const double lower = impl_->hard_lower[index];
    const double upper = impl_->hard_upper[index];
    const double center = 0.5 * (lower + upper);
    const double soft_lower = std::min(center, lower + soft_margin);
    const double soft_upper = std::max(center, upper - soft_margin);
    // Keep the current state feasible but prevent any further command into a soft boundary.
    impl_->robot.set_joint_limits(name, std::min(soft_lower, measured[index]), std::max(soft_upper, measured[index]));
    const double distance = std::min(measured[index] - lower, upper - measured[index]);
    const double activation = soft_margin > 0.0 ? std::clamp((soft_margin - distance) / soft_margin, 0.0, 1.0) : 0.0;
    impl_->robot.set_joint(name, measured[index]);
    impl_->robot.set_joint_velocity(name, 0.0);
    impl_->robot.set_velocity_limit(name, max_joint_velocity_rad_s);
    impl_->posture->set_joint(name, measured[index] + 0.10 * activation * (center - measured[index]));
  }
  impl_->posture->configure("nero_posture", "soft", options.continuity_weight + 0.05);
  impl_->robot.update_kinematics();
  impl_->solver.dt = dt_s;
  impl_->task.set_T_world_frame(to_affine(target));
  impl_->task.configure("nero_tcp", "soft", options.position_weight, options.orientation_weight);
  const Eigen::VectorXd delta = impl_->solver.solve(false);
  if (delta.rows() != impl_->robot.model.nv || !delta.allFinite()) throw std::runtime_error("Placo QP IK returned an invalid joint increment");
  ContinuousIkResult result;
  result.joints = measured;
  for (Eigen::Index index = 0; index < 7; ++index) {
    const char* name = kJointNames[static_cast<std::size_t>(index)];
    const int velocity_offset = impl_->robot.get_joint_v_offset(name);
    const double lower = impl_->hard_lower[index];
    const double upper = impl_->hard_upper[index];
    result.joints[index] = std::clamp(measured[index] + delta[velocity_offset], lower, upper);
    const double margin = std::min(result.joints[index] - lower, upper - result.joints[index]);
    result.minimum_joint_margin_rad = index == 0 ? margin : std::min(result.minimum_joint_margin_rad, margin);
    impl_->robot.set_joint(name, result.joints[index]);
  }
  impl_->robot.update_kinematics();
  const Eigen::Affine3d actual = impl_->robot.get_T_world_frame(impl_->tip_link);
  result.position_error_m = (target.position - actual.translation()).norm();
  result.orientation_error_rad = orientation_error(target.orientation.normalized().toRotationMatrix(), actual.rotation());
  return result;
}

}  // namespace xrobot_cpp
