#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "xrobot_cpp/control_core.hpp"

namespace xrobot_cpp {

using Mat7 = Eigen::Matrix<double, 7, 7>;

struct JointTrajectoryState {
  Vec7 position{Vec7::Zero()};
  Vec7 velocity{Vec7::Zero()};
  Vec7 acceleration{Vec7::Zero()};
};

struct DynamicsTerms {
  Mat7 mass{Mat7::Zero()};
  Vec7 coriolis{Vec7::Zero()};
  Vec7 gravity{Vec7::Zero()};
};

struct MitSetpoint {
  double position_rad{};
  double velocity_rad_s{};
  double kp_nm_rad{};
  double kd_nm_s_rad{};
  double feedforward_torque_nm{};
};

// One target pose is solved as one 6-D task. The weights and soft joint-limit
// barrier avoid splitting translation and orientation into competing IK passes.
struct ContinuousIkOptions {
  double position_weight{1.0};
  double orientation_weight{1.2};
  double position_tolerance_m{0.0015};
  double orientation_tolerance_rad{0.05};
  double continuity_weight{0.015};
  double joint_limit_weight{0.00002};
  double soft_limit_margin_rad{0.06};
  double damping{0.025};
  double max_iteration_step_rad{0.025};
  int max_iterations{80};
};

struct ContinuousIkResult {
  Vec7 joints{Vec7::Zero()};
  double position_error_m{};
  double orientation_error_rad{};
  double minimum_joint_margin_rad{};
};

class JerkLimitedJointPlanner {
 public:
  JerkLimitedJointPlanner(Vec7 max_velocity_rad_s, Vec7 max_acceleration_rad_s2, Vec7 max_jerk_rad_s3);
  void reset(const Vec7& position);
  void set_goal(const Vec7& position);
  [[nodiscard]] const Vec7& goal() const { return goal_; }
  [[nodiscard]] const JointTrajectoryState& state() const { return state_; }
  [[nodiscard]] JointTrajectoryState step(double dt_s);

 private:
  Vec7 max_velocity_rad_s_;
  Vec7 max_acceleration_rad_s2_;
  Vec7 max_jerk_rad_s3_;
  Vec7 goal_{Vec7::Zero()};
  JointTrajectoryState state_{};
  bool initialized_{false};
};

struct ComputedTorqueCommand {
  JointTrajectoryState desired;
  DynamicsTerms dynamics;
  Vec7 commanded_acceleration{Vec7::Zero()};
  Vec7 reflected_inertia_torque_nm{Vec7::Zero()};
  Vec7 viscous_friction_torque_nm{Vec7::Zero()};
  Vec7 coulomb_friction_torque_nm{Vec7::Zero()};
  Vec7 virtual_spring_torque_nm{Vec7::Zero()};
  Vec7 virtual_damping_torque_nm{Vec7::Zero()};
  Vec7 feedforward_torque_nm{Vec7::Zero()};
};

// Drive-side terms not represented by a rigid-link URDF. They are opt-in:
// real hardware keeps the zero default until torque calibration is available.
struct JointDriveModel {
  Vec7 reflected_inertia_kg_m2{Vec7::Zero()};
  Vec7 viscous_friction_nm_s_rad{Vec7::Zero()};
  Vec7 coulomb_friction_nm{Vec7::Zero()};
  double coulomb_smoothing_rad_s{0.01};

  [[nodiscard]] static JointDriveModel mujoco_default();
};

// KDL-backed NERO model. The URDF is the source of geometry, inertia,
// gravity, and the seven joint limits used by IK and inverse dynamics.
class NeroDynamicsModel {
 public:
  explicit NeroDynamicsModel(const std::string& urdf_path, std::string base_link = "base_link",
                             std::string tip_link = "link7");
  ~NeroDynamicsModel();
  NeroDynamicsModel(const NeroDynamicsModel&) = delete;
  NeroDynamicsModel& operator=(const NeroDynamicsModel&) = delete;
  NeroDynamicsModel(NeroDynamicsModel&&) noexcept;
  NeroDynamicsModel& operator=(NeroDynamicsModel&&) noexcept;

  [[nodiscard]] Pose forward_kinematics(const Vec7& joints) const;
  [[nodiscard]] Vec7 solve_ik(const Pose& target, const Vec7& seed, double orientation_tolerance_rad = 1e-3,
                              double position_tolerance_m = 5e-4) const;
  [[nodiscard]] ContinuousIkResult solve_continuous_ik(const Pose& target, const Vec7& seed,
                                                        const ContinuousIkOptions& options = {}) const;
  // First follows the previous branch. If that branch cannot meet the TCP task
  // or approaches a limit, retry deterministic seeds toward joint-space center
  // and select the valid solution closest to the previous branch.
  [[nodiscard]] ContinuousIkResult solve_redundant_continuous_ik(
      const Pose& target, const Vec7& reference, const ContinuousIkOptions& options = {},
      double minimum_joint_margin_rad = 0.05) const;
  // Bounded resolved-rate IK step for continuous teleoperation.
  [[nodiscard]] Vec7 solve_ik_step(const Pose& target, const Vec7& seed, double max_joint_step_rad) const;
  [[nodiscard]] DynamicsTerms dynamics(const Vec7& joints, const Vec7& velocity) const;
  [[nodiscard]] const Vec7& lower_limits() const;
  [[nodiscard]] const Vec7& upper_limits() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Estimates qdot from fresh joint position feedback. A long feedback stall
// resets the estimate so a stale delta cannot become a false high velocity.
class JointVelocityEstimator {
 public:
  explicit JointVelocityEstimator(double cutoff_hz = 12.0);
  void reset(const Vec7& position, Clock::time_point now);
  [[nodiscard]] Vec7 update(const Vec7& position, Clock::time_point now);

 private:
  double cutoff_hz_;
  bool initialized_{false};
  Vec7 position_{Vec7::Zero()};
  Vec7 velocity_{Vec7::Zero()};
  Clock::time_point last_{};
};

// Computed-torque outer loop. NERO receives the result through its MIT
// impedance interface, rather than a bare unsafe torque command.
class ComputedTorqueController {
 public:
  ComputedTorqueController(Vec7 spring_stiffness_nm_rad, Vec7 damping_nm_s_rad, Vec7 torque_limit_nm,
                           JointDriveModel drive_model = {});
  [[nodiscard]] ComputedTorqueCommand compute(const DynamicsTerms& dynamics,
                                               const Vec7& measured_position,
                                               const Vec7& measured_velocity,
                                               const JointTrajectoryState& desired) const;
  [[nodiscard]] ComputedTorqueCommand compute(const NeroDynamicsModel& model,
                                               const Vec7& measured_position,
                                               const Vec7& measured_velocity,
                                               const JointTrajectoryState& desired) const;
  [[nodiscard]] MitSetpoint mit_setpoint(const ComputedTorqueCommand& command,
                                         std::size_t joint_index,
                                         double inner_kp_nm_rad = 4.0,
                                         double inner_kd_nm_s_rad = 0.35) const;

 private:
  Vec7 spring_stiffness_nm_rad_;
  Vec7 damping_nm_s_rad_;
  Vec7 torque_limit_nm_;
  JointDriveModel drive_model_;
};

}  // namespace xrobot_cpp
