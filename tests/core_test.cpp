#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <array>

#include "xrobot_cpp/control_core.hpp"
#include "xrobot_cpp/dynamics_controller.hpp"
#include "xrobot_cpp/nero_can.hpp"
#include "xrobot_cpp/nero_safety.hpp"
#ifdef XROBOT_CPP_WITH_PLACO
#include "xrobot_cpp/placo_ik.hpp"
#endif
#ifdef XROBOT_CPP_WITH_MUJOCO
#include "xrobot_cpp/mujoco_simulation.hpp"
#endif

using namespace xrobot_cpp;

int main() {
  const auto start = Clock::now();

  OneEuroFilter6 filter(2.0, 0.0, 1.0);
  filter.anchor(Vec6::Zero(), start);
  Vec6 step = Vec6::Zero();
  step[0] = 1.0;
  const Vec6 filtered = filter.filter(step, start + std::chrono::milliseconds(100));
  assert(filtered[0] > 0.0 && filtered[0] < 1.0);

  SafetyGate gate(0.30);
  XrFrame frame;
  frame.sequence = 1;
  frame.timestamp_ns = 1;
  frame.device_id = "test";
  frame.deadman = true;
  assert(gate.observe(frame, start) == GateState::kActivate);
  assert(gate.active());
  assert(gate.poll(start + std::chrono::milliseconds(301)) == GateState::kTimeout);
  assert(!gate.active());

  const CanFrame mode = NeroCanProtocolV111::mode_joint(50, true);
  assert(mode.id == 0x151);
  assert(mode.data[0] == 0x01 && mode.data[1] == 0x01 && mode.data[2] == 50 && mode.data[6] == 0x01);

  Vec7 command = Vec7::Zero();
  command[0] = 1.5707963267948966;
  command[6] = -1.5707963267948966;
  const auto target = NeroCanProtocolV111::joint_target(command);
  assert(target[0].id == 0x155 && target[3].id == 0x170);
  assert(target[0].data[0] == 0x00 && target[0].data[1] == 0x01 && target[0].data[2] == 0x5F && target[0].data[3] == 0x90);
  assert(target[3].data[0] == 0xFF && target[3].data[1] == 0xFE && target[3].data[2] == 0xA0 && target[3].data[3] == 0x70);

  const CanFrame leader_mode = NeroCanProtocolV111::leader_zero_force_mode(50);
  const CanFrame leader_config = NeroCanProtocolV111::leader_zero_force_config();
  assert(leader_mode.id == 0x151 && leader_mode.data[0] == 0x01 && leader_mode.data[1] == 0xFF && leader_mode.data[2] == 50 && leader_mode.data[3] == 0x00 && leader_mode.data[6] == 0x02);
  assert(leader_config.id == 0x470 && leader_config.data[0] == 0xFA);

  const CanFrame mit_mode = NeroCanProtocolV111::mode_mit(50, true);
  assert(mit_mode.id == 0x151 && mit_mode.data[0] == 0x01 && mit_mode.data[1] == 0x06 && mit_mode.data[3] == 0xAD && mit_mode.data[6] == 0x01);
  const NeroMitCommand mit_command{.position_rad = 0.0, .velocity_rad_s = 0.0, .kp_nm_rad = 10.0, .kd_nm_s_rad = 0.8, .feedforward_torque_nm = 0.0};
  const CanFrame mit_frame = NeroCanProtocolV111::joint_mit(1, mit_command);
  assert(mit_frame.id == 0x15A);
  assert(mit_frame.data[0] == 0x7F && mit_frame.data[1] == 0xFF && mit_frame.data[2] == 0x7F);
  // Regression for the native MIT impedance fields: kp=10, kd=0.8, t_ff=0.
  assert(mit_frame.data[3] == 0xF0 && mit_frame.data[4] == 0x51);
  assert(mit_frame.data[5] == 0x94 && mit_frame.data[6] == 0x77 && mit_frame.data[7] == 0xFF);

  // NERO firmware v1.11 uses a 12-bit t_ff range of plus or minus 16 Nm for every joint.
  NeroMitCommand torque_range_test{};
  torque_range_test.feedforward_torque_nm = 8.0;
  const CanFrame j2_torque = NeroCanProtocolV111::joint_mit(2, torque_range_test);
  const CanFrame j4_torque = NeroCanProtocolV111::joint_mit(4, torque_range_test);
  const CanFrame j7_torque = NeroCanProtocolV111::joint_mit(7, torque_range_test);
  assert(j2_torque.data[6] == 0xFB && j2_torque.data[7] == 0xFF);
  assert(j4_torque.data[6] == 0xFB && j4_torque.data[7] == 0xFF);
  assert(j7_torque.data[6] == 0xFB && j7_torque.data[7] == 0xFF);

  const CanFrame gripper_command = NeroCanProtocolV111::gripper_width(0.094666, 1.0, true);
  assert(gripper_command.id == 0x159);
  assert(gripper_command.data[0] == 0x00 && gripper_command.data[1] == 0x01 && gripper_command.data[2] == 0x71 && gripper_command.data[3] == 0xCA);
  assert(gripper_command.data[4] == 0x03 && gripper_command.data[5] == 0xE8 && gripper_command.data[6] == 0x01);
  const CanFrame gripper_zero = NeroCanProtocolV111::gripper_control(0.0, 0.0, 0x00, 0xAE);
  assert(gripper_zero.id == 0x159 && gripper_zero.data[6] == 0x00 && gripper_zero.data[7] == 0xAE);
  const CanFrame gripper_reset = NeroCanProtocolV111::gripper_control(0.0, 0.0, 0x02);
  assert(gripper_reset.id == 0x159 && gripper_reset.data[6] == 0x02 && gripper_reset.data[7] == 0x00);
  const CanFrame gripper_enable_clear = NeroCanProtocolV111::gripper_control(0.0, 1.0, 0x03);
  assert(gripper_enable_clear.id == 0x159 && gripper_enable_clear.data[4] == 0x03 &&
         gripper_enable_clear.data[5] == 0xE8 && gripper_enable_clear.data[6] == 0x03);
  const CanFrame gripper_stroke = NeroCanProtocolV111::gripper_teaching_config(100, 1);
  assert(gripper_stroke.id == 0x47D && gripper_stroke.data[0] == 100 &&
         gripper_stroke.data[1] == 100 && gripper_stroke.data[2] == 1);
  const CanFrame gripper_stroke_query = NeroCanProtocolV111::gripper_teaching_query();
  assert(gripper_stroke_query.id == 0x477 && gripper_stroke_query.data[0] == 0x04);

  CanFrame gripper_frame = gripper_command;
  gripper_frame.id = 0x2A8;
  gripper_frame.data[6] = 0xC0;
  GripperFeedback gripper_feedback;
  assert(NeroCanProtocolV111::decode_gripper_feedback(gripper_frame, &gripper_feedback));
  assert(std::abs(gripper_feedback.width_m - 0.094666) < 1e-9);
  assert(gripper_feedback.status_code == 0xC0 && gripper_feedback.mode == 0x00);
  assert(gripper_feedback.enabled && gripper_feedback.homed && !gripper_feedback.fault);

  NeroFeedback feedback;
  CanFrame feedback_frame = target[0];
  feedback_frame.id = 0x2A5;
  feedback_frame.monotonic_ns = 42;
  assert(NeroCanProtocolV111::decode_joint_feedback(feedback_frame, &feedback));
  assert(std::abs(feedback.joints[0] - command[0]) < 2e-5);
  assert(feedback.joint_valid[0] && feedback.joint_valid[1]);

  const std::string urdf = "/home/lyh/nero/xrobot/third_party/agx_arm_urdf/nero/urdf/nero_description.urdf";
  NeroDynamicsModel model(urdf);
  const Vec7 q = (Vec7() << 0.0, -0.2, 0.3, 0.2, -0.1, 0.1, 0.0).finished();
  const Pose pose = model.forward_kinematics(q);
  const Vec7 ik = model.solve_ik(pose, q);
  assert((ik - q).norm() < 1e-3);
  // A pure TCP attitude command must not be discarded just because its local
  // waypoint is smaller than a previously loose orientation tolerance.
  Pose orientation_target = pose;
  orientation_target.orientation = (Eigen::Quaterniond(Eigen::AngleAxisd(0.05, Vec3::UnitZ())) * pose.orientation).normalized();
  const Vec7 orientation_ik = model.solve_ik(orientation_target, q, 0.010, 0.001);
  const Pose orientation_achieved = model.forward_kinematics(orientation_ik);
  const Eigen::AngleAxisd orientation_residual(
      (orientation_target.orientation * orientation_achieved.orientation.conjugate()).normalized());
  assert((orientation_ik - q).cwiseAbs().maxCoeff() > 1e-4);
  assert(orientation_residual.angle() < 0.010);
  // A redundant 7-DOF arm must retain a usable solution for pure TCP
  // attitude changes about each local Cartesian axis.
  ContinuousIkOptions absolute_ik_options;
  absolute_ik_options.position_tolerance_m = 0.005;
  absolute_ik_options.orientation_tolerance_rad = 0.030;
  absolute_ik_options.continuity_weight = 0.005;
  absolute_ik_options.joint_limit_weight = 0.000005;
  absolute_ik_options.max_iterations = 160;
  const std::array<Vec3, 3> absolute_axes{Vec3::UnitX(), Vec3::UnitY(), Vec3::UnitZ()};
  for (const Vec3& axis : absolute_axes) {
    Pose absolute_target = pose;
    absolute_target.orientation = (Eigen::Quaterniond(Eigen::AngleAxisd(0.05, axis)) * pose.orientation).normalized();
    const ContinuousIkResult absolute_solution = model.solve_continuous_ik(absolute_target, q, absolute_ik_options);
    assert(absolute_solution.position_error_m <= absolute_ik_options.position_tolerance_m);
    assert(absolute_solution.orientation_error_rad <= absolute_ik_options.orientation_tolerance_rad);
    assert((absolute_solution.joints - q).cwiseAbs().maxCoeff() > 1e-4);
    const ContinuousIkResult redundant_solution = model.solve_redundant_continuous_ik(
        absolute_target, q, absolute_ik_options, 0.05);
    assert(redundant_solution.position_error_m <= absolute_ik_options.position_tolerance_m);
    assert(redundant_solution.orientation_error_rad <= absolute_ik_options.orientation_tolerance_rad);
    assert(redundant_solution.minimum_joint_margin_rad > 0.0);
  }
  const Vec7 ik_step = model.solve_ik_step(pose, q, 0.015);
  assert((ik_step - q).norm() < 1e-8);
  const Vec7 continuous_seed = q + (Vec7() << 0.03, -0.02, 0.02, -0.01, 0.01, -0.02, 0.02).finished();
  const ContinuousIkResult continuous_ik = model.solve_continuous_ik(pose, continuous_seed);
  assert(continuous_ik.position_error_m < 1.5e-3);
  // A 1 mm Cartesian command must not be discarded by the continuous solver.
  Pose millimeter_target = pose;
  millimeter_target.position.x() += 0.001;
  ContinuousIkOptions millimeter_options;
  millimeter_options.position_tolerance_m = 0.0002;
  millimeter_options.orientation_tolerance_rad = std::numbers::pi;
  const ContinuousIkResult millimeter_ik = model.solve_continuous_ik(millimeter_target, q, millimeter_options);
  assert(millimeter_ik.position_error_m < 0.0002);
  assert((millimeter_ik.joints - q).cwiseAbs().maxCoeff() > 1e-5);
  assert(continuous_ik.orientation_error_rad < 0.05);
  assert((continuous_ik.joints - model.lower_limits()).minCoeff() > 0.0);
  assert((model.upper_limits() - continuous_ik.joints).minCoeff() > 0.0);
#ifdef XROBOT_CPP_WITH_PLACO
  const std::string control_urdf = "/home/lyh/nero/xrobot_cpp/assets/urdf/nero_control_tcp.urdf";
  NeroDynamicsModel control_model(control_urdf, "base_link", "gripper_tcp");
  PlacoNeroIkSolver placo_ik(control_urdf, "gripper_tcp");
  const Pose control_pose = control_model.forward_kinematics(q);
  Pose placo_translation_target = control_pose;
  placo_translation_target.position.x() += 0.001;
  ContinuousIkOptions placo_options;
  placo_options.position_tolerance_m = 0.002;
  placo_options.orientation_tolerance_rad = 0.05;
  const ContinuousIkResult placo_translation = placo_ik.solve(placo_translation_target, q, placo_options, 0.010, 0.30);
  assert(placo_translation.joints.allFinite());
  assert((placo_translation.joints - control_model.lower_limits()).minCoeff() >= -1e-10);
  assert((control_model.upper_limits() - placo_translation.joints).minCoeff() >= -1e-10);
  assert((placo_translation.joints - q).cwiseAbs().maxCoeff() > 1e-6);
  Pose placo_orientation_target = control_pose;
  placo_orientation_target.orientation = (Eigen::Quaterniond(Eigen::AngleAxisd(0.02, Vec3::UnitZ())) * control_pose.orientation).normalized();
  const ContinuousIkResult placo_orientation = placo_ik.solve(placo_orientation_target, q, placo_options, 0.010, 0.30);
  assert(placo_orientation.joints.allFinite());
  assert((placo_orientation.joints - q).cwiseAbs().maxCoeff() > 1e-6);
  Vec7 near_j2_upper = q;
  near_j2_upper[1] = control_model.upper_limits()[1] - 0.01;
  const Pose near_limit_pose = control_model.forward_kinematics(near_j2_upper);
  Pose boundary_target = near_limit_pose;
  boundary_target.position.x() += 0.001;
  const ContinuousIkResult boundary_solution = placo_ik.solve(boundary_target, near_j2_upper, placo_options, 0.010, 0.30);
  assert(boundary_solution.joints.allFinite());
  assert(boundary_solution.joints[1] <= near_j2_upper[1] + 1e-10);
  assert((boundary_solution.joints - control_model.lower_limits()).minCoeff() >= -1e-10);
  assert((control_model.upper_limits() - boundary_solution.joints).minCoeff() >= -1e-10);
#endif
  NeroFeedback healthy_feedback;
  healthy_feedback.joints = q;
  healthy_feedback.joint_valid.fill(true);
  healthy_feedback.driver_valid.fill(true);
  healthy_feedback.joint_enabled.fill(true);
  healthy_feedback.joint_fault.fill(false);
  const auto safety_now = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
  healthy_feedback.monotonic_ns = safety_now;
  NeroTorqueSafetyLimits torque_limits;
  torque_limits.max_velocity_rad_s = Vec7::Constant(0.5);
  torque_limits.max_torque_nm = Vec7::Constant(1.0);
  torque_limits.max_tracking_error_rad = 0.05;
  NeroTorqueSafetyGate torque_gate(model.lower_limits(), model.upper_limits(), 0.25, torque_limits);
  torque_gate.notify_keepalive(safety_now);
  JointTrajectoryState safe_desired{.position = q, .velocity = Vec7::Zero(), .acceleration = Vec7::Zero()};
  std::string safety_reason;
  assert(torque_gate.permits(safe_desired, Vec7::Zero(), Vec7::Constant(0.5), healthy_feedback, safety_now, &safety_reason));
  NeroCommandInterlock feedback_interlock(model.lower_limits(), model.upper_limits(), 0.25, 0.05);
  feedback_interlock.notify_control_input(safety_now);
  NeroFeedback concurrently_published_feedback = healthy_feedback;
  concurrently_published_feedback.monotonic_ns = safety_now + 1000;
  assert(feedback_interlock.observe(concurrently_published_feedback, safety_now) == NeroSafetyState::kReady);
  NeroFeedback stale_feedback = healthy_feedback;
  stale_feedback.monotonic_ns = safety_now - 300000000;
  assert(feedback_interlock.observe(stale_feedback, safety_now) == NeroSafetyState::kAwaitFeedback);
  torque_gate.notify_keepalive(safety_now);
  assert(!torque_gate.permits(safe_desired, Vec7::Zero(), Vec7::Constant(1.1), healthy_feedback, safety_now, &safety_reason));
  NeroCommandInterlock encoder_tolerance_interlock(model.lower_limits() - Vec7::Constant(0.10), model.upper_limits() + Vec7::Constant(0.10), 0.25, 0.25);
  encoder_tolerance_interlock.notify_control_input(safety_now);
  JointTrajectoryState recovering_desired{.position = q, .velocity = Vec7::Zero(), .acceleration = Vec7::Zero()};
  recovering_desired.position[2] = model.lower_limits()[2] - 0.05;
  NeroFeedback recovering_feedback = healthy_feedback;
  recovering_feedback.joints[2] = recovering_desired.position[2];
  assert(encoder_tolerance_interlock.permits(recovering_desired.position, recovering_feedback, safety_now, &safety_reason));
  recovering_desired.position[2] = model.lower_limits()[2] - 0.11;
  recovering_feedback.joints[2] = recovering_desired.position[2];
  assert(!encoder_tolerance_interlock.permits(recovering_desired.position, recovering_feedback, safety_now, &safety_reason));
  const DynamicsTerms terms = model.dynamics(q, Vec7::Zero());
  assert(terms.mass.allFinite() && terms.mass.isApprox(terms.mass.transpose(), 1e-8));
  assert(terms.gravity.allFinite() && terms.coriolis.norm() < 1e-8);
  JointTrajectoryState desired;
  desired.position = q + Vec7::Constant(0.01);
  desired.velocity = Vec7::Constant(0.1);
  desired.acceleration.setZero();
  JerkLimitedJointPlanner trajectory(Vec7::Constant(0.60), Vec7::Constant(1.50), Vec7::Constant(12.0));
  trajectory.reset(q);
  trajectory.set_goal(q + Vec7::Constant(0.10));
  JointTrajectoryState previous_trajectory = trajectory.state();
  for (int step = 0; step < 2000; ++step) {
    const JointTrajectoryState next_trajectory = trajectory.step(0.001);
    assert((next_trajectory.velocity.array().abs() <= 0.60 + 1e-12).all());
    assert((next_trajectory.acceleration.array().abs() <= 1.50 + 1e-12).all());
    assert(((next_trajectory.acceleration - previous_trajectory.acceleration).array().abs() <= 0.012 + 1e-12).all());
    previous_trajectory = next_trajectory;
  }
  assert((trajectory.state().position - trajectory.goal()).norm() < 2e-3);
  ComputedTorqueController controller(Vec7::Constant(8.0), Vec7::Constant(1.5), Vec7::Constant(4.0));
  const ComputedTorqueCommand torque_command = controller.compute(model, q, Vec7::Zero(), desired);
  assert(torque_command.feedforward_torque_nm.allFinite());
  assert(torque_command.virtual_spring_torque_nm.norm() > 0.0);
  assert(torque_command.virtual_damping_torque_nm.norm() > 0.0);
  assert((torque_command.feedforward_torque_nm.array().abs() <= 4.0 + 1e-12).all());
  const JointDriveModel simulated_drive = JointDriveModel::mujoco_default();
  ComputedTorqueController simulated_controller(
      Vec7::Constant(8.0), Vec7::Constant(1.5), Vec7::Constant(8.0), simulated_drive);
  JointTrajectoryState accelerating_desired = desired;
  accelerating_desired.acceleration = Vec7::Constant(0.5);
  const ComputedTorqueCommand simulated_torque_command = simulated_controller.compute(
      model, q, Vec7::Constant(0.10), accelerating_desired);
  assert(simulated_torque_command.reflected_inertia_torque_nm.norm() > 0.0);
  assert(simulated_torque_command.viscous_friction_torque_nm.norm() > 0.0);
  assert(simulated_torque_command.coulomb_friction_torque_nm.norm() > 0.0);

#ifdef XROBOT_CPP_WITH_MUJOCO
  NeroMujocoSimulation simulation("/home/lyh/nero/xrobot_cpp/assets/mujoco/nero_torque.xml");
  simulation.reset(q);
  assert((simulation.joint_position() - q).norm() < 1e-12);
  simulation.set_gripper_width(0.060);
  simulation.set_joint_torque(Vec7::Zero());
  simulation.step(10);
  assert(simulation.joint_position().allFinite());
  assert(simulation.joint_velocity().allFinite());
  assert(simulation.gripper_width_m() >= 0.0 && simulation.gripper_width_m() <= 0.100 + 1e-9);
#endif

  std::cout << "xrobot_cpp core tests passed\n";
  return 0;
}
