#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>

#include "xrobot_cpp/control_core.hpp"

namespace xrobot_cpp {

struct CanFrame {
  std::uint32_t id{};
  std::array<std::uint8_t, 8> data{};
  std::uint8_t size{8};
  std::uint64_t monotonic_ns{};
};

struct GripperFeedback {
  double width_m{};
  double force_n{};
  std::uint8_t status_code{};
  // Firmware 0x2A8 byte 7: 0x00=width, 0x01=angle.
  std::uint8_t mode{};
  bool enabled{};
  bool homed{};
  bool fault{};
  std::uint64_t monotonic_ns{};
};

struct CanReceiveDiagnostics {
  std::uint64_t received_frames{};
  std::uint64_t joint_feedback_frames{};
  std::uint32_t last_received_id{};
  std::uint32_t last_joint_feedback_id{};
  std::uint64_t last_received_monotonic_ns{};
  std::uint64_t last_joint_feedback_monotonic_ns{};
};

struct NeroFeedback {
  Vec7 joints{Vec7::Zero()};
  std::array<bool, 7> joint_valid{};
  std::array<bool, 7> driver_valid{};
  std::array<bool, 7> joint_enabled{};
  std::array<bool, 7> joint_fault{};
  std::uint64_t monotonic_ns{};
  std::uint64_t joint_state_monotonic_ns{};
  std::uint64_t joint_state_sequence{};
  std::uint8_t joint_state_cycle_mask{};
};

struct NeroMitCommand {
  double position_rad{};
  double velocity_rad_s{};
  double kp_nm_rad{};
  double kd_nm_s_rad{};
  double feedforward_torque_nm{};
};

// Byte-exact NERO firmware v1.11 protocol subset, derived from official pyAgxArm.
class NeroCanProtocolV111 {
 public:
  static constexpr std::uint32_t kLeaderFollowerConfig = 0x470;
  static constexpr std::uint32_t kModeControl = 0x151;
  static constexpr std::uint32_t kJoint12 = 0x155;
  static constexpr std::uint32_t kJoint34 = 0x156;
  static constexpr std::uint32_t kJoint56 = 0x157;
  static constexpr std::uint32_t kJoint7 = 0x170;
  static constexpr std::uint32_t kMitJoint1 = 0x15A;
  static constexpr std::uint32_t kMitJoint7 = 0x160;
  static constexpr std::uint32_t kMotorEnable = 0x471;
  static constexpr std::uint32_t kGripperControl = 0x159;
  static constexpr std::uint32_t kFeedbackJoint12 = 0x2A5;
  static constexpr std::uint32_t kFeedbackJoint34 = 0x2A6;
  static constexpr std::uint32_t kFeedbackJoint56 = 0x2A7;
  static constexpr std::uint32_t kFeedbackJoint7 = 0x2A9;
  static constexpr std::uint32_t kFeedbackGripper = 0x2A8;

  static CanFrame mode_joint(std::uint8_t speed_percent, bool enable_feedback);
  static CanFrame mode_mit(std::uint8_t speed_percent, bool enable_feedback);
  static CanFrame normal_mode_feedback(std::uint8_t speed_percent = 50);
  static CanFrame normal_single_arm_config();
  static CanFrame leader_zero_force_mode(std::uint8_t speed_percent = 50);
  static CanFrame leader_zero_force_config();
  static std::array<CanFrame, 4> joint_target(const Vec7& radians);
  static CanFrame motor_enable(std::uint8_t joint_index, bool enabled);
  static CanFrame gripper_width(double width_m, double force_n, bool enable);
  static CanFrame gripper_control(double width_m, double force_n, std::uint8_t status_code,
                                  std::uint8_t set_zero = 0x00);
  static CanFrame gripper_teaching_config(std::uint8_t maximum_stroke_mm,
                                          std::uint8_t teaching_friction = 1);
  static CanFrame gripper_teaching_query();
  static CanFrame joint_mit(std::uint8_t joint_index, const NeroMitCommand& command);
  static bool decode_joint_feedback(const CanFrame& frame, NeroFeedback* feedback);
  static bool decode_driver_feedback(const CanFrame& frame, NeroFeedback* feedback);
  static bool decode_gripper_feedback(const CanFrame& frame, GripperFeedback* feedback);
};

class NeroSocketCan {
 public:
  explicit NeroSocketCan(std::string interface_name = "can0");
  ~NeroSocketCan();
  NeroSocketCan(const NeroSocketCan&) = delete;
  NeroSocketCan& operator=(const NeroSocketCan&) = delete;

  void open_read_only();
  void close();
  [[nodiscard]] bool is_open() const;
  [[nodiscard]] std::optional<CanFrame> read(std::chrono::milliseconds timeout);
  [[nodiscard]] NeroFeedback feedback() const;
  [[nodiscard]] GripperFeedback gripper_feedback() const;
  [[nodiscard]] CanReceiveDiagnostics receive_diagnostics() const;

  // These require an explicit caller-side safety policy. They are never invoked by default CLI modes.
  void send_mode_joint(std::uint8_t speed_percent, bool enable_feedback = true);
  void send_mode_mit(std::uint8_t speed_percent, bool enable_feedback = true);
  void send_normal_mode_feedback(std::uint8_t speed_percent = 50);
  void send_normal_single_arm_config();
  void send_leader_zero_force_mode(std::uint8_t speed_percent = 50);
  void send_leader_zero_force_config();
  void send_joint_target(const Vec7& radians);
  void send_joint_mit(std::uint8_t joint_index, const NeroMitCommand& command);
  void send_motor_enable(std::uint8_t joint_index, bool enabled);
  void send_gripper_width(double width_m, double force_n, bool enable = true);
  void send_gripper_control(double width_m, double force_n, std::uint8_t status_code,
                            std::uint8_t set_zero = 0x00);
  void send_gripper_teaching_config(std::uint8_t maximum_stroke_mm,
                                    std::uint8_t teaching_friction = 1);
  void send_gripper_teaching_query();

 private:
  void send(const CanFrame& frame);
  void audit_transmit(const CanFrame& frame);
  int socket_{-1};
  std::string interface_name_;
  std::ofstream tx_audit_;
  mutable std::mutex mutex_;
  NeroFeedback feedback_;
  GripperFeedback gripper_feedback_;
  CanReceiveDiagnostics receive_diagnostics_;
};

}  // namespace xrobot_cpp
