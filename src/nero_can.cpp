#include "xrobot_cpp/nero_can.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <stdexcept>

#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace xrobot_cpp {
namespace {
constexpr double kRadiansPerMillidegree = M_PI / 180000.0;

std::uint64_t monotonic_ns() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      Clock::now().time_since_epoch()).count());
}

void put_i32_be(std::array<std::uint8_t, 8>* data, std::size_t offset, std::int32_t value) {
  const auto raw = static_cast<std::uint32_t>(value);
  (*data)[offset] = static_cast<std::uint8_t>((raw >> 24U) & 0xFFU);
  (*data)[offset + 1] = static_cast<std::uint8_t>((raw >> 16U) & 0xFFU);
  (*data)[offset + 2] = static_cast<std::uint8_t>((raw >> 8U) & 0xFFU);
  (*data)[offset + 3] = static_cast<std::uint8_t>(raw & 0xFFU);
}

std::int32_t get_i32_be(const std::array<std::uint8_t, 8>& data, std::size_t offset) {
  const auto raw = (static_cast<std::uint32_t>(data[offset]) << 24U) |
                   (static_cast<std::uint32_t>(data[offset + 1]) << 16U) |
                   (static_cast<std::uint32_t>(data[offset + 2]) << 8U) |
                   static_cast<std::uint32_t>(data[offset + 3]);
  return static_cast<std::int32_t>(raw);
}

std::int32_t radians_to_millidegrees(double radians) {
  if (!std::isfinite(radians)) {
    throw std::invalid_argument("NERO joint target must be finite");
  }
  const double value = std::round(radians / kRadiansPerMillidegree);
  if (value < static_cast<double>(INT32_MIN) || value > static_cast<double>(INT32_MAX)) {
    throw std::out_of_range("NERO joint target exceeds protocol int32 range");
  }
  return static_cast<std::int32_t>(value);
}

CanFrame joint_pair(std::uint32_t id, std::int32_t first, std::int32_t second) {
  CanFrame frame;
  frame.id = id;
  put_i32_be(&frame.data, 0, first);
  put_i32_be(&frame.data, 4, second);
  return frame;
}

std::uint16_t float_to_uint(double value, double minimum, double maximum, unsigned bits) {
  if (!std::isfinite(value) || value < minimum || value > maximum) {
    throw std::out_of_range("MIT value exceeds v1.11 protocol range");
  }
  const auto max_raw = static_cast<std::uint32_t>((1U << bits) - 1U);
  const double encoded = (value - minimum) * static_cast<double>(max_raw) / (maximum - minimum);
  return static_cast<std::uint16_t>(std::clamp(encoded, 0.0, static_cast<double>(max_raw)));
}
}  // namespace

CanFrame NeroCanProtocolV111::mode_joint(std::uint8_t speed_percent, bool enable_feedback) {
  CanFrame frame;
  frame.id = kModeControl;
  frame.data[0] = 0x01;  // CAN control
  frame.data[1] = 0x01;  // MOVE J
  frame.data[2] = speed_percent;
  frame.data[3] = 0x00;  // position/velocity mode
  frame.data[6] = enable_feedback ? 0x01 : 0x00;
  return frame;
}

CanFrame NeroCanProtocolV111::mode_mit(std::uint8_t speed_percent, bool enable_feedback) {
  CanFrame frame;
  frame.id = kModeControl;
  frame.data[0] = 0x01;
  frame.data[1] = 0x06;
  frame.data[2] = speed_percent;
  frame.data[3] = 0xAD;
  frame.data[6] = enable_feedback ? 0x01 : 0x00;
  return frame;
}

CanFrame NeroCanProtocolV111::normal_single_arm_config() {
  CanFrame frame;
  frame.id = kLeaderFollowerConfig;
  return frame;
}

CanFrame NeroCanProtocolV111::leader_zero_force_config() {
  CanFrame frame;
  frame.id = kLeaderFollowerConfig;
  frame.data[0] = 0xFA;  // official SDK leader zero-force drag configuration
  return frame;
}

CanFrame NeroCanProtocolV111::normal_mode_feedback(std::uint8_t speed_percent) {
  CanFrame frame;
  frame.id = kModeControl;
  frame.data[0] = 0x01;  // CAN control
  frame.data[1] = 0xFF;  // retain current movement mode
  frame.data[2] = speed_percent;
  frame.data[6] = 0x01;  // enable CAN push feedback
  return frame;
}

CanFrame NeroCanProtocolV111::leader_zero_force_mode(std::uint8_t speed_percent) {
  CanFrame frame;
  frame.id = kModeControl;
  frame.data[0] = 0x01;  // CAN control
  frame.data[1] = 0xFF;  // retain current movement mode
  frame.data[2] = speed_percent;
  frame.data[3] = 0x00;
  frame.data[6] = 0x02;  // official SDK: leader CAN push
  return frame;
}

std::array<CanFrame, 4> NeroCanProtocolV111::joint_target(const Vec7& radians) {
  return {joint_pair(kJoint12, radians_to_millidegrees(radians[0]), radians_to_millidegrees(radians[1])),
          joint_pair(kJoint34, radians_to_millidegrees(radians[2]), radians_to_millidegrees(radians[3])),
          joint_pair(kJoint56, radians_to_millidegrees(radians[4]), radians_to_millidegrees(radians[5])),
          joint_pair(kJoint7, radians_to_millidegrees(radians[6]), 0)};
}

CanFrame NeroCanProtocolV111::gripper_width(double width_m, double force_n, bool enable) {
  return gripper_control(width_m, force_n, enable ? 0x01U : 0x00U);
}

CanFrame NeroCanProtocolV111::gripper_control(double width_m, double force_n,
                                                std::uint8_t status_code, std::uint8_t set_zero) {
  if (!std::isfinite(width_m) || !std::isfinite(force_n) || width_m < 0.0 || force_n < 0.0) {
    throw std::invalid_argument("gripper width and force must be finite and non-negative");
  }
  if (status_code > 0x07U || (set_zero != 0x00U && set_zero != 0xAEU)) {
    throw std::invalid_argument("invalid gripper v1.11 status or zero command");
  }
  const double width_um = std::round(width_m * 1e6);
  const double force_mn = std::round(force_n * 1e3);
  if (width_um > static_cast<double>(INT32_MAX) || force_mn > 65535.0) {
    throw std::out_of_range("gripper target exceeds v1.11 protocol range");
  }
  CanFrame frame;
  frame.id = kGripperControl;
  put_i32_be(&frame.data, 0, static_cast<std::int32_t>(width_um));
  frame.data[4] = static_cast<std::uint8_t>((static_cast<std::uint16_t>(force_mn) >> 8U) & 0xFFU);
  frame.data[5] = static_cast<std::uint8_t>(static_cast<std::uint16_t>(force_mn) & 0xFFU);
  frame.data[6] = status_code;
  frame.data[7] = set_zero;
  return frame;
}

CanFrame NeroCanProtocolV111::gripper_teaching_config(std::uint8_t maximum_stroke_mm,
                                                            std::uint8_t teaching_friction) {
  if ((maximum_stroke_mm != 70U && maximum_stroke_mm != 100U) ||
      teaching_friction < 1U || teaching_friction > 10U) {
    throw std::invalid_argument("gripper teaching configuration requires a 70 or 100 mm stroke and friction in [1, 10]");
  }
  CanFrame frame;
  frame.id = 0x47DU;
  frame.data[0] = 100U;
  frame.data[1] = maximum_stroke_mm;
  frame.data[2] = teaching_friction;
  return frame;
}

CanFrame NeroCanProtocolV111::gripper_teaching_query() {
  CanFrame frame;
  frame.id = 0x477U;
  frame.data[0] = 0x04U;
  return frame;
}

CanFrame NeroCanProtocolV111::joint_mit(std::uint8_t joint_index, const NeroMitCommand& command) {
  if (joint_index < 1 || joint_index > 7) throw std::invalid_argument("NERO MIT joint index must be in [1, 7]");
  const std::uint16_t position = float_to_uint(command.position_rad, -12.5, 12.5, 16);
  const double protocol_velocity = joint_index == 6 ? command.velocity_rad_s : -command.velocity_rad_s;
  const std::uint16_t velocity = float_to_uint(protocol_velocity, -45.0, 45.0, 12);
  const std::uint16_t kp = float_to_uint(command.kp_nm_rad, 0.0, 500.0, 12);
  const std::uint16_t kd = float_to_uint(command.kd_nm_s_rad, -5.0, 5.0, 12);
  // NERO firmware v1.11 packs t_ff as a 12-bit field with one +/-16 Nm range
  // for every joint. Older firmware used per-joint ranges, which must not be
  // applied to a v1.11 arm.
  constexpr double torque_limit_nm = 16.0;
  const std::uint16_t torque = float_to_uint(
      command.feedforward_torque_nm, -torque_limit_nm, torque_limit_nm, 12);
  CanFrame frame;
  frame.id = kMitJoint1 + static_cast<std::uint32_t>(joint_index - 1);
  frame.data[0] = static_cast<std::uint8_t>(position >> 8U);
  frame.data[1] = static_cast<std::uint8_t>(position & 0xFFU);
  frame.data[2] = static_cast<std::uint8_t>(velocity >> 4U);
  frame.data[3] = static_cast<std::uint8_t>(((velocity & 0x0FU) << 4U) | (kp >> 8U));
  frame.data[4] = static_cast<std::uint8_t>(kp & 0xFFU);
  frame.data[5] = static_cast<std::uint8_t>(kd >> 4U);
  frame.data[6] = static_cast<std::uint8_t>(((kd & 0x0FU) << 4U) | (torque >> 8U));
  frame.data[7] = static_cast<std::uint8_t>(torque & 0xFFU);
  return frame;
}

CanFrame NeroCanProtocolV111::motor_enable(std::uint8_t joint_index, bool enabled) {
  if (joint_index < 1 || joint_index > 8) {
    throw std::invalid_argument("NERO motor index must be in [1, 8]");
  }
  CanFrame frame;
  frame.id = kMotorEnable;
  frame.data[0] = joint_index;
  frame.data[1] = enabled ? 0x02 : 0x01;
  return frame;
}

bool NeroCanProtocolV111::decode_joint_feedback(const CanFrame& frame, NeroFeedback* feedback) {
  if (feedback == nullptr || frame.size != 8) {
    return false;
  }
  std::size_t index{};
  bool paired = true;
  switch (frame.id) {
    case kFeedbackJoint12: index = 0; break;
    case kFeedbackJoint34: index = 2; break;
    case kFeedbackJoint56: index = 4; break;
    case kFeedbackJoint7: index = 6; paired = false; break;
    default: return false;
  }
  feedback->joints[static_cast<Eigen::Index>(index)] =
      static_cast<double>(get_i32_be(frame.data, 0)) * kRadiansPerMillidegree;
  feedback->joint_valid[index] = true;
  if (paired) {
    feedback->joints[static_cast<Eigen::Index>(index + 1)] =
        static_cast<double>(get_i32_be(frame.data, 4)) * kRadiansPerMillidegree;
    feedback->joint_valid[index + 1] = true;
  }
  feedback->monotonic_ns = frame.monotonic_ns;
  const std::uint8_t cycle_bit = static_cast<std::uint8_t>(1U << (index / 2));
  feedback->joint_state_cycle_mask = static_cast<std::uint8_t>(feedback->joint_state_cycle_mask | cycle_bit);
  if (feedback->joint_state_cycle_mask == 0x0FU) {
    feedback->joint_state_cycle_mask = 0;
    feedback->joint_state_monotonic_ns = frame.monotonic_ns;
    ++feedback->joint_state_sequence;
  }
  return true;
}

bool NeroCanProtocolV111::decode_driver_feedback(const CanFrame& frame, NeroFeedback* feedback) {
  if (feedback == nullptr || frame.size != 8 || frame.id < 0x261 || frame.id > 0x267) return false;
  const std::size_t index = static_cast<std::size_t>(frame.id - 0x261);
  const std::uint8_t status = frame.data[5];
  feedback->driver_valid[index] = true;
  feedback->joint_enabled[index] = (status & 0x40U) != 0;
  feedback->joint_fault[index] = (status & 0xBFU) != 0;
  feedback->monotonic_ns = frame.monotonic_ns;
  return true;
}

bool NeroCanProtocolV111::decode_gripper_feedback(const CanFrame& frame, GripperFeedback* feedback) {
  if (feedback == nullptr || frame.id != kFeedbackGripper || frame.size != 8) return false;
  feedback->width_m = static_cast<double>(get_i32_be(frame.data, 0)) * 1e-6;
  const auto force_mn = static_cast<std::uint16_t>((static_cast<std::uint16_t>(frame.data[4]) << 8U) | frame.data[5]);
  feedback->force_n = static_cast<double>(force_mn) * 1e-3;
  const std::uint8_t status = frame.data[6];
  feedback->status_code = status;
  feedback->mode = frame.data[7];
  feedback->enabled = (status & 0x40U) != 0;
  feedback->homed = (status & 0x80U) != 0;
  feedback->fault = (status & 0x3FU) != 0;
  feedback->monotonic_ns = frame.monotonic_ns;
  return true;
}

NeroSocketCan::NeroSocketCan(std::string interface_name) : interface_name_(std::move(interface_name)) {
  const char* audit_path = std::getenv("XROBOT_CPP_TX_AUDIT_PATH");
  if (audit_path != nullptr && audit_path[0] != '\0') {
    tx_audit_.open(audit_path, std::ios::app);
    if (!tx_audit_) throw std::runtime_error("cannot open XROBOT_CPP_TX_AUDIT_PATH");
  }
}

NeroSocketCan::~NeroSocketCan() { close(); }

void NeroSocketCan::open_read_only() {
  if (socket_ >= 0) {
    return;
  }
  socket_ = ::socket(PF_CAN, SOCK_RAW | SOCK_CLOEXEC, CAN_RAW);
  if (socket_ < 0) {
    throw std::runtime_error("SocketCAN socket creation failed: " + std::string(std::strerror(errno)));
  }
  ifreq request{};
  std::strncpy(request.ifr_name, interface_name_.c_str(), IFNAMSIZ - 1);
  if (::ioctl(socket_, SIOCGIFINDEX, &request) < 0) {
    const std::string error = std::strerror(errno);
    close();
    throw std::runtime_error("SocketCAN interface lookup failed: " + error);
  }
  sockaddr_can address{};
  address.can_family = AF_CAN;
  address.can_ifindex = request.ifr_ifindex;
  if (::bind(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
    const std::string error = std::strerror(errno);
    close();
    throw std::runtime_error("SocketCAN bind failed: " + error);
  }
}

void NeroSocketCan::close() {
  if (socket_ >= 0) {
    ::close(socket_);
    socket_ = -1;
  }
}

bool NeroSocketCan::is_open() const { return socket_ >= 0; }

std::optional<CanFrame> NeroSocketCan::read(std::chrono::milliseconds timeout) {
  if (socket_ < 0) {
    throw std::logic_error("SocketCAN read requested before open");
  }
  pollfd descriptor{.fd = socket_, .events = POLLIN, .revents = 0};
  const int ready = ::poll(&descriptor, 1, static_cast<int>(timeout.count()));
  if (ready == 0) return std::nullopt;
  if (ready < 0) throw std::runtime_error("SocketCAN poll failed: " + std::string(std::strerror(errno)));
  can_frame native{};
  if (::read(socket_, &native, sizeof(native)) != static_cast<ssize_t>(sizeof(native))) {
    throw std::runtime_error("SocketCAN read failed: " + std::string(std::strerror(errno)));
  }
  CanFrame frame;
  frame.id = native.can_id & CAN_SFF_MASK;
  frame.size = native.can_dlc;
  std::copy_n(native.data, std::min<std::size_t>(frame.size, frame.data.size()), frame.data.begin());
  frame.monotonic_ns = monotonic_ns();
  std::lock_guard lock(mutex_);
  ++receive_diagnostics_.received_frames;
  receive_diagnostics_.last_received_id = frame.id;
  receive_diagnostics_.last_received_monotonic_ns = frame.monotonic_ns;
  const bool joint_feedback = NeroCanProtocolV111::decode_joint_feedback(frame, &feedback_);
  if (joint_feedback) {
    ++receive_diagnostics_.joint_feedback_frames;
    receive_diagnostics_.last_joint_feedback_id = frame.id;
    receive_diagnostics_.last_joint_feedback_monotonic_ns = frame.monotonic_ns;
  }
  NeroCanProtocolV111::decode_driver_feedback(frame, &feedback_);
  NeroCanProtocolV111::decode_gripper_feedback(frame, &gripper_feedback_);
  return frame;
}

GripperFeedback NeroSocketCan::gripper_feedback() const {
  std::lock_guard lock(mutex_);
  return gripper_feedback_;
}

CanReceiveDiagnostics NeroSocketCan::receive_diagnostics() const {
  std::lock_guard lock(mutex_);
  return receive_diagnostics_;
}

NeroFeedback NeroSocketCan::feedback() const {
  std::lock_guard lock(mutex_);
  return feedback_;
}

void NeroSocketCan::send(const CanFrame& frame) {
  if (socket_ < 0) throw std::logic_error("SocketCAN send requested before open");
  can_frame native{};
  native.can_id = frame.id;
  native.can_dlc = frame.size;
  std::copy_n(frame.data.begin(), frame.size, native.data);
  if (::write(socket_, &native, sizeof(native)) != static_cast<ssize_t>(sizeof(native))) {
    throw std::runtime_error("SocketCAN write failed: " + std::string(std::strerror(errno)));
  }
  audit_transmit(frame);
}

void NeroSocketCan::audit_transmit(const CanFrame& frame) {
  if (!tx_audit_) return;
  tx_audit_ << "{\"event\":\"can_tx\",\"monotonic_ns\":" << monotonic_ns()
            << ",\"interface\":\"" << interface_name_ << "\",\"arbitration_id\":" << frame.id
            << ",\"dlc\":" << static_cast<unsigned>(frame.size) << ",\"data_hex\":\"";
  for (std::size_t index = 0; index < frame.size; ++index) {
    tx_audit_ << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<unsigned>(frame.data[index]);
  }
  tx_audit_ << std::dec << "\"}\n";

}

void NeroSocketCan::send_mode_joint(std::uint8_t speed_percent, bool enable_feedback) {
  send(NeroCanProtocolV111::mode_joint(speed_percent, enable_feedback));
}

void NeroSocketCan::send_mode_mit(std::uint8_t speed_percent, bool enable_feedback) {
  send(NeroCanProtocolV111::mode_mit(speed_percent, enable_feedback));
}

void NeroSocketCan::send_normal_single_arm_config() {
  send(NeroCanProtocolV111::normal_single_arm_config());
}

void NeroSocketCan::send_leader_zero_force_mode(std::uint8_t speed_percent) {
  send(NeroCanProtocolV111::leader_zero_force_mode(speed_percent));
}

void NeroSocketCan::send_leader_zero_force_config() {
  send(NeroCanProtocolV111::leader_zero_force_config());
}

void NeroSocketCan::send_normal_mode_feedback(std::uint8_t speed_percent) {
  send(NeroCanProtocolV111::normal_mode_feedback(speed_percent));
}

void NeroSocketCan::send_joint_target(const Vec7& radians) {
  for (const CanFrame& frame : NeroCanProtocolV111::joint_target(radians)) send(frame);
}

void NeroSocketCan::send_joint_mit(std::uint8_t joint_index, const NeroMitCommand& command) {
  send(NeroCanProtocolV111::joint_mit(joint_index, command));
}

void NeroSocketCan::send_gripper_width(double width_m, double force_n, bool enable) {
  send(NeroCanProtocolV111::gripper_width(width_m, force_n, enable));
}

void NeroSocketCan::send_gripper_control(double width_m, double force_n,
                                         std::uint8_t status_code, std::uint8_t set_zero) {
  send(NeroCanProtocolV111::gripper_control(width_m, force_n, status_code, set_zero));
}

void NeroSocketCan::send_gripper_teaching_config(std::uint8_t maximum_stroke_mm,
                                                         std::uint8_t teaching_friction) {
  send(NeroCanProtocolV111::gripper_teaching_config(maximum_stroke_mm, teaching_friction));
}

void NeroSocketCan::send_gripper_teaching_query() {
  send(NeroCanProtocolV111::gripper_teaching_query());
}

void NeroSocketCan::send_motor_enable(std::uint8_t joint_index, bool enabled) {
  send(NeroCanProtocolV111::motor_enable(joint_index, enabled));
}

}  // namespace xrobot_cpp
