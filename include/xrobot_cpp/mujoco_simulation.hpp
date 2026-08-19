#pragma once

#include <memory>
#include <string>

#include "xrobot_cpp/control_core.hpp"
#include "xrobot_cpp/dynamics_controller.hpp"

struct mjData_;
typedef struct mjData_ mjData;
struct mjModel_;
typedef struct mjModel_ mjModel;

namespace xrobot_cpp {

// A hardware-independent NERO plant. It owns an MJCF model with seven torque
// motors and two position-controlled gripper fingers. It never opens SocketCAN.
class NeroMujocoSimulation {
 public:
  explicit NeroMujocoSimulation(const std::string& model_path);
  ~NeroMujocoSimulation();
  NeroMujocoSimulation(const NeroMujocoSimulation&) = delete;
  NeroMujocoSimulation& operator=(const NeroMujocoSimulation&) = delete;
  NeroMujocoSimulation(NeroMujocoSimulation&&) noexcept;
  NeroMujocoSimulation& operator=(NeroMujocoSimulation&&) noexcept;

  [[nodiscard]] Vec7 joint_position() const;
  [[nodiscard]] Vec7 joint_velocity() const;
  [[nodiscard]] Vec7 joint_acceleration() const;
  [[nodiscard]] double gripper_width_m() const;
  [[nodiscard]] double time_s() const;
  [[nodiscard]] const mjModel* model() const;
  [[nodiscard]] mjData* data();
  void set_payload_mass_kg(double mass_kg);
  [[nodiscard]] DynamicsTerms dynamics_terms() const;
  [[nodiscard]] Pose body_pose(const std::string& body_name) const;

  void reset(const Vec7& joint_position);
  void set_joint_torque(const Vec7& torque_nm);
  void set_joint_disturbance(const Vec7& torque_nm);
  void set_gripper_width(double width_m);
  void step();
  void step(std::size_t count);

 private:
  mjModel* model_{nullptr};
  mjData* data_{nullptr};
  mjData* gravity_data_{nullptr};
  int joint_qpos_address_[7]{};
  int joint_dof_address_[7]{};
  int torque_actuator_[7]{};
  int gripper_qpos_address_[2]{};
  int gripper_actuator_[2]{};
  int payload_body_{-1};
};

}  // namespace xrobot_cpp
