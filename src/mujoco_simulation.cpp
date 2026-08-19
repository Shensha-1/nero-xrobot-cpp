#include "xrobot_cpp/mujoco_simulation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>

namespace xrobot_cpp {
namespace {

int require_id(const mjModel* model, mjtObj type, const char* name) {
  const int id = mj_name2id(model, type, name);
  if (id < 0) throw std::runtime_error(std::string("MuJoCo model is missing ") + name);
  return id;
}

}  // namespace

NeroMujocoSimulation::NeroMujocoSimulation(const std::string& model_path) {
  std::array<char, 1024> error{};
  model_ = mj_loadXML(model_path.c_str(), nullptr, error.data(), static_cast<int>(error.size()));
  if (model_ == nullptr) throw std::runtime_error("cannot load MuJoCo NERO model: " + std::string(error.data()));
  data_ = mj_makeData(model_);
  gravity_data_ = mj_makeData(model_);
  if (data_ == nullptr || gravity_data_ == nullptr) {
    if (data_ != nullptr) mj_deleteData(data_);
    if (gravity_data_ != nullptr) mj_deleteData(gravity_data_);
    mj_deleteModel(model_);
    data_ = nullptr;
    gravity_data_ = nullptr;
    model_ = nullptr;
    throw std::runtime_error("cannot allocate MuJoCo simulation data");
  }

  for (int index = 0; index < 7; ++index) {
    const std::string joint = "joint" + std::to_string(index + 1);
    const std::string actuator = joint + "_torque";
    const int joint_id = require_id(model_, mjOBJ_JOINT, joint.c_str());
    joint_qpos_address_[index] = model_->jnt_qposadr[joint_id];
    joint_dof_address_[index] = model_->jnt_dofadr[joint_id];
    torque_actuator_[index] = require_id(model_, mjOBJ_ACTUATOR, actuator.c_str());
  }
  constexpr std::array<const char*, 2> kGripperJoints{"gripper_left", "gripper_right"};
  constexpr std::array<const char*, 2> kGripperActuators{"gripper_left_position", "gripper_right_position"};
  for (int index = 0; index < 2; ++index) {
    const int joint_id = require_id(model_, mjOBJ_JOINT, kGripperJoints[index]);
    gripper_qpos_address_[index] = model_->jnt_qposadr[joint_id];
    gripper_actuator_[index] = require_id(model_, mjOBJ_ACTUATOR, kGripperActuators[index]);
  }
  payload_body_ = require_id(model_, mjOBJ_BODY, "payload");
  mj_forward(model_, data_);
}

NeroMujocoSimulation::~NeroMujocoSimulation() {
  if (data_ != nullptr) mj_deleteData(data_);
  if (gravity_data_ != nullptr) mj_deleteData(gravity_data_);
  if (model_ != nullptr) mj_deleteModel(model_);
}

NeroMujocoSimulation::NeroMujocoSimulation(NeroMujocoSimulation&& other) noexcept
    : model_(std::exchange(other.model_, nullptr)), data_(std::exchange(other.data_, nullptr)), gravity_data_(std::exchange(other.gravity_data_, nullptr)) {
  std::copy(std::begin(other.joint_qpos_address_), std::end(other.joint_qpos_address_), std::begin(joint_qpos_address_));
  std::copy(std::begin(other.joint_dof_address_), std::end(other.joint_dof_address_), std::begin(joint_dof_address_));
  std::copy(std::begin(other.torque_actuator_), std::end(other.torque_actuator_), std::begin(torque_actuator_));
  std::copy(std::begin(other.gripper_qpos_address_), std::end(other.gripper_qpos_address_), std::begin(gripper_qpos_address_));
  std::copy(std::begin(other.gripper_actuator_), std::end(other.gripper_actuator_), std::begin(gripper_actuator_));
}

NeroMujocoSimulation& NeroMujocoSimulation::operator=(NeroMujocoSimulation&& other) noexcept {
  if (this == &other) return *this;
  if (data_ != nullptr) mj_deleteData(data_);
  if (gravity_data_ != nullptr) mj_deleteData(gravity_data_);
  if (model_ != nullptr) mj_deleteModel(model_);
  model_ = std::exchange(other.model_, nullptr);
  data_ = std::exchange(other.data_, nullptr);
  gravity_data_ = std::exchange(other.gravity_data_, nullptr);
  std::copy(std::begin(other.joint_qpos_address_), std::end(other.joint_qpos_address_), std::begin(joint_qpos_address_));
  std::copy(std::begin(other.joint_dof_address_), std::end(other.joint_dof_address_), std::begin(joint_dof_address_));
  std::copy(std::begin(other.torque_actuator_), std::end(other.torque_actuator_), std::begin(torque_actuator_));
  std::copy(std::begin(other.gripper_qpos_address_), std::end(other.gripper_qpos_address_), std::begin(gripper_qpos_address_));
  std::copy(std::begin(other.gripper_actuator_), std::end(other.gripper_actuator_), std::begin(gripper_actuator_));
  return *this;
}

Vec7 NeroMujocoSimulation::joint_position() const {
  Vec7 result;
  for (int index = 0; index < 7; ++index) result[index] = data_->qpos[joint_qpos_address_[index]];
  return result;
}

Vec7 NeroMujocoSimulation::joint_velocity() const {
  Vec7 result;
  for (int index = 0; index < 7; ++index) result[index] = data_->qvel[joint_dof_address_[index]];
  return result;
}

Vec7 NeroMujocoSimulation::joint_acceleration() const {
  Vec7 result;
  for (int index = 0; index < 7; ++index) result[index] = data_->qacc[joint_dof_address_[index]];
  return result;
}

double NeroMujocoSimulation::gripper_width_m() const {
  return data_->qpos[gripper_qpos_address_[0]] + data_->qpos[gripper_qpos_address_[1]];
}

double NeroMujocoSimulation::time_s() const { return data_->time; }

const mjModel* NeroMujocoSimulation::model() const { return model_; }

mjData* NeroMujocoSimulation::data() { return data_; }

void NeroMujocoSimulation::set_payload_mass_kg(double mass_kg) {
  if (!std::isfinite(mass_kg) || mass_kg < 0.0 || mass_kg > 3.0) throw std::invalid_argument("MuJoCo payload mass must be in [0, 3] kg");
  const double effective_mass = std::max(mass_kg, 1e-6);
  model_->body_mass[payload_body_] = effective_mass;
  constexpr double side_m = 0.040;
  const double diagonal_inertia = effective_mass * side_m * side_m / 6.0;
  for (int axis = 0; axis < 3; ++axis) model_->body_inertia[3 * payload_body_ + axis] = diagonal_inertia;
  // mj_setConst recomputes model constants and may reset mjData. Preserve the live state
  // so a payload update never changes the simulated arm pose.
  std::vector<mjtNum> qpos(data_->qpos, data_->qpos + model_->nq);
  std::vector<mjtNum> qvel(data_->qvel, data_->qvel + model_->nv);
  std::vector<mjtNum> control(data_->ctrl, data_->ctrl + model_->nu);
  mj_setConst(model_, data_);
  mju_copy(data_->qpos, qpos.data(), model_->nq);
  mju_copy(data_->qvel, qvel.data(), model_->nv);
  mju_copy(data_->ctrl, control.data(), model_->nu);
  mj_forward(model_, data_);
}

DynamicsTerms NeroMujocoSimulation::dynamics_terms() const {
  DynamicsTerms result;
  std::vector<mjtNum> mass(static_cast<std::size_t>(model_->nv) * static_cast<std::size_t>(model_->nv));
  mj_fullM(model_, mass.data(), data_->qM);
  mju_copy(gravity_data_->qpos, data_->qpos, model_->nq);
  mju_zero(gravity_data_->qvel, model_->nv);
  mj_forward(model_, gravity_data_);
  for (int row = 0; row < 7; ++row) {
    for (int column = 0; column < 7; ++column) result.mass(row, column) = mass[static_cast<std::size_t>(joint_dof_address_[row]) * model_->nv + joint_dof_address_[column]];
    result.gravity[row] = gravity_data_->qfrc_bias[joint_dof_address_[row]];
    result.coriolis[row] = data_->qfrc_bias[joint_dof_address_[row]] - result.gravity[row];
  }
  return result;
}

Pose NeroMujocoSimulation::body_pose(const std::string& body_name) const {
  const int body_id = require_id(model_, mjOBJ_BODY, body_name.c_str());
  Pose result;
  result.position = Vec3(data_->xpos[3 * body_id], data_->xpos[3 * body_id + 1], data_->xpos[3 * body_id + 2]);
  Eigen::Matrix3d rotation;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) rotation(row, column) = data_->xmat[9 * body_id + 3 * row + column];
  }
  result.orientation = Eigen::Quaterniond(rotation).normalized();
  return result;
}

void NeroMujocoSimulation::reset(const Vec7& joint_position) {
  if (!joint_position.allFinite()) throw std::invalid_argument("MuJoCo reset position must be finite");
  mj_resetData(model_, data_);
  for (int index = 0; index < 7; ++index) data_->qpos[joint_qpos_address_[index]] = joint_position[index];
  set_joint_torque(Vec7::Zero());
  set_gripper_width(0.040);
  mj_forward(model_, data_);
}

void NeroMujocoSimulation::set_joint_torque(const Vec7& torque_nm) {
  if (!torque_nm.allFinite()) throw std::invalid_argument("MuJoCo torque must be finite");
  for (int index = 0; index < 7; ++index) data_->ctrl[torque_actuator_[index]] = torque_nm[index];
}

void NeroMujocoSimulation::set_joint_disturbance(const Vec7& torque_nm) {
  if (!torque_nm.allFinite()) throw std::invalid_argument("MuJoCo disturbance torque must be finite");
  for (int index = 0; index < 7; ++index) data_->qfrc_applied[joint_dof_address_[index]] = torque_nm[index];
}

void NeroMujocoSimulation::set_gripper_width(double width_m) {
  if (!std::isfinite(width_m)) throw std::invalid_argument("MuJoCo gripper width must be finite");
  const double finger_travel = std::clamp(width_m * 0.5, 0.0, 0.040);
  data_->ctrl[gripper_actuator_[0]] = finger_travel;
  data_->ctrl[gripper_actuator_[1]] = finger_travel;
}

void NeroMujocoSimulation::step() { mj_step(model_, data_); }

void NeroMujocoSimulation::step(std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) step();
}

}  // namespace xrobot_cpp
