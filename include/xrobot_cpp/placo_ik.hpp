#pragma once

#include <memory>
#include <string>

#include "xrobot_cpp/dynamics_controller.hpp"

namespace xrobot_cpp {

class PlacoNeroIkSolver {
 public:
  explicit PlacoNeroIkSolver(const std::string& urdf_path, std::string tip_link = "gripper_tcp");
  ~PlacoNeroIkSolver();
  PlacoNeroIkSolver(const PlacoNeroIkSolver&) = delete;
  PlacoNeroIkSolver& operator=(const PlacoNeroIkSolver&) = delete;
  PlacoNeroIkSolver(PlacoNeroIkSolver&&) noexcept;
  PlacoNeroIkSolver& operator=(PlacoNeroIkSolver&&) noexcept;

  [[nodiscard]] ContinuousIkResult solve(const Pose& target, const Vec7& measured,
                                         const ContinuousIkOptions& options = {},
                                         double dt_s = 0.01,
                                         double max_joint_velocity_rad_s = 1.0);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xrobot_cpp
