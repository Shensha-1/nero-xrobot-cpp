#include "xrobot_cpp/dynamics_controller.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <limits>
#include <optional>
#include <vector>
#include <stdexcept>
#include <utility>

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/jacobian.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/chainiksolvervel_wdls.hpp>
#include <kdl/jntspaceinertiamatrix.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <urdf/model.h>

namespace xrobot_cpp {
namespace {

constexpr std::size_t kJointCount = 7;

KDL::JntArray to_kdl(const Vec7& vector) {
  KDL::JntArray result(kJointCount);
  for (std::size_t index = 0; index < kJointCount; ++index) result(index) = vector[static_cast<Eigen::Index>(index)];
  return result;
}

[[maybe_unused]] Vec7 from_kdl(const KDL::JntArray& vector) {
  if (vector.rows() != kJointCount) throw std::runtime_error("KDL joint vector is not 7 DoF");
  Vec7 result;
  for (std::size_t index = 0; index < kJointCount; ++index) result[static_cast<Eigen::Index>(index)] = vector(index);
  return result;
}

Pose from_kdl(const KDL::Frame& frame) {
  Pose pose;
  pose.position = Vec3(frame.p.x(), frame.p.y(), frame.p.z());
  double x{};
  double y{};
  double z{};
  double w{};
  frame.M.GetQuaternion(x, y, z, w);
  pose.orientation = Eigen::Quaterniond(w, x, y, z).normalized();
  return pose;
}

KDL::Frame to_kdl(const Pose& pose) {
  if (!pose.position.allFinite() || !pose.orientation.coeffs().allFinite()) {
    throw std::invalid_argument("IK target must be finite");
  }
  const Eigen::Quaterniond orientation = pose.orientation.normalized();
  return {KDL::Rotation::Quaternion(orientation.x(), orientation.y(), orientation.z(), orientation.w()),
          KDL::Vector(pose.position.x(), pose.position.y(), pose.position.z())};
}

void require_finite_nonnegative(const Vec7& value, const char* name) {
  if (!value.allFinite() || (value.array() < 0.0).any()) {
    throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
  }
}

}  // namespace

struct NeroDynamicsModel::Impl {
  KDL::Chain chain;
  Vec7 lower{Vec7::Zero()};
  Vec7 upper{Vec7::Zero()};
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk;
  std::unique_ptr<KDL::ChainJntToJacSolver> jacobian;
  std::unique_ptr<KDL::ChainIkSolverVel_wdls> ik_velocity;
  std::unique_ptr<KDL::ChainIkSolverPos_NR_JL> ik_position;
  std::unique_ptr<KDL::ChainIkSolverPos_LMA> ik_position_lma;
  std::unique_ptr<KDL::ChainDynParam> dynamics;
};

NeroDynamicsModel::NeroDynamicsModel(const std::string& urdf_path, std::string base_link, std::string tip_link)
    : impl_(std::make_unique<Impl>()) {
  KDL::Tree tree;
  if (!kdl_parser::treeFromFile(urdf_path, tree)) {
    throw std::runtime_error("cannot build KDL tree from URDF: " + urdf_path);
  }
  if (!tree.getChain(base_link, tip_link, impl_->chain) || impl_->chain.getNrOfJoints() != kJointCount) {
    throw std::runtime_error("URDF must expose exactly seven joints from " + base_link + " to " + tip_link);
  }

  urdf::Model urdf_model;
  if (!urdf_model.initFile(urdf_path)) throw std::runtime_error("cannot parse URDF joint limits: " + urdf_path);
  std::size_t joint_index = 0;
  for (unsigned int segment_index = 0; segment_index < impl_->chain.getNrOfSegments(); ++segment_index) {
    const KDL::Joint& joint = impl_->chain.getSegment(segment_index).getJoint();
    if (joint.getType() == KDL::Joint::None) continue;
    const auto urdf_joint = urdf_model.getJoint(joint.getName());
    if (!urdf_joint || !urdf_joint->limits) {
      throw std::runtime_error("URDF has no limits for KDL joint: " + joint.getName());
    }
    impl_->lower[static_cast<Eigen::Index>(joint_index)] = urdf_joint->limits->lower;
    impl_->upper[static_cast<Eigen::Index>(joint_index)] = urdf_joint->limits->upper;
    ++joint_index;
  }
  if (joint_index != kJointCount || !(impl_->lower.array() < impl_->upper.array()).all()) {
    throw std::runtime_error("invalid NERO URDF joint limits");
  }

  impl_->fk = std::make_unique<KDL::ChainFkSolverPos_recursive>(impl_->chain);
  impl_->jacobian = std::make_unique<KDL::ChainJntToJacSolver>(impl_->chain);
  impl_->ik_velocity = std::make_unique<KDL::ChainIkSolverVel_wdls>(impl_->chain, 1e-6, 150);
  impl_->ik_velocity->setLambda(0.02);
  impl_->ik_position = std::make_unique<KDL::ChainIkSolverPos_NR_JL>(
      impl_->chain, to_kdl(impl_->lower), to_kdl(impl_->upper), *impl_->fk, *impl_->ik_velocity, 500, 1e-4);
  Eigen::Matrix<double, 6, 1> lma_weights;
  lma_weights << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
  impl_->ik_position_lma = std::make_unique<KDL::ChainIkSolverPos_LMA>(impl_->chain, lma_weights, 1e-5, 2000, 1e-12);
  impl_->dynamics = std::make_unique<KDL::ChainDynParam>(impl_->chain, KDL::Vector(0.0, 0.0, -9.81));
}

NeroDynamicsModel::~NeroDynamicsModel() = default;
NeroDynamicsModel::NeroDynamicsModel(NeroDynamicsModel&&) noexcept = default;
NeroDynamicsModel& NeroDynamicsModel::operator=(NeroDynamicsModel&&) noexcept = default;

Pose NeroDynamicsModel::forward_kinematics(const Vec7& joints) const {
  if (!joints.allFinite()) throw std::invalid_argument("forward-kinematics joints must be finite");
  KDL::Frame frame;
  if (impl_->fk->JntToCart(to_kdl(joints), frame) < 0) throw std::runtime_error("KDL forward kinematics failed");
  return from_kdl(frame);
}

Vec7 NeroDynamicsModel::solve_ik(const Pose& target, const Vec7& seed, double orientation_tolerance_rad,
                                   double position_tolerance_m) const {
  if (!std::isfinite(orientation_tolerance_rad) || orientation_tolerance_rad <= 0.0 || orientation_tolerance_rad > std::numbers::pi ||
      !std::isfinite(position_tolerance_m) || position_tolerance_m <= 0.0) {
    throw std::invalid_argument("IK tolerances must be positive and orientation tolerance no greater than pi");
  }
  if (!seed.allFinite()) throw std::invalid_argument("IK seed must be finite");
  const KDL::Frame target_frame = to_kdl(target);
  const auto task_error = [&](const Vec7& joints) {
    KDL::Frame current;
    if (impl_->fk->JntToCart(to_kdl(joints), current) < 0) throw std::runtime_error("KDL forward kinematics failed");
    const KDL::Twist twist = KDL::diff(current, target_frame);
    Vec6 error;
    error << twist.vel.x(), twist.vel.y(), twist.vel.z(), twist.rot.x(), twist.rot.y(), twist.rot.z();
    return error;
  };
  Vec7 joints = seed.cwiseMax(impl_->lower + Vec7::Constant(1e-4)).cwiseMin(impl_->upper - Vec7::Constant(1e-4));
  constexpr double kDamping = 0.035;
  constexpr double kMaxStepRad = 0.025;
  const bool soft_orientation = orientation_tolerance_rad > 1e-3;
  for (int iteration = 0; iteration < 600; ++iteration) {
    const Vec6 error = task_error(joints);
    if (error.head<3>().norm() <= position_tolerance_m && error.tail<3>().norm() <= orientation_tolerance_rad) return joints;
    KDL::Jacobian jacobian(kJointCount);
    if (impl_->jacobian->JntToJac(to_kdl(joints), jacobian) < 0) throw std::runtime_error("KDL Jacobian calculation failed");
    const Eigen::Matrix<double, 6, 7> jacobian_matrix = jacobian.data;
    Vec7 delta;
    if (soft_orientation) {
      // Translation is primary; orientation is corrected in its redundant null space.
      const Eigen::Matrix<double, 3, 7> position_jacobian = jacobian_matrix.topRows<3>();
      const Eigen::Matrix3d position_regularized = position_jacobian * position_jacobian.transpose() +
          Eigen::Matrix3d::Identity() * (0.005 * 0.005);
      const Eigen::Matrix<double, 7, 3> position_pinv = position_jacobian.transpose() *
          position_regularized.ldlt().solve(Eigen::Matrix3d::Identity());
      const Mat7 nullspace = Mat7::Identity() - position_pinv * position_jacobian;
      const Eigen::Matrix<double, 3, 7> orientation_jacobian = jacobian_matrix.bottomRows<3>() * nullspace;
      const Eigen::Matrix3d orientation_regularized = orientation_jacobian * orientation_jacobian.transpose() +
          Eigen::Matrix3d::Identity() * (kDamping * kDamping);
      const Eigen::Matrix<double, 7, 3> orientation_pinv = nullspace * orientation_jacobian.transpose() *
          orientation_regularized.ldlt().solve(Eigen::Matrix3d::Identity());
      delta = position_pinv * error.head<3>() + orientation_pinv * error.tail<3>();
    } else {
      const Eigen::Matrix<double, 6, 6> regularized = jacobian_matrix * jacobian_matrix.transpose() +
          Eigen::Matrix<double, 6, 6>::Identity() * (kDamping * kDamping);
      const Eigen::Matrix<double, 7, 6> damped_pinv = jacobian_matrix.transpose() *
          regularized.ldlt().solve(Eigen::Matrix<double, 6, 6>::Identity());
      delta = damped_pinv * error;
    }
    Vec7 limit_avoidance = Vec7::Zero();
    for (Eigen::Index index = 0; index < 7; ++index) {
      const double lower_margin = std::max(0.02, joints[index] - impl_->lower[index]);
      const double upper_margin = std::max(0.02, impl_->upper[index] - joints[index]);
      limit_avoidance[index] = 0.0005 * (1.0 / (lower_margin * lower_margin) - 1.0 / (upper_margin * upper_margin));
    }
    delta += 0.05 * limit_avoidance;
    const double largest_delta = delta.cwiseAbs().maxCoeff();
    if (!std::isfinite(largest_delta)) throw std::runtime_error("local DLS IK produced a non-finite update");
    if (largest_delta > kMaxStepRad) delta *= kMaxStepRad / largest_delta;
    const double current_cost = error.head<3>().squaredNorm() / (position_tolerance_m * position_tolerance_m) +
        std::pow(std::max(0.0, error.tail<3>().norm() - orientation_tolerance_rad) / orientation_tolerance_rad, 2.0);
    bool accepted = false;
    double step_scale = 1.0;
    for (int trial = 0; trial < 8; ++trial) {
      const Vec7 candidate = (joints + step_scale * delta)
          .cwiseMax(impl_->lower + Vec7::Constant(1e-4))
          .cwiseMin(impl_->upper - Vec7::Constant(1e-4));
      const Vec6 candidate_error = task_error(candidate);
      const double candidate_cost = candidate_error.head<3>().squaredNorm() / (position_tolerance_m * position_tolerance_m) +
          std::pow(std::max(0.0, candidate_error.tail<3>().norm() - orientation_tolerance_rad) / orientation_tolerance_rad, 2.0);
      if (candidate_cost < current_cost) { joints = candidate; accepted = true; break; }
      step_scale *= 0.5;
    }
    if (!accepted) break;
  }
  const Vec6 residual = task_error(joints);
  std::ostringstream message;
  message << "continuous DLS IK did not converge: position_error_m=" << residual.head<3>().norm()
          << " orientation_error_rad=" << residual.tail<3>().norm() << " orientation_tolerance_rad=" << orientation_tolerance_rad
          << " max_joint_delta_rad=" << (joints - seed).cwiseAbs().maxCoeff();
  throw std::runtime_error(message.str());
}

ContinuousIkResult NeroDynamicsModel::solve_continuous_ik(const Pose& target, const Vec7& seed,
                                                           const ContinuousIkOptions& options) const {
  if (!seed.allFinite() || !std::isfinite(options.position_weight) || !std::isfinite(options.orientation_weight) ||
      !std::isfinite(options.position_tolerance_m) || !std::isfinite(options.orientation_tolerance_rad) ||
      !std::isfinite(options.continuity_weight) || !std::isfinite(options.joint_limit_weight) ||
      !std::isfinite(options.soft_limit_margin_rad) || !std::isfinite(options.damping) ||
      !std::isfinite(options.max_iteration_step_rad) || options.position_weight <= 0.0 ||
      options.orientation_weight <= 0.0 || options.position_tolerance_m <= 0.0 ||
      options.orientation_tolerance_rad <= 0.0 || options.orientation_tolerance_rad > std::numbers::pi ||
      options.continuity_weight < 0.0 || options.joint_limit_weight < 0.0 ||
      options.soft_limit_margin_rad <= 0.0 || options.damping <= 0.0 ||
      options.max_iteration_step_rad <= 0.0 || options.max_iterations <= 0) {
    throw std::invalid_argument("invalid continuous IK options");
  }
  const KDL::Frame target_frame = to_kdl(target);
  const Vec7 lower = impl_->lower + Vec7::Constant(1e-4);
  const Vec7 upper = impl_->upper - Vec7::Constant(1e-4);
  Vec7 joints = seed.cwiseMax(lower).cwiseMin(upper);
  const auto task_error = [&](const Vec7& q) {
    KDL::Frame current;
    if (impl_->fk->JntToCart(to_kdl(q), current) < 0) throw std::runtime_error("KDL forward kinematics failed");
    const KDL::Twist twist = KDL::diff(current, target_frame);
    Vec6 error;
    error << twist.vel.x(), twist.vel.y(), twist.vel.z(), twist.rot.x(), twist.rot.y(), twist.rot.z();
    return error;
  };
  const auto barrier = [&](const Vec7& q, Vec7* negative_gradient) {
    double cost = 0.0;
    if (negative_gradient) negative_gradient->setZero();
    for (Eigen::Index i = 0; i < 7; ++i) {
      const double lo = std::max(1e-4, q[i] - lower[i]);
      const double hi = std::max(1e-4, upper[i] - q[i]);
      const double m2 = options.soft_limit_margin_rad * options.soft_limit_margin_rad;
      cost += m2 / (lo * lo) + m2 / (hi * hi);
      if (negative_gradient) (*negative_gradient)[i] = 2.0 * m2 * (1.0 / (lo * lo * lo) - 1.0 / (hi * hi * hi));
    }
    return cost;
  };
  const auto objective = [&](const Vec7& q, const Vec6& error) {
    const double position_scale = options.position_weight / options.position_tolerance_m;
    const double orientation_scale = options.orientation_weight / options.orientation_tolerance_rad;
    return position_scale * position_scale * error.head<3>().squaredNorm() +
        orientation_scale * orientation_scale * error.tail<3>().squaredNorm() +
        options.continuity_weight * (q - seed).squaredNorm() + options.joint_limit_weight * barrier(q, nullptr);
  };
  Vec6 error = task_error(joints);
  for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
    if (error.head<3>().norm() <= options.position_tolerance_m &&
        error.tail<3>().norm() <= options.orientation_tolerance_rad) break;
    KDL::Jacobian jacobian(kJointCount);
    if (impl_->jacobian->JntToJac(to_kdl(joints), jacobian) < 0) throw std::runtime_error("KDL Jacobian calculation failed");
    const double position_scale = options.position_weight / options.position_tolerance_m;
    const double orientation_scale = options.orientation_weight / options.orientation_tolerance_rad;
    Eigen::Matrix<double, 6, 7> weighted_jacobian = jacobian.data;
    weighted_jacobian.topRows<3>() *= position_scale;
    weighted_jacobian.bottomRows<3>() *= orientation_scale;
    Vec6 weighted_error = error;
    weighted_error.head<3>() *= position_scale;
    weighted_error.tail<3>() *= orientation_scale;
    const Eigen::JacobiSVD<Eigen::Matrix<double, 6, 7>> svd(weighted_jacobian, 0);
    const double sigma_min = svd.singularValues()[5];
    const double damping = options.damping + std::max(0.0, 0.05 - sigma_min);
    Vec7 negative_gradient;
    (void)barrier(joints, &negative_gradient);
    const Mat7 hessian = weighted_jacobian.transpose() * weighted_jacobian +
        Mat7::Identity() * (damping * damping + options.continuity_weight);
    Vec7 delta = hessian.ldlt().solve(weighted_jacobian.transpose() * weighted_error -
        options.continuity_weight * (joints - seed) + options.joint_limit_weight * negative_gradient);
    const double maximum = delta.cwiseAbs().maxCoeff();
    if (!std::isfinite(maximum)) throw std::runtime_error("unified continuous IK produced a non-finite update");
    if (maximum > options.max_iteration_step_rad) delta *= options.max_iteration_step_rad / maximum;
    const double current_cost = objective(joints, error);
    bool accepted = false;
    double scale = 1.0;
    for (int trial = 0; trial < 8; ++trial) {
      const Vec7 candidate = (joints + scale * delta).cwiseMax(lower).cwiseMin(upper);
      const Vec6 candidate_error = task_error(candidate);
      if (objective(candidate, candidate_error) < current_cost) {
        joints = candidate;
        error = candidate_error;
        accepted = true;
        break;
      }
      scale *= 0.5;
    }
    if (!accepted) break;
  }
  ContinuousIkResult result{.joints = joints, .position_error_m = error.head<3>().norm(),
                            .orientation_error_rad = error.tail<3>().norm(),
                            .minimum_joint_margin_rad = std::min((joints - impl_->lower).minCoeff(),
                                                                 (impl_->upper - joints).minCoeff())};
  if (result.position_error_m > options.position_tolerance_m || result.orientation_error_rad > options.orientation_tolerance_rad) {
    std::ostringstream message;
    message << "unified continuous IK did not converge: position_error_m=" << result.position_error_m
            << " orientation_error_rad=" << result.orientation_error_rad
            << " min_joint_margin_rad=" << result.minimum_joint_margin_rad;
    throw std::runtime_error(message.str());
  }
  return result;
}

ContinuousIkResult NeroDynamicsModel::solve_redundant_continuous_ik(
    const Pose& target, const Vec7& reference, const ContinuousIkOptions& options,
    double minimum_joint_margin_rad) const {
  if (!reference.allFinite() || !std::isfinite(minimum_joint_margin_rad) || minimum_joint_margin_rad < 0.0) {
    throw std::invalid_argument("invalid redundant IK reference or joint margin");
  }
  const auto valid = [&](const ContinuousIkResult& result) {
    return result.position_error_m <= options.position_tolerance_m &&
        result.orientation_error_rad <= options.orientation_tolerance_rad;
  };
  try {
    const ContinuousIkResult primary = solve_continuous_ik(target, reference, options);
    if (valid(primary) && primary.minimum_joint_margin_rad >= minimum_joint_margin_rad) return primary;
  } catch (const std::exception&) {
    // Retry below from deterministic redundant seeds.
  }

  const Vec7 lower = impl_->lower + Vec7::Constant(1e-4);
  const Vec7 upper = impl_->upper - Vec7::Constant(1e-4);
  const Vec7 center = 0.5 * (lower + upper);
  const std::array<double, 4> center_blends{0.20, 0.45, 0.70, 1.00};
  std::vector<Vec7> seeds;
  seeds.reserve(center_blends.size() + 16);
  for (const double blend : center_blends) {
    seeds.push_back(((1.0 - blend) * reference + blend * center).cwiseMax(lower).cwiseMin(upper));
  }
  // The final joint is redundant for the 6-D TCP task. Probe both sides of
  // its interval to escape a wrist branch without imposing a preferred pose.
  for (const double direction : {-1.0, 1.0}) {
    Vec7 wrist_seed = reference;
    wrist_seed[6] = std::clamp(center[6] + direction * 0.30 * (upper[6] - lower[6]), lower[6], upper[6]);
    seeds.push_back(wrist_seed);
  }
  // Probe alternate elbow/wrist postures without making any one branch preferred.
  for (Eigen::Index joint = 0; joint < 7; ++joint) {
    for (const double fraction : {0.20, 0.80}) {
      Vec7 branch_seed = center;
      branch_seed[joint] = lower[joint] + fraction * (upper[joint] - lower[joint]);
      seeds.push_back(branch_seed);
    }
  }

  std::optional<ContinuousIkResult> best;
  double best_score = std::numeric_limits<double>::infinity();
  for (const Vec7& seed : seeds) {
    ContinuousIkOptions branch_options = options;
    branch_options.continuity_weight = std::min(options.continuity_weight, 0.001);
    try {
      const ContinuousIkResult candidate = solve_continuous_ik(target, seed, branch_options);
      if (!valid(candidate)) continue;
      const double distance = (candidate.joints - reference).squaredNorm();
      // A soft margin ranks solutions but never makes a TCP-valid solution invalid.
      const double margin_shortfall = std::max(0.0, minimum_joint_margin_rad - candidate.minimum_joint_margin_rad);
      const double score = distance + 4.0 * margin_shortfall * margin_shortfall;
      if (score < best_score) {
        best = candidate;
        best_score = score;
      }
    } catch (const std::exception&) {
      // This seed belongs to a non-convergent branch. Try the remaining ones.
    }
  }
  if (best) return *best;
  throw std::runtime_error("redundant continuous IK found no branch satisfying the requested TCP pose");
}

Vec7 NeroDynamicsModel::solve_ik_step(const Pose& target, const Vec7& seed, double max_joint_step_rad) const {
  if (!seed.allFinite()) throw std::invalid_argument("IK seed must be finite");
  if (!std::isfinite(max_joint_step_rad) || max_joint_step_rad <= 0.0) {
    throw std::invalid_argument("maximum IK joint step must be positive");
  }
  KDL::Frame current;
  if (impl_->fk->JntToCart(to_kdl(seed), current) < 0) throw std::runtime_error("KDL forward kinematics failed");
  const KDL::Twist twist = KDL::diff(current, to_kdl(target));
  Vec6 error;
  error << twist.vel.x(), twist.vel.y(), twist.vel.z(), twist.rot.x(), twist.rot.y(), twist.rot.z();
  KDL::Jacobian jacobian(kJointCount);
  if (impl_->jacobian->JntToJac(to_kdl(seed), jacobian) < 0) {
    throw std::runtime_error("KDL Jacobian calculation failed");
  }
  const Eigen::Matrix<double, 6, 7> matrix = jacobian.data;
  constexpr double kStepDamping = 0.035;
  const Eigen::Matrix<double, 6, 6> regularized = matrix * matrix.transpose() +
      Eigen::Matrix<double, 6, 6>::Identity() * (kStepDamping * kStepDamping);
  const Vec7 raw_delta = matrix.transpose() * regularized.ldlt().solve(error);
  const Vec7 bounded_delta = raw_delta.cwiseMax(-max_joint_step_rad).cwiseMin(max_joint_step_rad);
  return (seed + bounded_delta).cwiseMax(impl_->lower).cwiseMin(impl_->upper);
}

DynamicsTerms NeroDynamicsModel::dynamics(const Vec7& joints, const Vec7& velocity) const {
  if (!joints.allFinite() || !velocity.allFinite()) throw std::invalid_argument("dynamics state must be finite");
  const KDL::JntArray q = to_kdl(joints);
  const KDL::JntArray qdot = to_kdl(velocity);
  KDL::JntSpaceInertiaMatrix mass(kJointCount);
  KDL::JntArray coriolis(kJointCount);
  KDL::JntArray gravity(kJointCount);
  if (impl_->dynamics->JntToMass(q, mass) < 0 || impl_->dynamics->JntToCoriolis(q, qdot, coriolis) < 0 ||
      impl_->dynamics->JntToGravity(q, gravity) < 0) {
    throw std::runtime_error("KDL inverse dynamics failed");
  }
  DynamicsTerms result;
  for (std::size_t row = 0; row < kJointCount; ++row) {
    result.coriolis[static_cast<Eigen::Index>(row)] = coriolis(row);
    result.gravity[static_cast<Eigen::Index>(row)] = gravity(row);
    for (std::size_t column = 0; column < kJointCount; ++column) {
      result.mass(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) = mass(row, column);
    }
  }
  return result;
}

const Vec7& NeroDynamicsModel::lower_limits() const { return impl_->lower; }
const Vec7& NeroDynamicsModel::upper_limits() const { return impl_->upper; }

JointVelocityEstimator::JointVelocityEstimator(double cutoff_hz) : cutoff_hz_(cutoff_hz) {
  if (!std::isfinite(cutoff_hz_) || cutoff_hz_ <= 0.0) throw std::invalid_argument("velocity cutoff must be positive");
}

void JointVelocityEstimator::reset(const Vec7& position, Clock::time_point now) {
  if (!position.allFinite()) throw std::invalid_argument("velocity-estimator position must be finite");
  position_ = position;
  velocity_.setZero();
  last_ = now;
  initialized_ = true;
}

Vec7 JointVelocityEstimator::update(const Vec7& position, Clock::time_point now) {
  if (!initialized_) {
    reset(position, now);
    return velocity_;
  }
  const double dt = std::chrono::duration<double>(now - last_).count();
  if (!std::isfinite(dt) || dt <= 0.0) return velocity_;
  if (dt > 0.10) {
    reset(position, now);
    return velocity_;
  }
  const Vec7 raw = (position - position_) / dt;
  const double alpha = 1.0 - std::exp(-2.0 * std::numbers::pi * cutoff_hz_ * dt);
  velocity_ += alpha * (raw - velocity_);
  position_ = position;
  last_ = now;
  return velocity_;
}

JerkLimitedJointPlanner::JerkLimitedJointPlanner(Vec7 max_velocity_rad_s, Vec7 max_acceleration_rad_s2, Vec7 max_jerk_rad_s3)
    : max_velocity_rad_s_(std::move(max_velocity_rad_s)),
      max_acceleration_rad_s2_(std::move(max_acceleration_rad_s2)),
      max_jerk_rad_s3_(std::move(max_jerk_rad_s3)) {
  require_finite_nonnegative(max_velocity_rad_s_, "S-curve maximum velocity");
  require_finite_nonnegative(max_acceleration_rad_s2_, "S-curve maximum acceleration");
  require_finite_nonnegative(max_jerk_rad_s3_, "S-curve maximum jerk");
  if ((max_velocity_rad_s_.array() <= 0.0).any() ||
      (max_acceleration_rad_s2_.array() <= 0.0).any() ||
      (max_jerk_rad_s3_.array() <= 0.0).any()) {
    throw std::invalid_argument("S-curve limits must be positive");
  }
}

void JerkLimitedJointPlanner::reset(const Vec7& position) {
  if (!position.allFinite()) throw std::invalid_argument("S-curve reset position must be finite");
  goal_ = position;
  state_.position = position;
  state_.velocity.setZero();
  state_.acceleration.setZero();
  initialized_ = true;
}

void JerkLimitedJointPlanner::set_goal(const Vec7& position) {
  if (!position.allFinite()) throw std::invalid_argument("S-curve goal must be finite");
  if (!initialized_) reset(position);
  goal_ = position;
}

JointTrajectoryState JerkLimitedJointPlanner::step(double dt_s) {
  if (!initialized_) throw std::logic_error("S-curve planner must be reset before stepping");
  if (!std::isfinite(dt_s) || dt_s <= 0.0 || dt_s > 0.10) {
    throw std::invalid_argument("S-curve time step must be in (0, 0.10]");
  }
  for (Eigen::Index joint = 0; joint < state_.position.size(); ++joint) {
    const double error = goal_[joint] - state_.position[joint];
    if (std::abs(error) < 1e-7 && std::abs(state_.velocity[joint]) < 1e-6 &&
        std::abs(state_.acceleration[joint]) < 1e-5) {
      state_.position[joint] = goal_[joint];
      state_.velocity[joint] = 0.0;
      state_.acceleration[joint] = 0.0;
      continue;
    }
    constexpr double kNaturalFrequencyRadS = 4.5;
    const double acceleration_goal = std::clamp(
        kNaturalFrequencyRadS * kNaturalFrequencyRadS * error -
            2.0 * kNaturalFrequencyRadS * state_.velocity[joint],
        -max_acceleration_rad_s2_[joint], max_acceleration_rad_s2_[joint]);
    const double acceleration_delta = std::clamp(acceleration_goal - state_.acceleration[joint],
                                                  -max_jerk_rad_s3_[joint] * dt_s,
                                                  max_jerk_rad_s3_[joint] * dt_s);
    state_.acceleration[joint] = std::clamp(state_.acceleration[joint] + acceleration_delta,
                                             -max_acceleration_rad_s2_[joint], max_acceleration_rad_s2_[joint]);
    state_.velocity[joint] = std::clamp(state_.velocity[joint] + state_.acceleration[joint] * dt_s,
                                         -max_velocity_rad_s_[joint], max_velocity_rad_s_[joint]);
    state_.position[joint] += state_.velocity[joint] * dt_s;
    if ((error > 0.0 && state_.position[joint] >= goal_[joint]) ||
        (error < 0.0 && state_.position[joint] <= goal_[joint])) {
      state_.position[joint] = goal_[joint];
      state_.velocity[joint] = 0.0;
      state_.acceleration[joint] = 0.0;
    }
  }
  return state_;
}

JointDriveModel JointDriveModel::mujoco_default() {
  // These values match assets/mujoco/nero_torque.xml. They are simulation
  // parameters, not an unverified statement about the real NERO drivetrain.
  JointDriveModel result;
  result.reflected_inertia_kg_m2 = Vec7::Constant(0.015);
  result.viscous_friction_nm_s_rad = Vec7::Constant(0.20);
  result.coulomb_friction_nm = Vec7::Constant(0.05);
  result.coulomb_smoothing_rad_s = 0.01;
  return result;
}

ComputedTorqueController::ComputedTorqueController(Vec7 spring_stiffness_nm_rad, Vec7 damping_nm_s_rad,
                                                   Vec7 torque_limit_nm, JointDriveModel drive_model)
    : spring_stiffness_nm_rad_(std::move(spring_stiffness_nm_rad)), damping_nm_s_rad_(std::move(damping_nm_s_rad)),
      torque_limit_nm_(std::move(torque_limit_nm)), drive_model_(std::move(drive_model)) {
  require_finite_nonnegative(spring_stiffness_nm_rad_, "virtual spring stiffness");
  require_finite_nonnegative(damping_nm_s_rad_, "virtual damping");
  if (!torque_limit_nm_.allFinite() || (torque_limit_nm_.array() <= 0.0).any() || (torque_limit_nm_.array() > 16.0).any()) {
    throw std::invalid_argument("torque limit must be within (0, 16] Nm");
  }
  require_finite_nonnegative(drive_model_.reflected_inertia_kg_m2, "reflected inertia");
  require_finite_nonnegative(drive_model_.viscous_friction_nm_s_rad, "viscous friction");
  require_finite_nonnegative(drive_model_.coulomb_friction_nm, "coulomb friction");
  if (!std::isfinite(drive_model_.coulomb_smoothing_rad_s) || drive_model_.coulomb_smoothing_rad_s <= 0.0) {
    throw std::invalid_argument("coulomb smoothing velocity must be positive");
  }
}

ComputedTorqueCommand ComputedTorqueController::compute(const DynamicsTerms& dynamics,
                                                         const Vec7& measured_position,
                                                         const Vec7& measured_velocity,
                                                         const JointTrajectoryState& desired) const {
  if (!measured_position.allFinite() || !measured_velocity.allFinite() || !desired.position.allFinite() ||
      !desired.velocity.allFinite() || !desired.acceleration.allFinite() || !dynamics.mass.allFinite() ||
      !dynamics.coriolis.allFinite() || !dynamics.gravity.allFinite()) {
    throw std::invalid_argument("computed-torque inputs must be finite");
  }
  ComputedTorqueCommand result;
  result.desired = desired;
  result.dynamics = dynamics;
  result.commanded_acceleration = desired.acceleration;
  result.reflected_inertia_torque_nm = drive_model_.reflected_inertia_kg_m2.cwiseProduct(result.commanded_acceleration);
  result.viscous_friction_torque_nm = drive_model_.viscous_friction_nm_s_rad.cwiseProduct(measured_velocity);
  result.coulomb_friction_torque_nm = drive_model_.coulomb_friction_nm.cwiseProduct(
      (measured_velocity.array() / drive_model_.coulomb_smoothing_rad_s).tanh().matrix());
  result.virtual_spring_torque_nm = spring_stiffness_nm_rad_.cwiseProduct(desired.position - measured_position);
  result.virtual_damping_torque_nm = damping_nm_s_rad_.cwiseProduct(desired.velocity - measured_velocity);
  const Vec7 torque = result.dynamics.mass * result.commanded_acceleration + result.reflected_inertia_torque_nm +
                      result.dynamics.coriolis + result.dynamics.gravity + result.viscous_friction_torque_nm +
                      result.coulomb_friction_torque_nm +
                      result.virtual_spring_torque_nm + result.virtual_damping_torque_nm;
  result.feedforward_torque_nm = torque.cwiseMax(-torque_limit_nm_).cwiseMin(torque_limit_nm_);
  return result;
}

ComputedTorqueCommand ComputedTorqueController::compute(const NeroDynamicsModel& model,
                                                       const Vec7& measured_position,
                                                       const Vec7& measured_velocity,
                                                       const JointTrajectoryState& desired) const {
  return compute(model.dynamics(measured_position, measured_velocity), measured_position, measured_velocity, desired);
}

MitSetpoint ComputedTorqueController::mit_setpoint(const ComputedTorqueCommand& command, std::size_t joint_index,
                                                    double inner_kp_nm_rad, double inner_kd_nm_s_rad) const {
  if (joint_index >= kJointCount || !std::isfinite(inner_kp_nm_rad) || !std::isfinite(inner_kd_nm_s_rad) ||
      inner_kp_nm_rad < 0.0 || inner_kp_nm_rad > 500.0 || inner_kd_nm_s_rad < -5.0 || inner_kd_nm_s_rad > 5.0) {
    throw std::invalid_argument("invalid MIT setpoint parameters");
  }
  const Eigen::Index index = static_cast<Eigen::Index>(joint_index);
  return {.position_rad = command.desired.position[index], .velocity_rad_s = command.desired.velocity[index],
          .kp_nm_rad = inner_kp_nm_rad, .kd_nm_s_rad = inner_kd_nm_s_rad,
          .feedforward_torque_nm = command.feedforward_torque_nm[index]};
}

}  // namespace xrobot_cpp
