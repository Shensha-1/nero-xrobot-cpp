#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "xrobot_cpp/control_core.hpp"
#include "xrobot_cpp/dynamics_controller.hpp"
#include "xrobot_cpp/nero_can.hpp"
#include "xrobot_cpp/nero_safety.hpp"
#ifdef XROBOT_CPP_WITH_PLACO
#include "xrobot_cpp/placo_ik.hpp"
#endif

namespace {
constexpr double kNeroLowSpeedSpringStiffnessNmRad = 8.0;
constexpr double kNeroLowSpeedDampingNmSRad = 6.0;
volatile std::sig_atomic_t g_mit_stop_requested = 0;
void request_mit_stop(int) { g_mit_stop_requested = 1; }

// NERO firmware v1.11 encodes MIT t_ff as signed 12-bit +/-16 Nm on
// every joint. Keep 1 Nm of encoding headroom in the C++ command gate.
// This is a protocol command limit, not a motor continuous-rating claim.
xrobot_cpp::Vec7 mit_support_torque_limits() {
  return xrobot_cpp::Vec7::Constant(15.0);
}

#ifdef XROBOT_CPP_WITH_XR
Eigen::Matrix3d load_absolute_controller_to_tcp_rotation(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open absolute-orientation calibration: " + path.string());
  const std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  if (!std::regex_search(json, std::regex(R"(\"absolute_orientation_calibrated\"\s*:\s*true)"))) {
    throw std::runtime_error("absolute-orientation calibration is not confirmed in " + path.string());
  }
  const std::size_t key = json.find("\"controller_to_tcp_rotation\"");
  if (key == std::string::npos) throw std::runtime_error("calibration lacks controller_to_tcp_rotation");
  const std::size_t begin = json.find("[", key);
  if (begin == std::string::npos) throw std::runtime_error("calibration rotation array is malformed");
  const std::string values = json.substr(begin);
  const std::regex number(R"([-+]?(?:[0-9]*\.)?[0-9]+(?:[eE][-+]?[0-9]+)?)");
  Eigen::Matrix3d rotation;
  auto it = std::sregex_iterator(values.begin(), values.end(), number);
  const auto end = std::sregex_iterator();
  for (int index = 0; index < 9; ++index) {
    if (it == end) throw std::runtime_error("calibration rotation needs nine finite values");
    const double value = std::stod(it->str());
    if (!std::isfinite(value)) throw std::runtime_error("calibration rotation contains non-finite value");
    rotation(index / 3, index % 3) = value;
    ++it;
  }
  if (!(rotation.transpose() * rotation).isApprox(Eigen::Matrix3d::Identity(), 1e-8) ||
      std::abs(rotation.determinant() - 1.0) > 1e-8) {
    throw std::runtime_error("controller_to_tcp_rotation is not a proper rotation");
  }
  return rotation;
}

Eigen::Matrix3d load_calibrated_base_from_xr_rotation(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open XR calibration: " + path.string());
  const std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  const std::size_t calibrated_key = json.find("\"base_from_xr_rotation\"");
  const std::size_t key = calibrated_key == std::string::npos ? json.find("\"base_from_view_rotation\"") : calibrated_key;
  if (key == std::string::npos) throw std::runtime_error("calibration lacks base_from_xr_rotation or base_from_view_rotation");
  const std::size_t begin = json.find("[", key);
  if (begin == std::string::npos) throw std::runtime_error("calibration base rotation array is malformed");
  const std::string values = json.substr(begin);
  const std::regex number(R"([-+]?(?:[0-9]*\.)?[0-9]+(?:[eE][-+]?[0-9]+)?)");
  Eigen::Matrix3d base_from_view;
  auto it = std::sregex_iterator(values.begin(), values.end(), number);
  const auto end = std::sregex_iterator();
  for (int index = 0; index < 9; ++index) {
    if (it == end) throw std::runtime_error("base_from_view_rotation needs nine finite values");
    const double value = std::stod(it->str());
    if (!std::isfinite(value)) throw std::runtime_error("base_from_view_rotation contains non-finite value");
    base_from_view(index / 3, index % 3) = value;
    ++it;
  }
  if (!(base_from_view.transpose() * base_from_view).isApprox(Eigen::Matrix3d::Identity(), 1e-8) ||
      std::abs(base_from_view.determinant() - 1.0) > 1e-8) {
    throw std::runtime_error("base_from_view_rotation is not a proper rotation");
  }
  if (calibrated_key != std::string::npos) return base_from_view;
  Eigen::Matrix3d xr_to_world;
  xr_to_world << 0.0, 0.0, -1.0,
                 -1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0;
  return base_from_view * xr_to_world;
}
#endif

// Real MIT control deliberately uses only the URDF gravity term as t_ff.
// Inertial and Coriolis compensation are left out until they have an
// independently validated torque model.  The firmware MIT Kp/Kd fields carry
// the low-stiffness position/velocity impedance around this nominal support.
xrobot_cpp::Vec7 nominal_gravity_feedforward(const xrobot_cpp::DynamicsTerms& dynamics,
                                              const xrobot_cpp::Vec7& torque_limits) {
  if (!dynamics.gravity.allFinite() || !torque_limits.allFinite()) {
    throw std::invalid_argument("gravity feedforward inputs must be finite");
  }
  return dynamics.gravity.cwiseMax(-torque_limits).cwiseMin(torque_limits);
}

// Slow integral trim removes the static residual left by an imperfect mass or
// torque scale model.  It is deliberately bounded, updated only at low joint
// speed, and starts from zero on every MIT entry; G(q) remains the nominal
// support command.
void update_gravity_trim(xrobot_cpp::Vec7* trim_nm, const xrobot_cpp::Vec7& position_error,
                         const xrobot_cpp::Vec7& measured_velocity, double dt_s) {
  if (trim_nm == nullptr || !position_error.allFinite() || !measured_velocity.allFinite() ||
      !std::isfinite(dt_s) || dt_s <= 0.0) {
    throw std::invalid_argument("invalid gravity-trim inputs");
  }
  constexpr double kIntegralGainNmRadS = 8.0;
  constexpr double kVelocityActivationRadS = 0.05;
  constexpr double kTrimLimitNm = 0.25;
  for (Eigen::Index joint = 0; joint < trim_nm->size(); ++joint) {
    if (std::abs(measured_velocity[joint]) <= kVelocityActivationRadS) {
      (*trim_nm)[joint] = std::clamp((*trim_nm)[joint] + kIntegralGainNmRadS * position_error[joint] * dt_s,
                                     -kTrimLimitNm, kTrimLimitNm);
    }
  }
}

std::string json_vector(const xrobot_cpp::Vec7& values) {
  std::ostringstream out;
  out << std::setprecision(17) << '[';
  for (Eigen::Index index = 0; index < values.size(); ++index) {
    if (index != 0) out << ',';
    out << values[index];
  }
  return out.str() + ']';
}

[[maybe_unused]] std::string json_quaternion(const Eigen::Quaterniond& rotation) {
  const Eigen::Quaterniond q = rotation.normalized();
  std::ostringstream out;
  out << std::setprecision(17) << '[' << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << ']';
  return out.str();
}

void write_mit_telemetry(std::ofstream* output, std::uint64_t monotonic_ns,
                         const xrobot_cpp::NeroFeedback& feedback,
                         const xrobot_cpp::Vec7& velocity,
                         const xrobot_cpp::JointTrajectoryState& desired,
                         const xrobot_cpp::DynamicsTerms& dynamics,
                         const xrobot_cpp::Vec7& feedforward_torque,
                         const xrobot_cpp::Vec7& kp, const xrobot_cpp::Vec7& kd,
                         const std::string_view mode,
                         const double gripper_width_m = std::numeric_limits<double>::quiet_NaN()) {
  if (output == nullptr || !*output) return;
  *output << "{\"event\":\"mit_cycle\",\"mode\":\"" << mode
          << "\",\"monotonic_ns\":" << monotonic_ns
          << ",\"joint_state_sequence\":" << feedback.joint_state_sequence
          << ",\"gripper_width_m\":";
  if (std::isfinite(gripper_width_m)) *output << gripper_width_m;
  else *output << "null";
  *output
          << ",\"q_rad\":" << json_vector(feedback.joints)
          << ",\"qd_rad_s\":" << json_vector(velocity)
          << ",\"q_des_rad\":" << json_vector(desired.position)
          << ",\"qd_des_rad_s\":" << json_vector(desired.velocity)
          << ",\"qdd_des_rad_s2\":" << json_vector(desired.acceleration)
          << ",\"gravity_nm\":" << json_vector(dynamics.gravity)
          << ",\"coriolis_nm\":" << json_vector(dynamics.coriolis)
          << ",\"mass_diag_kg_m2\":" << json_vector(dynamics.mass.diagonal())
          << ",\"tau_ff_nm\":" << json_vector(feedforward_torque)
          << ",\"kp_nm_rad\":" << json_vector(kp)
          << ",\"kd_nm_s_rad\":" << json_vector(kd)
          // Firmware MIT torque reference: t_ff + kp(q_des-q) + kd(qd_des-qd).
          << ",\"tau_mit_equivalent_nm\":" << json_vector(
              feedforward_torque + kp.cwiseProduct(desired.position - feedback.joints) +
              kd.cwiseProduct(desired.velocity - velocity)) << "}\n";
}

#ifdef XROBOT_CPP_WITH_XR
void write_xr_ik_target_event(std::ofstream* output, std::uint64_t monotonic_ns,
                              const xrobot_cpp::XrFrame& frame,
                              const xrobot_cpp::Pose& requested,
                              const xrobot_cpp::ContinuousIkResult& result,
                              double solve_ms) {
  if (output == nullptr || !*output) return;
  *output << "{\"event\":\"xr_ik_target\",\"monotonic_ns\":" << monotonic_ns
          << ",\"xr_sequence\":" << frame.sequence
          << ",\"controller_orientation_xyzw\":" << json_quaternion(frame.controller.orientation)
          << ",\"tcp_requested_orientation_xyzw\":" << json_quaternion(requested.orientation)
          << ",\"q_goal_rad\":" << json_vector(result.joints)
          << ",\"position_error_m\":" << result.position_error_m
          << ",\"orientation_error_rad\":" << result.orientation_error_rad
          << ",\"solve_ms\":" << solve_ms << "}\n";

}

void write_xr_gripper_command_event(std::ofstream* output, std::uint64_t monotonic_ns,
                                    const xrobot_cpp::XrFrame& frame,
                                    double target_width_m, double force_n) {
  if (output == nullptr || !*output) return;
  *output << "{\"event\":\"xr_gripper_command\",\"monotonic_ns\":" << monotonic_ns
          << ",\"xr_sequence\":" << frame.sequence
          << ",\"trigger\":" << frame.trigger
          << ",\"grip\":" << frame.grip
          << ",\"target_width_m\":" << target_width_m
          << ",\"force_n\":" << force_n << "}\n";

}

void write_gravity_trim_watchdog_event(std::ofstream* output, std::uint64_t monotonic_ns,
                                       const xrobot_cpp::NeroFeedback& feedback,
                                       const xrobot_cpp::DynamicsTerms& dynamics,
                                       const xrobot_cpp::Vec7& trim_nm,
                                       double saturation_duration_s) {
  if (output == nullptr || !*output) return;
  *output << "{\"event\":\"gravity_trim_watchdog\",\"joint\":2"
          << ",\"monotonic_ns\":" << monotonic_ns
          << ",\"saturation_duration_s\":" << saturation_duration_s
          << ",\"q_rad\":" << json_vector(feedback.joints)
          << ",\"gravity_nm\":" << json_vector(dynamics.gravity)
          << ",\"gravity_trim_nm\":" << json_vector(trim_nm) << "}\n";
  output->flush();
}
#endif

// This is intentionally an offline diagnostic. Encoder motion alone cannot
// establish traceable motor torque; it estimates the effective correction that
// best explains a recorded MIT trajectory using the current URDF dynamics.
std::optional<xrobot_cpp::Vec7> json_vec7(const std::string& line, const std::string& key) {
  const std::string label = "\"" + key + "\":[";
  const std::size_t begin = line.find(label);
  if (begin == std::string::npos) return std::nullopt;
  const std::size_t values_begin = begin + label.size();
  const std::size_t values_end = line.find(']', values_begin);
  if (values_end == std::string::npos) return std::nullopt;
  std::stringstream values(line.substr(values_begin, values_end - values_begin));
  xrobot_cpp::Vec7 result;
  std::string value;
  Eigen::Index index = 0;
  while (std::getline(values, value, ',')) {
    if (index >= result.size()) return std::nullopt;
    try {
      result[index++] = std::stod(value);
    } catch (const std::exception&) {
      return std::nullopt;
    }
  }
  if (index != result.size() || !result.allFinite()) return std::nullopt;
  return result;
}

std::optional<std::uint64_t> json_uint64(const std::string& line, const std::string& key) {
  const std::string label = "\"" + key + "\":";
  const std::size_t begin = line.find(label);
  if (begin == std::string::npos) return std::nullopt;
  const std::size_t value_begin = begin + label.size();
  const std::size_t value_end = line.find_first_not_of("0123456789", value_begin);
  try {
    return static_cast<std::uint64_t>(std::stoull(line.substr(value_begin, value_end - value_begin)));
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

struct EffectiveTorqueSample {
  std::uint64_t monotonic_ns{};
  xrobot_cpp::Vec7 q{xrobot_cpp::Vec7::Zero()};
  xrobot_cpp::Vec7 qd{xrobot_cpp::Vec7::Zero()};
  xrobot_cpp::Vec7 tau{xrobot_cpp::Vec7::Zero()};
};

int run_mit_telemetry_summary(int argc, char** argv) {
  std::filesystem::path input_path;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--input" && index + 1 < argc) input_path = argv[++index];
    else throw std::invalid_argument("usage: --mit-telemetry-summary --input MIT_TELEMETRY.jsonl");
  }
  if (input_path.empty()) throw std::invalid_argument("--input is required");
  std::ifstream input(input_path);
  if (!input) throw std::runtime_error("cannot open MIT telemetry: " + input_path.string());
  std::optional<xrobot_cpp::Vec7> first_q, last_q, last_q_des;
  std::size_t samples = 0;
  double max_velocity = 0.0, max_equivalent_torque = 0.0;
  std::string line;
  while (std::getline(input, line)) {
    const auto q = json_vec7(line, "q_rad");
    const auto q_des = json_vec7(line, "q_des_rad");
    const auto qd = json_vec7(line, "qd_rad_s");
    const auto tau = json_vec7(line, "tau_mit_equivalent_nm");
    if (!q || !q_des || !qd || !tau) continue;
    if (!first_q) first_q = *q;
    last_q = *q;
    last_q_des = *q_des;
    max_velocity = std::max(max_velocity, qd->cwiseAbs().maxCoeff());
    max_equivalent_torque = std::max(max_equivalent_torque, tau->cwiseAbs().maxCoeff());
    ++samples;
  }
  if (!first_q || !last_q || !last_q_des) throw std::runtime_error("telemetry contains no complete MIT samples");
  const std::string urdf = std::string(XROBOT_CPP_SOURCE_DIR) + "/assets/urdf/nero_control_tcp.urdf";
  xrobot_cpp::NeroDynamicsModel model(urdf, "base_link", "gripper_tcp");
  const xrobot_cpp::Pose actual = model.forward_kinematics(*last_q);
  const xrobot_cpp::Pose desired = model.forward_kinematics(*last_q_des);
  const double tcp_position_error_m = (desired.position - actual.position).norm();
  const double tcp_orientation_error_rad = Eigen::AngleAxisd(desired.orientation * actual.orientation.conjugate()).angle();
  std::cout << "MIT telemetry summary: samples=" << samples
            << " final_joint_tracking_error_rad=" << (*last_q_des - *last_q).norm()
            << " final_tcp_position_error_m=" << tcp_position_error_m
            << " final_tcp_orientation_error_rad=" << tcp_orientation_error_rad
            << " max_velocity_rad_s=" << max_velocity
            << " max_equivalent_torque_nm=" << max_equivalent_torque << "\n";
  return EXIT_SUCCESS;
}

int run_effective_torque_identify(int argc, char** argv) {
  std::vector<std::filesystem::path> input_paths;
  std::filesystem::path output_path{"configs/nero_effective_torque_calibration.json"};
  std::size_t minimum_samples = 100;
  std::optional<Eigen::Index> requested_joint;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--input" && index + 1 < argc) input_paths.emplace_back(argv[++index]);
    else if (argument == "--output" && index + 1 < argc) output_path = argv[++index];
    else if (argument == "--min-samples" && index + 1 < argc) minimum_samples = static_cast<std::size_t>(std::stoul(argv[++index]));
    else if (argument == "--joint" && index + 1 < argc) {
      const int joint = std::stoi(argv[++index]);
      if (joint < 1 || joint > 7) throw std::invalid_argument("--joint must be in [1, 7]");
      requested_joint = joint - 1;
    } else {
      throw std::invalid_argument("usage: --effective-torque-identify --input MIT_TELEMETRY.jsonl [--input MORE.jsonl] [--joint 1..7] [--output CONFIG.json] [--min-samples N]");
    }
  }
  if (input_paths.empty()) throw std::invalid_argument("at least one --input is required");
  if (minimum_samples < 20 || minimum_samples > 1000000) throw std::invalid_argument("--min-samples must be in [20, 1000000]");
  std::vector<EffectiveTorqueSample> samples;
  for (const auto& input_path : input_paths) {
    std::ifstream input(input_path);
    if (!input) throw std::runtime_error("cannot open MIT telemetry: " + input_path.string());
    std::string line;
    while (std::getline(input, line)) {
      const auto timestamp = json_uint64(line, "monotonic_ns");
      const auto q = json_vec7(line, "q_rad");
      const auto qd = json_vec7(line, "qd_rad_s");
      const auto tau_equivalent = json_vec7(line, "tau_mit_equivalent_nm");
      const auto tau = tau_equivalent ? tau_equivalent : json_vec7(line, "tau_ff_nm");
      if (!timestamp || !q || !qd || !tau) continue;
      samples.push_back({*timestamp, *q, *qd, *tau});
    }
  }
  if (samples.size() < minimum_samples) {
    throw std::runtime_error("insufficient MIT samples: got " + std::to_string(samples.size()) + ", need at least " + std::to_string(minimum_samples));
  }

  const std::string urdf = std::string(XROBOT_CPP_SOURCE_DIR) + "/assets/urdf/nero_control_tcp.urdf";
  xrobot_cpp::NeroDynamicsModel model(urdf, "base_link", "gripper_tcp");
  xrobot_cpp::Vec7 torque_scale = xrobot_cpp::Vec7::Ones();
  xrobot_cpp::Vec7 gravity_scale = xrobot_cpp::Vec7::Ones();
  xrobot_cpp::Vec7 gravity_command_scale = xrobot_cpp::Vec7::Ones();
  xrobot_cpp::Vec7 candidate_torque_scale = xrobot_cpp::Vec7::Zero();
  xrobot_cpp::Vec7 candidate_gravity_scale = xrobot_cpp::Vec7::Zero();
  xrobot_cpp::Vec7 candidate_gravity_command_scale = xrobot_cpp::Vec7::Zero();
  xrobot_cpp::Vec7 bias_nm = xrobot_cpp::Vec7::Zero();
  xrobot_cpp::Vec7 viscous_nm_s_rad = xrobot_cpp::Vec7::Zero();
  xrobot_cpp::Vec7 coulomb_nm = xrobot_cpp::Vec7::Zero();
  xrobot_cpp::Vec7 condition_number = xrobot_cpp::Vec7::Zero();
  xrobot_cpp::Vec7 rms_residual_nm = xrobot_cpp::Vec7::Zero();
  xrobot_cpp::Vec7 q_span_rad = xrobot_cpp::Vec7::Zero();
  xrobot_cpp::Vec7 qd_peak_rad_s = xrobot_cpp::Vec7::Zero();
  xrobot_cpp::Vec7 tau_span_nm = xrobot_cpp::Vec7::Zero();
  xrobot_cpp::Vec7 gravity_span_nm = xrobot_cpp::Vec7::Zero();
  std::array<bool, 7> identifiable{};
  std::array<bool, 7> fit_solved{};
  std::array<bool, 7> local_response{};
  std::array<bool, 7> gravity_excitation{};
  std::size_t accepted = 0;

  for (Eigen::Index joint = 0; joint < 7; ++joint) {
    if (requested_joint && joint != *requested_joint) continue;
    std::vector<Eigen::Matrix<double, 1, 5>> rows;
    std::vector<double> observations;
    double q_min = std::numeric_limits<double>::infinity();
    double q_max = -std::numeric_limits<double>::infinity();
    double tau_min = std::numeric_limits<double>::infinity();
    double tau_max = -std::numeric_limits<double>::infinity();
    double gravity_min = std::numeric_limits<double>::infinity();
    double gravity_max = -std::numeric_limits<double>::infinity();
    double qd_peak = 0.0;
    for (std::size_t sample_index = 1; sample_index < samples.size(); ++sample_index) {
      const auto& previous = samples[sample_index - 1];
      const auto& current = samples[sample_index];
      if (current.monotonic_ns <= previous.monotonic_ns) continue;
      const double dt = static_cast<double>(current.monotonic_ns - previous.monotonic_ns) * 1e-9;
      if (dt < 0.004 || dt > 0.040) continue;
      const xrobot_cpp::Vec7 qdd = (current.qd - previous.qd) / dt;
      if (!qdd.allFinite() || qdd.cwiseAbs().maxCoeff() > 20.0) continue;
      const xrobot_cpp::DynamicsTerms terms = model.dynamics(current.q, current.qd);
      // M qdd + C = alpha*tau_ff - gamma*G - bias - fv*qd - fc*tanh(qd/vs).
      Eigen::Matrix<double, 1, 5> row;
      row << current.tau[joint], -terms.gravity[joint], -1.0, -current.qd[joint], -std::tanh(current.qd[joint] / 0.03);
      const double observation = (terms.mass * qdd + terms.coriolis)[joint];
      if (!row.allFinite() || !std::isfinite(observation)) continue;
      rows.push_back(row);
      observations.push_back(observation);
      q_min = std::min(q_min, current.q[joint]);
      q_max = std::max(q_max, current.q[joint]);
      tau_min = std::min(tau_min, current.tau[joint]);
      tau_max = std::max(tau_max, current.tau[joint]);
      gravity_min = std::min(gravity_min, terms.gravity[joint]);
      gravity_max = std::max(gravity_max, terms.gravity[joint]);
      qd_peak = std::max(qd_peak, std::abs(current.qd[joint]));
    }
    if (rows.empty()) continue;
    q_span_rad[joint] = q_max - q_min;
    qd_peak_rad_s[joint] = qd_peak;
    tau_span_nm[joint] = tau_max - tau_min;
    gravity_span_nm[joint] = gravity_max - gravity_min;
    local_response[joint] = rows.size() >= minimum_samples && q_span_rad[joint] >= 0.001 &&
                            qd_peak_rad_s[joint] >= 0.005 && tau_span_nm[joint] >= 0.10;
    // At one near-static pose, G(q) is almost a constant.  Its scale is then
    // inseparable from bias and torque scale, so require multiple gravity arms.
    gravity_excitation[joint] = gravity_span_nm[joint] >= 0.20;
    if (!local_response[joint] || !gravity_excitation[joint]) continue;
    Eigen::MatrixXd design(rows.size(), 5);
    Eigen::VectorXd target(rows.size());
    for (std::size_t row_index = 0; row_index < rows.size(); ++row_index) {
      design.row(static_cast<Eigen::Index>(row_index)) = rows[row_index];
      target[static_cast<Eigen::Index>(row_index)] = observations[row_index];
    }
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(design, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd singular = svd.singularValues();
    if (singular.size() != 5 || singular[4] < 1e-5) continue;
    condition_number[joint] = singular[0] / singular[4];
    if (!std::isfinite(condition_number[joint]) || condition_number[joint] > 1e5) continue;
    const Eigen::VectorXd coefficients = svd.solve(target);
    const Eigen::VectorXd residual = design * coefficients - target;
    rms_residual_nm[joint] = std::sqrt(residual.squaredNorm() / static_cast<double>(residual.size()));
    if (!coefficients.allFinite()) continue;
    fit_solved[joint] = true;
    candidate_torque_scale[joint] = coefficients[0];
    candidate_gravity_scale[joint] = coefficients[1];
    candidate_gravity_command_scale[joint] = coefficients[1] / coefficients[0];
    if (std::abs(coefficients[0]) < 0.05 || rms_residual_nm[joint] > 3.0) continue;
    torque_scale[joint] = coefficients[0];
    gravity_scale[joint] = coefficients[1];
    gravity_command_scale[joint] = coefficients[1] / coefficients[0];
    bias_nm[joint] = coefficients[2];
    viscous_nm_s_rad[joint] = coefficients[3];
    coulomb_nm[joint] = coefficients[4];
    identifiable[joint] = std::isfinite(gravity_command_scale[joint]) && gravity_command_scale[joint] >= 0.25 && gravity_command_scale[joint] <= 4.0;
    accepted += identifiable[joint] ? 1U : 0U;
  }

  if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
  std::ofstream output(output_path);
  if (!output) throw std::runtime_error("cannot write effective torque calibration: " + output_path.string());
  output << std::setprecision(12)
         << "{\n  \"schema\": \"nero_effective_torque_model_v1\",\n"
         << "  \"source\": \"encoder-only MIT system identification; not an absolute torque calibration\",\n"
         << "  \"input_count\": " << input_paths.size() << ",\n"
         << "  \"requested_joint\": " << (requested_joint ? std::to_string(*requested_joint + 1) : "null") << ",\n"
         << "  \"samples\": " << samples.size() << ",\n"
         << "  \"identifiable_joints\": " << accepted << ",\n"
         << "  \"usable_for_gravity_compensation\": "
         << ((requested_joint ? identifiable[*requested_joint] : accepted == 7) ? "true" : "false") << ",\n"
         << "  \"torque_scale\": " << json_vector(torque_scale) << ",\n"
         << "  \"gravity_model_scale\": " << json_vector(gravity_scale) << ",\n"
         << "  \"gravity_command_scale\": " << json_vector(gravity_command_scale) << ",\n"
         << "  \"candidate_torque_scale\": " << json_vector(candidate_torque_scale) << ",\n"
         << "  \"candidate_gravity_model_scale\": " << json_vector(candidate_gravity_scale) << ",\n"
         << "  \"candidate_gravity_command_scale\": " << json_vector(candidate_gravity_command_scale) << ",\n"
         << "  \"bias_nm\": " << json_vector(bias_nm) << ",\n"
         << "  \"viscous_nm_s_rad\": " << json_vector(viscous_nm_s_rad) << ",\n"
         << "  \"coulomb_nm\": " << json_vector(coulomb_nm) << ",\n"
         << "  \"condition_number\": " << json_vector(condition_number) << ",\n"
         << "  \"rms_residual_nm\": " << json_vector(rms_residual_nm) << ",\n"
         << "  \"q_span_rad\": " << json_vector(q_span_rad) << ",\n"
         << "  \"qd_peak_rad_s\": " << json_vector(qd_peak_rad_s) << ",\n"
         << "  \"tau_span_nm\": " << json_vector(tau_span_nm) << ",\n"
         << "  \"gravity_span_nm\": " << json_vector(gravity_span_nm) << ",\n"
         << "  \"local_response\": [";
  for (std::size_t index = 0; index < local_response.size(); ++index) output << (index ? "," : "") << (local_response[index] ? "true" : "false");
  output << "],\n  \"gravity_excitation\": [";
  for (std::size_t index = 0; index < gravity_excitation.size(); ++index) output << (index ? "," : "") << (gravity_excitation[index] ? "true" : "false");
  output << "],\n  \"fit_solved\": [";
  for (std::size_t index = 0; index < fit_solved.size(); ++index) output << (index ? "," : "") << (fit_solved[index] ? "true" : "false");
  output << "],\n  \"identifiable\": [";
  for (std::size_t index = 0; index < identifiable.size(); ++index) output << (index ? "," : "") << (identifiable[index] ? "true" : "false");
  output << "]\n}\n";
  std::cout << "Effective torque-model identification completed: samples=" << samples.size()
            << " identifiable_joints=" << accepted << "/7 output=" << output_path.string() << "\n"
            << "gravity_command_scale=" << gravity_command_scale.transpose() << "\n";
  const bool usable = requested_joint ? identifiable[*requested_joint] : accepted == 7;
  if (!usable) {
    if (requested_joint && local_response[*requested_joint] && !gravity_excitation[*requested_joint]) {
      std::cout << "LOCAL RESPONSE CONFIRMED for J" << (*requested_joint + 1)
                << ": collect the same probe at two or more safe poses with a total G(q) span of at least 0.20Nm before fitting gravity scale.\n";
    } else {
      std::cout << "RESULT NOT USABLE: the log lacks sufficient independent excitation for the requested gravity-model fit.\n";
    }
    return EXIT_FAILURE;
  }
  std::cout << "RESULT IS A SOFTWARE EFFECTIVE MODEL ONLY. Do not treat it as a traceable Nm calibration.\n";
  return EXIT_SUCCESS;
}
}
#ifdef XROBOT_CPP_WITH_MUJOCO
#include "xrobot_cpp/mujoco_simulation.hpp"
#endif
#ifdef XROBOT_CPP_WITH_MUJOCO_VIEWER
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#endif

#ifdef XROBOT_CPP_WITH_MUJOCO
int run_mujoco_dry_run(int argc, char** argv) {
  double seconds = 3.0;
  double gripper_width_m = 0.060;
  double payload_mass_kg = 0.0;
  std::string model_path = std::string(XROBOT_CPP_SOURCE_DIR) + "/assets/mujoco/nero_torque.xml";
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--seconds" && index + 1 < argc) seconds = std::stod(argv[++index]);
    else if (argument == "--gripper-width-m" && index + 1 < argc) gripper_width_m = std::stod(argv[++index]);
    else if (argument == "--model" && index + 1 < argc) model_path = argv[++index];
    else if (argument == "--payload-mass-kg" && index + 1 < argc) payload_mass_kg = std::stod(argv[++index]);
    else throw std::invalid_argument("usage: --mujoco-dry-run [--seconds S] [--gripper-width-m M]|--mujoco-benchmark [--seconds S] [--payload-masses-kg 0,0.2,0.5] [--output PATH]|--mujoco-xr-teleop [--payload-mass-kg KG] [--model PATH]");
  }
  if (!std::isfinite(seconds) || seconds <= 0.0 || seconds > 60.0) throw std::invalid_argument("--seconds must be in (0, 60]");
  if (!std::isfinite(gripper_width_m) || gripper_width_m < 0.0 || gripper_width_m > 0.100) throw std::invalid_argument("--gripper-width-m must be in [0, 0.100]");
  if (!std::isfinite(payload_mass_kg) || payload_mass_kg < 0.0 || payload_mass_kg > 3.0) throw std::invalid_argument("--payload-mass-kg must be in [0, 3]");
  const std::string urdf = std::string(XROBOT_CPP_SOURCE_DIR) + "/../xrobot/third_party/agx_arm_urdf/nero/urdf/nero_description.urdf";
  if (!std::filesystem::exists(model_path) || !std::filesystem::exists(urdf)) throw std::runtime_error("MuJoCo model or NERO URDF is missing");
  xrobot_cpp::NeroDynamicsModel dynamics(urdf);
  xrobot_cpp::NeroMujocoSimulation simulation(model_path);
  const xrobot_cpp::Vec7 initial = (xrobot_cpp::Vec7() << 0.0, -0.20, 0.30, 0.20, -0.10, 0.10, 0.0).finished();
  const xrobot_cpp::Vec7 reachable_goal = initial + (xrobot_cpp::Vec7() << 0.08, -0.06, 0.04, 0.03, -0.03, 0.02, 0.04).finished();
  const xrobot_cpp::Pose tcp_target = dynamics.forward_kinematics(reachable_goal);
  const xrobot_cpp::Vec7 target = dynamics.solve_ik(tcp_target, initial);
  const xrobot_cpp::JointTrajectoryState desired{.position = target, .velocity = xrobot_cpp::Vec7::Zero(), .acceleration = xrobot_cpp::Vec7::Zero()};
  xrobot_cpp::ComputedTorqueController controller(xrobot_cpp::Vec7::Constant(kNeroLowSpeedSpringStiffnessNmRad), xrobot_cpp::Vec7::Constant(kNeroLowSpeedDampingNmSRad), (xrobot_cpp::Vec7() << 6.0, 6.0, 5.0, 5.0, 4.0, 3.0, 2.0).finished(), xrobot_cpp::JointDriveModel::mujoco_default());
  simulation.reset(initial);
  simulation.set_payload_mass_kg(payload_mass_kg);
  simulation.set_gripper_width(gripper_width_m);
  const std::size_t steps = static_cast<std::size_t>(std::llround(seconds * 1000.0));
  xrobot_cpp::ComputedTorqueCommand command;
  for (std::size_t step = 0; step < steps; ++step) {
    command = controller.compute(simulation.dynamics_terms(), simulation.joint_position(), simulation.joint_velocity(), desired);
    simulation.set_joint_torque(command.feedforward_torque_nm);
    simulation.step();
  }
  const xrobot_cpp::Vec7 final_position = simulation.joint_position();
  std::cout << "MuJoCo NERO dynamics run completed. No SocketCAN interface was opened.\n";
  std::cout << "model=" << model_path << " sim_time_s=" << simulation.time_s() << "\n";
  std::cout << "q_target=[" << target.transpose() << "]\n";
  std::cout << "q_final=[" << final_position.transpose() << "]\n";
  std::cout << "q_error_norm_rad=" << (target - final_position).norm() << "\n";
  std::cout << "tau_final_nm=[" << command.feedforward_torque_nm.transpose() << "]\n";
  std::cout << "gripper_width_m=" << simulation.gripper_width_m() << "\n";
  std::cout << "payload_mass_kg=" << payload_mass_kg << "\n";
  return EXIT_SUCCESS;
}

int run_mujoco_smooth_torque_test(int argc, char** argv) {
  double seconds = 2.0;
  double payload_mass_kg = 0.20;
  double spring_stiffness_nm_rad = 2.0, damping_nm_s_rad = 0.25, torque_limit_nm = 0.35;
  std::filesystem::path output_path{"results/mujoco_smooth_torque.json"};
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--seconds" && index + 1 < argc) seconds = std::stod(argv[++index]);
    else if (argument == "--payload-mass-kg" && index + 1 < argc) payload_mass_kg = std::stod(argv[++index]);
    else if (argument == "--spring-stiffness" && index + 1 < argc) spring_stiffness_nm_rad = std::stod(argv[++index]);
    else if (argument == "--damping" && index + 1 < argc) damping_nm_s_rad = std::stod(argv[++index]);
    else if (argument == "--torque-limit" && index + 1 < argc) torque_limit_nm = std::stod(argv[++index]);
    else if (argument == "--output" && index + 1 < argc) output_path = argv[++index];
    else throw std::invalid_argument("usage: --mujoco-smooth-torque-test [--seconds S] [--payload-mass-kg KG] [--output PATH]");
  }
  if (!std::isfinite(seconds) || seconds < 1.0 || seconds > 10.0) throw std::invalid_argument("--seconds must be in [1, 10]");
  if (!std::isfinite(payload_mass_kg) || payload_mass_kg < 0.0 || payload_mass_kg > 3.0) throw std::invalid_argument("--payload-mass-kg must be in [0, 3]");
  if (!std::isfinite(spring_stiffness_nm_rad) || spring_stiffness_nm_rad <= 0.0 || spring_stiffness_nm_rad > 20.0) throw std::invalid_argument("--spring-stiffness must be in (0, 20]");
  if (!std::isfinite(damping_nm_s_rad) || damping_nm_s_rad <= 0.0 || damping_nm_s_rad > 5.0) throw std::invalid_argument("--damping must be in (0, 5]");
  if (!std::isfinite(torque_limit_nm) || torque_limit_nm <= 0.0 || torque_limit_nm > 2.0) throw std::invalid_argument("--torque-limit must be in (0, 2]");
  if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
  const std::string root(XROBOT_CPP_SOURCE_DIR);
  xrobot_cpp::NeroDynamicsModel dynamics(root + "/assets/urdf/nero_control_tcp.urdf", "base_link", "gripper_tcp");
#ifdef XROBOT_CPP_WITH_PLACO
  xrobot_cpp::PlacoNeroIkSolver placo_ik(root + "/assets/urdf/nero_control_tcp.urdf", "gripper_tcp");
#endif
  xrobot_cpp::NeroMujocoSimulation simulation(root + "/assets/mujoco/nero_torque.xml");
  const xrobot_cpp::Vec7 initial = (xrobot_cpp::Vec7() << 0.0, -0.20, 0.30, 0.20, -0.10, 0.10, 0.0).finished();
  const xrobot_cpp::Vec7 goal = initial + (xrobot_cpp::Vec7() << 0.020, -0.015, 0.010, 0.008, -0.006, 0.005, 0.004).finished();
  const xrobot_cpp::Vec7 torque_limit = xrobot_cpp::Vec7::Constant(torque_limit_nm);
  const xrobot_cpp::Vec7 max_torque_delta = xrobot_cpp::Vec7::Constant(0.03);
  xrobot_cpp::ComputedTorqueController controller(xrobot_cpp::Vec7::Constant(spring_stiffness_nm_rad), xrobot_cpp::Vec7::Constant(damping_nm_s_rad), torque_limit, xrobot_cpp::JointDriveModel::mujoco_default());
  xrobot_cpp::JerkLimitedJointPlanner trajectory(xrobot_cpp::Vec7::Constant(0.20), xrobot_cpp::Vec7::Constant(0.50), xrobot_cpp::Vec7::Constant(5.0));
  simulation.reset(initial);
  simulation.set_payload_mass_kg(payload_mass_kg);
  trajectory.reset(initial);
  trajectory.set_goal(goal);
  xrobot_cpp::JointTrajectoryState desired = trajectory.state();
  xrobot_cpp::Vec7 applied_torque = xrobot_cpp::Vec7::Zero();
  double peak_torque_nm = 0.0, peak_delta_nm = 0.0;
  std::size_t limit_hits = 0;
  const std::size_t steps = static_cast<std::size_t>(std::llround(seconds * 1000.0));
  for (std::size_t step = 0; step < steps; ++step) {
    if (step % 10 == 0) {
      desired = trajectory.step(0.010);
      xrobot_cpp::JointTrajectoryState impedance_desired = desired;
      impedance_desired.acceleration.setZero();
      const xrobot_cpp::ComputedTorqueCommand raw = controller.compute(simulation.dynamics_terms(), simulation.joint_position(), simulation.joint_velocity(), impedance_desired);
      const xrobot_cpp::Vec7 next_torque = applied_torque + (raw.feedforward_torque_nm - applied_torque).cwiseMax(-max_torque_delta).cwiseMin(max_torque_delta);
      peak_delta_nm = std::max(peak_delta_nm, (next_torque - applied_torque).cwiseAbs().maxCoeff());
      applied_torque = next_torque;
    }
    peak_torque_nm = std::max(peak_torque_nm, applied_torque.cwiseAbs().maxCoeff());
    simulation.set_joint_torque(applied_torque);
    simulation.step();
    const xrobot_cpp::Vec7 q = simulation.joint_position();
    for (int joint = 0; joint < 7; ++joint) if (q[joint] < dynamics.lower_limits()[joint] - 1e-6 || q[joint] > dynamics.upper_limits()[joint] + 1e-6) ++limit_hits;
  }
  const double final_error_rad = (desired.position - simulation.joint_position()).norm();
  const bool passed = peak_torque_nm <= torque_limit_nm + 1e-6 && peak_delta_nm <= 0.030001 && final_error_rad <= 0.20 && limit_hits == 0;
  std::ofstream output(output_path);
  if (!output) throw std::runtime_error("cannot open smooth-torque report: " + output_path.string());
  output << "{\"passed\":" << (passed ? "true" : "false")
         << ",\"spring_stiffness_nm_rad\":" << spring_stiffness_nm_rad << ",\"damping_nm_s_rad\":" << damping_nm_s_rad
         << ",\"inertial_feedforward_enabled\":false"
         << ",\"torque_limit_nm\":" << torque_limit_nm << ",\"torque_slew_nm_per_10ms\":0.03"
         << ",\"peak_torque_nm\":" << peak_torque_nm
         << ",\"peak_torque_delta_nm\":" << peak_delta_nm
         << ",\"final_joint_error_rad\":" << final_error_rad
         << ",\"limit_hits\":" << limit_hits << "}\n";
  std::cout << "MuJoCo smooth-torque test " << (passed ? "PASSED" : "FAILED")
            << ": final_error=" << final_error_rad << "rad peak_tau=" << peak_torque_nm
            << "Nm peak_delta=" << peak_delta_nm << "Nm limit_hits=" << limit_hits
            << " report=" << output_path.string() << "\n";
  std::cout << "No SocketCAN interface was opened.\n";
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

std::vector<double> parse_benchmark_payloads(std::string_view text) {
  std::vector<double> values;
  std::stringstream stream{std::string(text)};
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) throw std::invalid_argument("--payload-masses-kg contains an empty value");
    const double value = std::stod(token);
    if (!std::isfinite(value) || value < 0.0 || value > 3.0) throw std::invalid_argument("each payload mass must be in [0, 3] kg");
    values.push_back(value);
  }
  if (values.empty()) throw std::invalid_argument("--payload-masses-kg must not be empty");
  return values;
}

int run_mujoco_benchmark(int argc, char** argv) {
  double seconds = 3.0;
  double gripper_width_m = 0.060;
  std::vector<double> payload_masses{0.0, 0.20, 0.50};
  std::filesystem::path output_path{"results/nero_mujoco_benchmark.csv"};
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--seconds" && index + 1 < argc) seconds = std::stod(argv[++index]);
    else if (argument == "--gripper-width-m" && index + 1 < argc) gripper_width_m = std::stod(argv[++index]);
    else if (argument == "--payload-masses-kg" && index + 1 < argc) payload_masses = parse_benchmark_payloads(argv[++index]);
    else if (argument == "--output" && index + 1 < argc) output_path = argv[++index];
    else throw std::invalid_argument("usage: --mujoco-benchmark [--seconds S] [--gripper-width-m M] [--payload-masses-kg 0,0.2,0.5] [--output PATH]");
  }
  if (!std::isfinite(seconds) || seconds <= 0.0 || seconds > 60.0) throw std::invalid_argument("--seconds must be in (0, 60]");
  if (!std::isfinite(gripper_width_m) || gripper_width_m < 0.0 || gripper_width_m > 0.100) throw std::invalid_argument("--gripper-width-m must be in [0, 0.100]");
  if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
  std::ofstream csv(output_path);
  if (!csv) throw std::runtime_error("cannot open benchmark output: " + output_path.string());
  csv << "scenario,payload_kg,sim_time_s,wall_time_s,real_time_factor,final_joint_error_norm_rad,rms_joint_error_norm_rad,peak_joint_error_rad,final_tcp_position_error_m,peak_torque_nm,torque_limit_hits\n";

  const std::string root(XROBOT_CPP_SOURCE_DIR);
  xrobot_cpp::NeroDynamicsModel dynamics(root + "/assets/urdf/nero_control_tcp.urdf", "base_link", "gripper_tcp");
  const std::string model_path = root + "/assets/mujoco/nero_torque.xml";
  const xrobot_cpp::Vec7 initial = (xrobot_cpp::Vec7() << 0.0, -0.20, 0.30, 0.20, -0.10, 0.10, 0.0).finished();
  const xrobot_cpp::Vec7 torque_limit = (xrobot_cpp::Vec7() << 6.0, 6.0, 5.0, 5.0, 4.0, 3.0, 2.0).finished();
  const xrobot_cpp::ComputedTorqueController controller(xrobot_cpp::Vec7::Constant(kNeroLowSpeedSpringStiffnessNmRad), xrobot_cpp::Vec7::Constant(kNeroLowSpeedDampingNmSRad), torque_limit, xrobot_cpp::JointDriveModel::mujoco_default());
  struct BenchmarkCase { const char* name; xrobot_cpp::Vec7 delta; };
  const std::vector<BenchmarkCase> cases{
      {"reach", (xrobot_cpp::Vec7() << 0.08, -0.06, 0.04, 0.03, -0.03, 0.02, 0.04).finished()},
      {"lift", (xrobot_cpp::Vec7() << -0.06, 0.10, -0.07, 0.05, 0.02, -0.03, 0.00).finished()},
      {"wrist", (xrobot_cpp::Vec7() << 0.02, -0.02, 0.02, 0.00, 0.10, 0.08, 0.20).finished()},
  };
  const std::size_t steps = static_cast<std::size_t>(std::llround(seconds * 1000.0));
  double worst_error = 0.0;
  std::size_t completed = 0;
  for (const BenchmarkCase& benchmark_case : cases) {
    const xrobot_cpp::Pose target_tcp = dynamics.forward_kinematics(initial + benchmark_case.delta);
    const xrobot_cpp::Vec7 target = dynamics.solve_ik(target_tcp, initial);
    const xrobot_cpp::JointTrajectoryState desired{.position = target, .velocity = xrobot_cpp::Vec7::Zero(), .acceleration = xrobot_cpp::Vec7::Zero()};
    for (const double payload_mass_kg : payload_masses) {
      xrobot_cpp::NeroMujocoSimulation simulation(model_path);
      simulation.reset(initial);
      simulation.set_payload_mass_kg(payload_mass_kg);
      simulation.set_gripper_width(gripper_width_m);
      double squared_error_sum = 0.0;
      double peak_joint_error = 0.0;
      double peak_torque = 0.0;
      std::size_t torque_limit_hits = 0;
      const auto wall_start = std::chrono::steady_clock::now();
      for (std::size_t step = 0; step < steps; ++step) {
        const xrobot_cpp::ComputedTorqueCommand command = controller.compute(simulation.dynamics_terms(), simulation.joint_position(), simulation.joint_velocity(), desired);
        const xrobot_cpp::Vec7 error = target - simulation.joint_position();
        squared_error_sum += error.squaredNorm();
        peak_joint_error = std::max(peak_joint_error, error.cwiseAbs().maxCoeff());
        peak_torque = std::max(peak_torque, command.feedforward_torque_nm.cwiseAbs().maxCoeff());
        torque_limit_hits += static_cast<std::size_t>((command.feedforward_torque_nm.cwiseAbs().array() >= (torque_limit.array() - 1e-9)).count());
        simulation.set_joint_torque(command.feedforward_torque_nm);
        simulation.step();
      }
      const double wall_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
      const xrobot_cpp::Vec7 final_error = target - simulation.joint_position();
      const double final_error_norm = final_error.norm();
      const double rms_error_norm = std::sqrt(squared_error_sum / static_cast<double>(steps));
      const double tcp_error_m = (dynamics.forward_kinematics(simulation.joint_position()).position - target_tcp.position).norm();
      const double real_time_factor = simulation.time_s() / std::max(wall_time_s, 1e-9);
      csv << benchmark_case.name << "," << payload_mass_kg << "," << simulation.time_s() << "," << wall_time_s << "," << real_time_factor << "," << final_error_norm << "," << rms_error_norm << "," << peak_joint_error << "," << tcp_error_m << "," << peak_torque << "," << torque_limit_hits << "\n";
      std::cout << "BENCH " << benchmark_case.name << " payload=" << payload_mass_kg << "kg final_error=" << final_error_norm << "rad tcp_error=" << tcp_error_m << "m peak_tau=" << peak_torque << "Nm rtf=" << real_time_factor << " limit_hits=" << torque_limit_hits << "\n";
      worst_error = std::max(worst_error, final_error_norm);
      ++completed;
    }
  }
  std::cout << "MuJoCo benchmark completed: " << completed << " runs, worst_final_joint_error=" << worst_error << "rad\n";
  std::cout << "csv=" << output_path.string() << "\n";
  std::cout << "No SocketCAN interface was opened.\n";
  return EXIT_SUCCESS;
}

int run_mujoco_gravity_impedance_hold(int argc, char** argv) {
  double seconds = 3.0, payload_mass_kg = 0.20, spring_stiffness_nm_rad = 2.0, damping_nm_s_rad = 0.40;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--seconds" && index + 1 < argc) seconds = std::stod(argv[++index]);
    else if (argument == "--payload-mass-kg" && index + 1 < argc) payload_mass_kg = std::stod(argv[++index]);
    else if (argument == "--spring-stiffness" && index + 1 < argc) spring_stiffness_nm_rad = std::stod(argv[++index]);
    else if (argument == "--damping" && index + 1 < argc) damping_nm_s_rad = std::stod(argv[++index]);
    else throw std::invalid_argument("usage: --mujoco-gravity-impedance-hold [--seconds S] [--payload-mass-kg KG] [--spring-stiffness K] [--damping D]");
  }
  if (!std::isfinite(seconds) || seconds <= 0.0 || seconds > 30.0) throw std::invalid_argument("--seconds must be in (0, 30]");
  if (!std::isfinite(payload_mass_kg) || payload_mass_kg < 0.0 || payload_mass_kg > 3.0) throw std::invalid_argument("--payload-mass-kg must be in [0, 3]");
  if (!std::isfinite(spring_stiffness_nm_rad) || spring_stiffness_nm_rad <= 0.0 || spring_stiffness_nm_rad > 20.0) throw std::invalid_argument("--spring-stiffness must be in (0, 20]");
  if (!std::isfinite(damping_nm_s_rad) || damping_nm_s_rad <= 0.0 || damping_nm_s_rad > 5.0) throw std::invalid_argument("--damping must be in (0, 5]");
  const std::string root(XROBOT_CPP_SOURCE_DIR);
  xrobot_cpp::NeroMujocoSimulation simulation(root + "/assets/mujoco/nero_torque.xml");
  // This is the documented J2 working pose used by the real supported-arm
  // collection.  Its nominal gravity demand remains inside the conservative
  // per-axis MIT safety envelope, unlike the old near-limit test posture.
  const xrobot_cpp::Vec7 initial = (xrobot_cpp::Vec7() << -1.58336, -0.440905, 0.181444, 0.352644,
                                           -0.343149, 0.0412072, 0.138998).finished();
  const xrobot_cpp::Vec7 torque_limit = mit_support_torque_limits();
  const xrobot_cpp::Vec7 stiffness = xrobot_cpp::Vec7::Constant(spring_stiffness_nm_rad);
  const xrobot_cpp::Vec7 damping = xrobot_cpp::Vec7::Constant(damping_nm_s_rad);
  simulation.reset(initial);
  simulation.set_payload_mass_kg(payload_mass_kg);
  const xrobot_cpp::JointTrajectoryState hold{.position = initial, .velocity = xrobot_cpp::Vec7::Zero(), .acceleration = xrobot_cpp::Vec7::Zero()};
  const xrobot_cpp::Vec7 initial_gravity = simulation.dynamics_terms().gravity;
  xrobot_cpp::Vec7 applied_torque = initial_gravity.cwiseMax(-torque_limit).cwiseMin(torque_limit);
  double peak_drift = 0.0, peak_velocity = 0.0;
  std::size_t saturation_hits = 0;
  const std::size_t steps = static_cast<std::size_t>(std::llround(seconds * 1000.0));
  for (std::size_t step = 0; step < steps; ++step) {
    if (step % 10 == 0) {
      const xrobot_cpp::Vec7 q = simulation.joint_position();
      const xrobot_cpp::Vec7 qd = simulation.joint_velocity();
      const xrobot_cpp::Vec7 gravity = nominal_gravity_feedforward(simulation.dynamics_terms(), torque_limit);
      applied_torque += (gravity - applied_torque).cwiseMax(-xrobot_cpp::Vec7::Constant(0.03)).cwiseMin(xrobot_cpp::Vec7::Constant(0.03));
      const xrobot_cpp::Vec7 total = applied_torque + stiffness.cwiseProduct(hold.position - q) + damping.cwiseProduct(-qd);
      saturation_hits += static_cast<std::size_t>((total.cwiseAbs().array() >= torque_limit.array()).count());
      simulation.set_joint_torque(total.cwiseMax(-torque_limit).cwiseMin(torque_limit));
    }
    simulation.step();
    peak_drift = std::max(peak_drift, (simulation.joint_position() - initial).cwiseAbs().maxCoeff());
    peak_velocity = std::max(peak_velocity, simulation.joint_velocity().cwiseAbs().maxCoeff());
  }
  const double final_drift = (simulation.joint_position() - initial).cwiseAbs().maxCoeff();
  const double final_velocity = simulation.joint_velocity().cwiseAbs().maxCoeff();
  const xrobot_cpp::Vec7 final_position = simulation.joint_position();
  const bool passed = final_drift <= 0.010 && final_velocity <= 0.020 && saturation_hits == 0;
  std::cout << "MuJoCo gravity/low-impedance hold " << (passed ? "PASSED" : "FAILED")
            << ": K=" << spring_stiffness_nm_rad << " D=" << damping_nm_s_rad << " payload=" << payload_mass_kg << "kg final_drift_rad=" << final_drift
            << " peak_drift_rad=" << peak_drift << " final_velocity_rad_s=" << final_velocity
            << " peak_velocity_rad_s=" << peak_velocity << " saturation_hits=" << saturation_hits << "\n";
  std::cout << "initial_gravity_nm=[" << initial_gravity.transpose() << "] final_q_rad=[" << final_position.transpose() << "]\n";
  std::cout << "No SocketCAN interface was opened.\n";
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

int run_mujoco_model_parity() {
  const std::string root(XROBOT_CPP_SOURCE_DIR);
  const std::string bare_urdf = root + "/../xrobot/third_party/agx_arm_urdf/nero/urdf/nero_description.urdf";
  const std::string control_urdf = root + "/assets/urdf/nero_control_tcp.urdf";
  xrobot_cpp::NeroDynamicsModel kdl(bare_urdf);
  xrobot_cpp::NeroDynamicsModel control_kdl(control_urdf, "base_link", "gripper_tcp");
  xrobot_cpp::NeroMujocoSimulation simulation(root + "/assets/mujoco/nero_torque.xml");
  const xrobot_cpp::Vec7 q = (xrobot_cpp::Vec7() << 1.957, -0.3484, -0.0519, -1.0122, -0.3668, -0.1102, 0.1953).finished();
  simulation.reset(q);
  simulation.set_payload_mass_kg(0.0);
  const xrobot_cpp::Pose kdl_tip = kdl.forward_kinematics(q);
  const xrobot_cpp::Pose mj_base = simulation.body_pose("base_link");
  const xrobot_cpp::Pose mj_tip = simulation.body_pose("link7");
  const xrobot_cpp::Vec3 position_error = (mj_tip.position - mj_base.position) - kdl_tip.position;
  const double orientation_error_rad = Eigen::AngleAxisd(kdl_tip.orientation.conjugate() * mj_tip.orientation).angle();
  const xrobot_cpp::Vec7 kdl_gravity = control_kdl.dynamics(q, xrobot_cpp::Vec7::Zero()).gravity;
  const xrobot_cpp::Vec7 mj_gravity = simulation.dynamics_terms().gravity;
  const double gravity_relative_error = (mj_gravity - kdl_gravity).norm() / std::max(kdl_gravity.norm(), 1e-9);
  std::cout << "MuJoCo/URDF parity at fixed q:\n"
            << "  fk_position_error_m=" << position_error.norm() << " fk_orientation_error_rad=" << orientation_error_rad << "\n"
            << "  kdl_tip_position_m=[" << kdl_tip.position.transpose() << "] mj_link7_relative_m=[" << (mj_tip.position - mj_base.position).transpose() << "]\n"
            << "  kdl_gravity_nm=[" << kdl_gravity.transpose() << "]\n"
            << "  mujoco_gravity_nm=[" << mj_gravity.transpose() << "]\n"
            << "  gravity_relative_error=" << gravity_relative_error << "\n";
  const bool passed = position_error.norm() <= 1e-5 && orientation_error_rad <= 1e-5 && gravity_relative_error <= 0.05;
  std::cout << "MuJoCo model parity " << (passed ? "PASSED" : "FAILED") << ". No SocketCAN interface was opened.\n";
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

int run_mujoco_impedance_convergence_benchmark(int argc, char** argv) {
  std::size_t trajectories = 1000;
  double seconds = 3.0, payload_mass_kg = 0.20;
  double spring_stiffness_nm_rad = 6.0, damping_nm_s_rad = 0.80, gravity_scale = 1.0, torque_limit_nm = 0.0;
  unsigned int seed = 20260807U;
  std::filesystem::path output_path{"results/nero_mujoco_impedance_convergence.jsonl"};
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--trajectories" && index + 1 < argc) trajectories = static_cast<std::size_t>(std::stoul(argv[++index]));
    else if (argument == "--seconds" && index + 1 < argc) seconds = std::stod(argv[++index]);
    else if (argument == "--payload-mass-kg" && index + 1 < argc) payload_mass_kg = std::stod(argv[++index]);
    else if (argument == "--spring-stiffness" && index + 1 < argc) spring_stiffness_nm_rad = std::stod(argv[++index]);
    else if (argument == "--damping" && index + 1 < argc) damping_nm_s_rad = std::stod(argv[++index]);
    else if (argument == "--gravity-scale" && index + 1 < argc) gravity_scale = std::stod(argv[++index]);
    else if (argument == "--torque-limit" && index + 1 < argc) torque_limit_nm = std::stod(argv[++index]);
    else if (argument == "--seed" && index + 1 < argc) seed = static_cast<unsigned int>(std::stoul(argv[++index]));
    else if (argument == "--output" && index + 1 < argc) output_path = argv[++index];
    else throw std::invalid_argument("usage: --mujoco-impedance-convergence [--trajectories N] [--seconds S] [--payload-mass-kg KG] [--spring-stiffness K] [--damping D] [--gravity-scale S] [--torque-limit NM] [--seed N] [--output PATH]");
  }
  if (trajectories == 0 || trajectories > 10000) throw std::invalid_argument("--trajectories must be in [1, 10000]");
  if (!std::isfinite(seconds) || seconds < 1.0 || seconds > 10.0) throw std::invalid_argument("--seconds must be in [1, 10]");
  if (!std::isfinite(payload_mass_kg) || payload_mass_kg < 0.0 || payload_mass_kg > 3.0) throw std::invalid_argument("--payload-mass-kg must be in [0, 3]");
  if (!std::isfinite(spring_stiffness_nm_rad) || spring_stiffness_nm_rad <= 0.0 || spring_stiffness_nm_rad > 20.0) throw std::invalid_argument("--spring-stiffness must be in (0, 20]");
  if (!std::isfinite(damping_nm_s_rad) || damping_nm_s_rad <= 0.0 || damping_nm_s_rad > 5.0) throw std::invalid_argument("--damping must be in (0, 5]");
  if (!std::isfinite(gravity_scale) || gravity_scale < 0.50 || gravity_scale > 1.20) throw std::invalid_argument("--gravity-scale must be in [0.50, 1.20]");
  if (!std::isfinite(torque_limit_nm) || torque_limit_nm < 0.0 || torque_limit_nm > 6.0) throw std::invalid_argument("--torque-limit must be 0 (per-axis support limits) or in (0, 6]");
  if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
  std::ofstream output(output_path);
  if (!output) throw std::runtime_error("cannot open convergence benchmark output: " + output_path.string());

  const std::string root(XROBOT_CPP_SOURCE_DIR);
  xrobot_cpp::NeroDynamicsModel dynamics(root + "/assets/urdf/nero_control_tcp.urdf", "base_link", "gripper_tcp");
  const std::string model_path = root + "/assets/mujoco/nero_torque.xml";
  const xrobot_cpp::Vec7 initial = (xrobot_cpp::Vec7() << 0.0, -0.20, 0.30, 0.20, -0.10, 0.10, 0.0).finished();
  const xrobot_cpp::Vec7 max_delta = (xrobot_cpp::Vec7() << 0.040, 0.035, 0.035, 0.030, 0.025, 0.025, 0.020).finished();
  const xrobot_cpp::Vec7 torque_limit = torque_limit_nm > 0.0
      ? xrobot_cpp::Vec7::Constant(torque_limit_nm)
      : mit_support_torque_limits();
  const xrobot_cpp::Vec7 max_torque_delta = xrobot_cpp::Vec7::Constant(0.03);
  const xrobot_cpp::Vec7 stiffness = xrobot_cpp::Vec7::Constant(spring_stiffness_nm_rad);
  const xrobot_cpp::Vec7 damping = xrobot_cpp::Vec7::Constant(damping_nm_s_rad);
  const xrobot_cpp::ComputedTorqueController controller(stiffness, damping, torque_limit, xrobot_cpp::JointDriveModel::mujoco_default());
  const std::size_t steps = static_cast<std::size_t>(std::llround(seconds * 1000.0));
  constexpr std::size_t kControlDivider = 10;
  constexpr std::size_t kSettlingWindowSteps = 200;
  constexpr double kNearTargetErrorRad = 0.020;
  constexpr double kNearTargetAccelerationRadS2 = 0.25;
  constexpr double kFinalAccelerationRadS2 = 0.10;
  constexpr double kFinalVelocityRadS = 0.030;
  constexpr double kGainRisePerControlCycle = 0.025;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  std::vector<double> final_errors, near_peak_accelerations, final_accelerations, final_velocities;
  std::size_t passed_count = 0, total_limit_hits = 0, total_saturation_hits = 0;

  output << std::setprecision(10);
  for (std::size_t trial = 0; trial < trajectories; ++trial) {
    xrobot_cpp::Vec7 target = initial;
    for (Eigen::Index joint = 0; joint < 7; ++joint) target[joint] += unit(rng) * max_delta[joint];
    xrobot_cpp::NeroMujocoSimulation simulation(model_path);
    simulation.reset(initial);
    simulation.set_payload_mass_kg(payload_mass_kg);
    xrobot_cpp::JerkLimitedJointPlanner trajectory(
        xrobot_cpp::Vec7::Constant(0.20), xrobot_cpp::Vec7::Constant(0.50), xrobot_cpp::Vec7::Constant(5.0));
    trajectory.reset(initial);
    trajectory.set_goal(target);
    xrobot_cpp::JointTrajectoryState desired = trajectory.state();
    xrobot_cpp::Vec7 applied_feedforward = (gravity_scale * dynamics.dynamics(initial, xrobot_cpp::Vec7::Zero()).gravity)
        .cwiseMax(-torque_limit).cwiseMin(torque_limit);
    xrobot_cpp::Vec7 applied_gain_scale = xrobot_cpp::Vec7::Zero();
    xrobot_cpp::Vec7 applied_total_torque = applied_feedforward;
    double peak_torque_nm = 0.0, peak_torque_delta_nm = 0.0, near_peak_acceleration_rad_s2 = 0.0;
    std::size_t near_samples = 0, limit_hits = 0, saturation_hits = 0;
    for (std::size_t step = 0; step < steps; ++step) {
      if (step % kControlDivider == 0) {
        desired = trajectory.step(0.010);
        const xrobot_cpp::Vec7 q = simulation.joint_position();
        const xrobot_cpp::Vec7 qd = simulation.joint_velocity();
        const xrobot_cpp::DynamicsTerms terms = dynamics.dynamics(q, qd);
        const xrobot_cpp::Vec7 scaled_feedforward = gravity_scale * nominal_gravity_feedforward(terms, torque_limit);
        applied_feedforward += (scaled_feedforward - applied_feedforward)
            .cwiseMax(-max_torque_delta).cwiseMin(max_torque_delta);
        const xrobot_cpp::Vec7 impedance_torque = stiffness.cwiseProduct(desired.position - q) +
            damping.cwiseProduct(desired.velocity - qd);
        for (Eigen::Index joint = 0; joint < 7; ++joint) {
          const double headroom = std::max(0.0, torque_limit[joint] - std::abs(applied_feedforward[joint]));
          const double target_scale = std::abs(impedance_torque[joint]) > headroom && std::abs(impedance_torque[joint]) > 1e-9
              ? headroom / std::abs(impedance_torque[joint]) : 1.0;
          applied_gain_scale[joint] = target_scale < applied_gain_scale[joint]
              ? target_scale : std::min(target_scale, applied_gain_scale[joint] + kGainRisePerControlCycle);
        }
        const xrobot_cpp::Vec7 next_total_torque = applied_feedforward + applied_gain_scale.cwiseProduct(impedance_torque);
        peak_torque_delta_nm = std::max(peak_torque_delta_nm, (next_total_torque - applied_total_torque).cwiseAbs().maxCoeff());
        applied_total_torque = next_total_torque;
      }
      peak_torque_nm = std::max(peak_torque_nm, applied_total_torque.cwiseAbs().maxCoeff());
      saturation_hits += static_cast<std::size_t>((applied_total_torque.cwiseAbs().array() >= torque_limit.array() - 1e-9).count());
      simulation.set_joint_torque(applied_total_torque);
      simulation.step();
      const xrobot_cpp::Vec7 q = simulation.joint_position();
      for (Eigen::Index joint = 0; joint < 7; ++joint) {
        if (q[joint] < dynamics.lower_limits()[joint] - 1e-6 || q[joint] > dynamics.upper_limits()[joint] + 1e-6) ++limit_hits;
      }
      const bool trajectory_finished = (target - desired.position).norm() < 1e-5;
      const bool within_settling_window = step + kSettlingWindowSteps >= steps;
      if (trajectory_finished && within_settling_window && (target - q).norm() <= kNearTargetErrorRad) {
        ++near_samples;
        near_peak_acceleration_rad_s2 = std::max(near_peak_acceleration_rad_s2, simulation.joint_acceleration().norm());
      }
    }
    const double final_error_rad = (target - simulation.joint_position()).norm();
    const double final_tcp_error_m = (dynamics.forward_kinematics(target).position - dynamics.forward_kinematics(simulation.joint_position()).position).norm();
    const double final_acceleration_rad_s2 = simulation.joint_acceleration().norm();
    const double final_velocity_rad_s = simulation.joint_velocity().norm();
    const bool passed = limit_hits == 0 && near_samples >= 100 && final_error_rad <= kNearTargetErrorRad &&
        near_peak_acceleration_rad_s2 <= kNearTargetAccelerationRadS2 &&
        final_acceleration_rad_s2 <= kFinalAccelerationRadS2 && final_velocity_rad_s <= kFinalVelocityRadS;
    output << "{\"trial\":" << trial << ",\"passed\":" << (passed ? "true" : "false")
           << ",\"final_joint_error_rad\":" << final_error_rad
           << ",\"final_tcp_error_m\":" << final_tcp_error_m
           << ",\"peak_torque_nm\":" << peak_torque_nm
           << ",\"peak_torque_delta_nm\":" << peak_torque_delta_nm
           << ",\"saturation_hits\":" << saturation_hits
           << ",\"limit_hits\":" << limit_hits
           << ",\"near_target_samples\":" << near_samples
           << ",\"near_target_peak_acceleration_rad_s2\":" << near_peak_acceleration_rad_s2
           << ",\"final_acceleration_rad_s2\":" << final_acceleration_rad_s2
           << ",\"final_velocity_rad_s\":" << final_velocity_rad_s << "}\n";
    final_errors.push_back(final_error_rad);
    near_peak_accelerations.push_back(near_peak_acceleration_rad_s2);
    final_accelerations.push_back(final_acceleration_rad_s2);
    final_velocities.push_back(final_velocity_rad_s);
    total_limit_hits += limit_hits;
    total_saturation_hits += saturation_hits;
    if (passed) ++passed_count;
    if ((trial + 1) % 100 == 0 || trial + 1 == trajectories) {
      std::cout << "CONVERGENCE " << (trial + 1) << "/" << trajectories << " passed=" << passed_count << "\n";
    }
  }
  const auto percentile95 = [](std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(values.size() - 1)))];
  };
  const double p95_final_error = percentile95(final_errors);
  const double p95_near_peak_acc = percentile95(near_peak_accelerations);
  const double p95_final_acc = percentile95(final_accelerations);
  const double p95_final_velocity = percentile95(final_velocities);
  const bool passed = passed_count == trajectories;
  output << "{\"summary\":true,\"passed\":" << (passed ? "true" : "false")
         << ",\"trajectories\":" << trajectories << ",\"passed_trajectories\":" << passed_count
         << ",\"seed\":" << seed
         << ",\"spring_stiffness_nm_rad\":" << spring_stiffness_nm_rad
         << ",\"damping_nm_s_rad\":" << damping_nm_s_rad
         << ",\"gravity_scale\":" << gravity_scale
         << ",\"torque_limit_nm\":" << torque_limit_nm
         << ",\"torque_limits_nm\":" << json_vector(torque_limit)
         << ",\"payload_mass_kg\":" << payload_mass_kg
         << ",\"p95_final_joint_error_rad\":" << p95_final_error
         << ",\"p95_near_target_peak_acceleration_rad_s2\":" << p95_near_peak_acc
         << ",\"p95_final_acceleration_rad_s2\":" << p95_final_acc
         << ",\"p95_final_velocity_rad_s\":" << p95_final_velocity
         << ",\"total_saturation_hits\":" << total_saturation_hits
         << ",\"total_limit_hits\":" << total_limit_hits << "}\n";
  std::cout << "MuJoCo impedance convergence " << (passed ? "PASSED" : "FAILED")
            << ": trajectories=" << trajectories << " passed=" << passed_count
            << " p95_final_error=" << p95_final_error << "rad"
            << " p95_near_target_peak_acc=" << p95_near_peak_acc << "rad/s^2"
            << " p95_final_acc=" << p95_final_acc << "rad/s^2"
            << " p95_final_velocity=" << p95_final_velocity << "rad/s"
            << " gravity_scale=" << gravity_scale
            << " torque_limits=[" << torque_limit.transpose() << "]Nm"
            << " output=" << output_path.string() << "\n";
  std::cout << "No SocketCAN interface was opened.\n";
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

int run_mujoco_full_chain_test(int argc, char** argv) {
  std::filesystem::path output_path{"results/mujoco_full_chain.json"};
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--output" && index + 1 < argc) output_path = argv[++index];
    else throw std::invalid_argument("usage: --mujoco-full-chain-test [--output PATH]");
  }
  if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());

  const std::string root(XROBOT_CPP_SOURCE_DIR);
  xrobot_cpp::NeroDynamicsModel dynamics(root + "/assets/urdf/nero_control_tcp.urdf", "base_link", "gripper_tcp");
#ifdef XROBOT_CPP_WITH_PLACO
  xrobot_cpp::PlacoNeroIkSolver placo_ik(root + "/assets/urdf/nero_control_tcp.urdf", "gripper_tcp");
#endif
  xrobot_cpp::NeroMujocoSimulation simulation(root + "/assets/mujoco/nero_torque.xml");
  const xrobot_cpp::Vec7 initial = (xrobot_cpp::Vec7() << 0.0, -0.20, 0.30, 0.20, -0.10, 0.10, 0.0).finished();
  const xrobot_cpp::Vec7 torque_limit = (xrobot_cpp::Vec7() << 6.0, 6.0, 5.0, 5.0, 4.0, 3.0, 2.0).finished();
  const xrobot_cpp::ComputedTorqueController controller(
      xrobot_cpp::Vec7::Constant(kNeroLowSpeedSpringStiffnessNmRad),
      xrobot_cpp::Vec7::Constant(kNeroLowSpeedDampingNmSRad), torque_limit, xrobot_cpp::JointDriveModel::mujoco_default());
  xrobot_cpp::JerkLimitedJointPlanner trajectory(
      xrobot_cpp::Vec7::Constant(0.60), xrobot_cpp::Vec7::Constant(1.50), xrobot_cpp::Vec7::Constant(12.0));
  Eigen::Matrix3d base_from_xr;
  base_from_xr << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  xrobot_cpp::RelativeMapper mapper(0.70, 1.00, 0.012, 0.05, base_from_xr);
  xrobot_cpp::SafetyGate gate(0.15);
  simulation.reset(initial);
  simulation.set_payload_mass_kg(0.20);
  simulation.set_gripper_width(0.060);
  trajectory.reset(initial);
  xrobot_cpp::JointTrajectoryState desired = trajectory.state();
  xrobot_cpp::Vec7 joint_goal = initial;
  xrobot_cpp::Pose session_tcp_reference = dynamics.forward_kinematics(initial);
  double gripper_target_m = 0.060;
  double min_gripper_m = gripper_target_m, max_gripper_m = gripper_target_m;
  std::size_t activate_count = 0, release_count = 0, timeout_count = 0, saturation_count = 0, limit_violations = 0;
  bool ik_rejected = false, timeout_goal_held = true;
  xrobot_cpp::Vec7 timeout_goal = joint_goal;
  const auto start = xrobot_cpp::Clock::now();

  auto track_frame = [&](const xrobot_cpp::XrFrame& frame, xrobot_cpp::Clock::time_point now) {
    const xrobot_cpp::GateState state = gate.observe(frame, now);
    if (state == xrobot_cpp::GateState::kActivate) {
      ++activate_count;
      joint_goal = simulation.joint_position();
      trajectory.reset(joint_goal);
      desired = trajectory.state();
      session_tcp_reference = dynamics.forward_kinematics(joint_goal);
      mapper.rebase(frame, session_tcp_reference, now);
      return;
    }
    if (state == xrobot_cpp::GateState::kRelease) { ++release_count; return; }
    if (state != xrobot_cpp::GateState::kTrack) return;
    xrobot_cpp::Pose candidate = mapper.target(frame, now).pose;
    xrobot_cpp::Vec3 session_offset = candidate.position - session_tcp_reference.position;
    constexpr double kSessionWorkspaceRadiusM = 0.08;
    if (session_offset.norm() > kSessionWorkspaceRadiusM) session_offset *= kSessionWorkspaceRadiusM / session_offset.norm();
    candidate.position = session_tcp_reference.position + session_offset;
    const xrobot_cpp::Pose reference = dynamics.forward_kinematics(joint_goal);
    xrobot_cpp::Vec3 translation = candidate.position - reference.position;
    constexpr double kMaxTcpTranslationStepM = 0.001;
    if (translation.norm() > kMaxTcpTranslationStepM) translation *= kMaxTcpTranslationStepM / translation.norm();
    candidate.position = reference.position + translation;
    candidate.orientation = session_tcp_reference.orientation;
    try {
      joint_goal = dynamics.solve_ik_step(candidate, joint_goal, 0.015);
      trajectory.set_goal(joint_goal);
      gripper_target_m = std::clamp(gripper_target_m + (frame.grip - frame.trigger) * 0.00035, 0.0, 0.100);
    } catch (const std::exception&) {
      ik_rejected = true;
    }
  };

  for (std::size_t step = 0; step < 5000; ++step) {
    const double time_s = static_cast<double>(step) * 0.001;
    const auto now = start + std::chrono::milliseconds(step);
    const bool deliver_frame = (time_s >= 0.10 && time_s < 1.50) || (time_s >= 2.30 && time_s < 4.00);
    const bool release_frame = time_s >= 1.50 && time_s < 1.80;
    if ((deliver_frame || release_frame) && step % 10 == 0) {
      xrobot_cpp::XrFrame frame;
      frame.sequence = step + 1;
      frame.timestamp_ns = static_cast<std::uint64_t>(step) * 1000000ULL;
      frame.device_id = "synthetic-xr";
      frame.deadman = deliver_frame;
      frame.controller.position = xrobot_cpp::Vec3(0.035 * std::sin(2.0 * time_s), 0.040 * std::sin(1.3 * time_s), 0.020 * std::sin(0.9 * time_s));
      frame.trigger = (time_s > 0.60 && time_s < 0.95) ? 0.70 : 0.0;
      frame.grip = (time_s > 2.80 && time_s < 3.20) ? 0.70 : 0.0;
      track_frame(frame, now);
    }
    const xrobot_cpp::GateState polled_state = gate.poll(now);
    if (polled_state == xrobot_cpp::GateState::kTimeout) {
      ++timeout_count;
      timeout_goal = trajectory.goal();
    }
    if (timeout_count > 0 && time_s >= 1.80 && time_s < 2.30 && (trajectory.goal() - timeout_goal).norm() > 1e-12) timeout_goal_held = false;
    if (step == 2100) {
      const xrobot_cpp::Vec7 preserved_goal = joint_goal;
      xrobot_cpp::Pose unreachable = dynamics.forward_kinematics(joint_goal);
      unreachable.position += xrobot_cpp::Vec3(10.0, 10.0, 10.0);
      try { (void)dynamics.solve_ik(unreachable, joint_goal); }
      catch (const std::exception&) { ik_rejected = true; }
      if ((joint_goal - preserved_goal).norm() > 1e-12) throw std::runtime_error("IK failure altered the retained target");
    }
    desired = trajectory.step(0.001);
    xrobot_cpp::Vec7 disturbance = xrobot_cpp::Vec7::Zero();
    if (time_s > 0.95 && time_s < 1.20) disturbance[2] = 0.35;
    simulation.set_joint_disturbance(disturbance);
    const xrobot_cpp::ComputedTorqueCommand command = controller.compute(
        simulation.dynamics_terms(), simulation.joint_position(), simulation.joint_velocity(), desired);
    saturation_count += (command.feedforward_torque_nm.array().abs() >= torque_limit.array() - 1e-9).count();
    simulation.set_joint_torque(command.feedforward_torque_nm);
    simulation.set_gripper_width(gripper_target_m);
    simulation.step();
    min_gripper_m = std::min(min_gripper_m, simulation.gripper_width_m());
    max_gripper_m = std::max(max_gripper_m, simulation.gripper_width_m());
    const xrobot_cpp::Vec7 q = simulation.joint_position();
    for (int joint = 0; joint < 7; ++joint) {
      if (q[joint] < dynamics.lower_limits()[joint] - 1e-6 || q[joint] > dynamics.upper_limits()[joint] + 1e-6) ++limit_violations;
    }
  }
  const double final_joint_error_rad = (desired.position - simulation.joint_position()).norm();
  const double saturation_fraction = static_cast<double>(saturation_count) / static_cast<double>(5000 * 7);
  const bool passed = activate_count >= 2 && release_count >= 1 && timeout_count >= 1 && timeout_goal_held && ik_rejected &&
                      min_gripper_m < 0.055 && max_gripper_m > 0.055 && final_joint_error_rad < 0.05 &&
                      saturation_fraction <= 0.01 && limit_violations == 0;
  std::ofstream output(output_path);
  if (!output) throw std::runtime_error("cannot open full-chain report: " + output_path.string());
  output << "{\"passed\":" << (passed ? "true" : "false")
         << ",\"spring_stiffness_nm_rad\":" << kNeroLowSpeedSpringStiffnessNmRad
         << ",\"damping_nm_s_rad\":" << kNeroLowSpeedDampingNmSRad
         << ",\"activate_count\":" << activate_count << ",\"release_count\":" << release_count
         << ",\"timeout_count\":" << timeout_count << ",\"timeout_goal_held\":" << (timeout_goal_held ? "true" : "false")
         << ",\"ik_rejected\":" << (ik_rejected ? "true" : "false")
         << ",\"gripper_min_m\":" << min_gripper_m << ",\"gripper_max_m\":" << max_gripper_m
         << ",\"final_joint_error_rad\":" << final_joint_error_rad
         << ",\"saturation_fraction\":" << saturation_fraction
         << ",\"joint_limit_violations\":" << limit_violations << "}\n";
  std::cout << "MuJoCo full-chain test " << (passed ? "PASSED" : "FAILED")
            << ": activate=" << activate_count << " release=" << release_count << " timeout=" << timeout_count
            << " ik_rejected=" << ik_rejected << " final_q_error=" << final_joint_error_rad
            << " saturation=" << saturation_fraction << " limit_hits=" << limit_violations
            << " report=" << output_path.string() << "\n";
  std::cout << "No SocketCAN interface was opened.\n";
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

int run_mujoco_robustness(int argc, char** argv) {
  std::size_t trajectories = 20;
  double seconds = 5.0;
  unsigned int seed = 0;
  double payload_mass_kg = 0.20;
  double disturbance_nm = 0.35;
  double spring_stiffness_nm_rad = kNeroLowSpeedSpringStiffnessNmRad;
  double damping_nm_s_rad = kNeroLowSpeedDampingNmSRad;
  double goal_range_rad = 0.16;
  std::filesystem::path output_path{"results/nero_mujoco_robustness.jsonl"};
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--trajectories" && index + 1 < argc) trajectories = static_cast<std::size_t>(std::stoul(argv[++index]));
    else if (argument == "--seconds-per-trajectory" && index + 1 < argc) seconds = std::stod(argv[++index]);
    else if (argument == "--seed" && index + 1 < argc) seed = static_cast<unsigned int>(std::stoul(argv[++index]));
    else if (argument == "--payload-mass-kg" && index + 1 < argc) payload_mass_kg = std::stod(argv[++index]);
    else if (argument == "--disturbance-nm" && index + 1 < argc) disturbance_nm = std::stod(argv[++index]);
    else if (argument == "--spring-stiffness" && index + 1 < argc) spring_stiffness_nm_rad = std::stod(argv[++index]);
    else if (argument == "--damping" && index + 1 < argc) damping_nm_s_rad = std::stod(argv[++index]);
    else if (argument == "--goal-range-rad" && index + 1 < argc) goal_range_rad = std::stod(argv[++index]);
    else if (argument == "--output" && index + 1 < argc) output_path = argv[++index];
    else throw std::invalid_argument("usage: --mujoco-robustness [--trajectories N] [--seconds-per-trajectory S] [--seed N] [--payload-mass-kg KG] [--disturbance-nm NM] [--spring-stiffness K] [--damping D] [--goal-range-rad RAD] [--output PATH]");
  }
  if (trajectories == 0 || trajectories > 10000) throw std::invalid_argument("--trajectories must be in [1, 10000]");
  if (!std::isfinite(seconds) || seconds < 1.0 || seconds > 30.0) throw std::invalid_argument("--seconds-per-trajectory must be in [1, 30]");
  if (!std::isfinite(payload_mass_kg) || payload_mass_kg < 0.0 || payload_mass_kg > 3.0) throw std::invalid_argument("--payload-mass-kg must be in [0, 3]");
  if (!std::isfinite(disturbance_nm) || disturbance_nm < 0.0 || disturbance_nm > 2.0) throw std::invalid_argument("--disturbance-nm must be in [0, 2]");
  if (!std::isfinite(spring_stiffness_nm_rad) || spring_stiffness_nm_rad <= 0.0 || spring_stiffness_nm_rad > 80.0) throw std::invalid_argument("--spring-stiffness must be in (0, 80]");
  if (!std::isfinite(damping_nm_s_rad) || damping_nm_s_rad <= 0.0 || damping_nm_s_rad > 20.0) throw std::invalid_argument("--damping must be in (0, 20]");
  if (!std::isfinite(goal_range_rad) || goal_range_rad <= 0.0 || goal_range_rad > 0.30) throw std::invalid_argument("--goal-range-rad must be in (0, 0.30]");
  if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
  std::ofstream output(output_path);
  if (!output) throw std::runtime_error("cannot open robustness output: " + output_path.string());
  output << std::setprecision(12);
  const std::string root(XROBOT_CPP_SOURCE_DIR);
  xrobot_cpp::NeroDynamicsModel dynamics(root + "/assets/urdf/nero_control_tcp.urdf", "base_link", "gripper_tcp");
  const std::string model_path = root + "/assets/mujoco/nero_torque.xml";
  const xrobot_cpp::Vec7 initial = (xrobot_cpp::Vec7() << 0.0, -0.20, 0.30, 0.20, -0.10, 0.10, 0.0).finished();
  const xrobot_cpp::Vec7 torque_limit = (xrobot_cpp::Vec7() << 6.0, 6.0, 5.0, 5.0, 4.0, 3.0, 2.0).finished();
  const xrobot_cpp::ComputedTorqueController controller(xrobot_cpp::Vec7::Constant(spring_stiffness_nm_rad), xrobot_cpp::Vec7::Constant(damping_nm_s_rad), torque_limit, xrobot_cpp::JointDriveModel::mujoco_default());
  const std::size_t steps = static_cast<std::size_t>(std::llround(seconds * 1000.0));
  std::uniform_real_distribution<double> goal_delta(-goal_range_rad, goal_range_rad);
  std::uniform_real_distribution<double> phase(0.0, 2.0 * M_PI);
  std::uniform_real_distribution<double> frequency(0.7, 2.0);
  std::size_t pass_count = 0;
  std::size_t fail_count = 0;
  double worst_tcp_position_rms_m = 0.0;
  double worst_tcp_rotation_rms_rad = 0.0;
  for (std::size_t trajectory_index = 0; trajectory_index < trajectories; ++trajectory_index) {
    std::mt19937 trajectory_generator(seed + static_cast<unsigned int>(trajectory_index));
    xrobot_cpp::NeroMujocoSimulation simulation(model_path);
    simulation.reset(initial);
    simulation.set_payload_mass_kg(payload_mass_kg);
    xrobot_cpp::Vec7 goal = initial;
    for (int joint = 0; joint < 7; ++joint) goal[joint] = std::clamp(initial[joint] + goal_delta(trajectory_generator), dynamics.lower_limits()[joint] + 0.12, dynamics.upper_limits()[joint] - 0.12);
    xrobot_cpp::JerkLimitedJointPlanner trajectory(xrobot_cpp::Vec7::Constant(0.60), xrobot_cpp::Vec7::Constant(1.50), xrobot_cpp::Vec7::Constant(12.0));
    trajectory.reset(initial);
    trajectory.set_goal(goal);
    xrobot_cpp::Vec7 phases;
    xrobot_cpp::Vec7 frequencies;
    for (int joint = 0; joint < 7; ++joint) { phases[joint] = phase(trajectory_generator); frequencies[joint] = frequency(trajectory_generator); }
    double joint_squared_error = 0.0;
    double tcp_position_squared_error = 0.0;
    double tcp_rotation_squared_error = 0.0;
    double joint_error_max = 0.0;
    std::size_t saturation_count = 0;
    std::size_t joint_limit_violations = 0;
    bool aborted = false;
    std::vector<double> tcp_position_errors;
    std::vector<double> tcp_rotation_errors;
    tcp_position_errors.reserve(steps);
    tcp_rotation_errors.reserve(steps);
    for (std::size_t step = 0; step < steps; ++step) {
      const xrobot_cpp::JointTrajectoryState desired = trajectory.step(0.001);
      xrobot_cpp::Vec7 disturbance;
      for (int joint = 0; joint < 7; ++joint) disturbance[joint] = disturbance_nm * std::sin(2.0 * M_PI * frequencies[joint] * simulation.time_s() + phases[joint]);
      simulation.set_joint_disturbance(disturbance);
      const xrobot_cpp::ComputedTorqueCommand command = controller.compute(simulation.dynamics_terms(), simulation.joint_position(), simulation.joint_velocity(), desired);
      simulation.set_joint_torque(command.feedforward_torque_nm);
      simulation.step();
      const xrobot_cpp::Vec7 error = desired.position - simulation.joint_position();
      const xrobot_cpp::Pose desired_tcp = dynamics.forward_kinematics(desired.position);
      const xrobot_cpp::Pose actual_tcp = dynamics.forward_kinematics(simulation.joint_position());
      const double position_error = (desired_tcp.position - actual_tcp.position).norm();
      const Eigen::Quaterniond relative_rotation = desired_tcp.orientation.conjugate() * actual_tcp.orientation;
      const double rotation_error = Eigen::AngleAxisd(relative_rotation.normalized()).angle();
      if (!error.allFinite() || !std::isfinite(position_error) || !std::isfinite(rotation_error)) { aborted = true; break; }
      if (step >= steps / 2) {
        joint_squared_error += error.squaredNorm();
        joint_error_max = std::max(joint_error_max, error.norm());
        tcp_position_squared_error += position_error * position_error;
        tcp_rotation_squared_error += rotation_error * rotation_error;
        tcp_position_errors.push_back(position_error);
        tcp_rotation_errors.push_back(rotation_error);
        saturation_count += static_cast<std::size_t>((command.feedforward_torque_nm.cwiseAbs().array() >= torque_limit.array() - 1e-9).count());
        for (int joint = 0; joint < 7; ++joint) if (simulation.joint_position()[joint] < dynamics.lower_limits()[joint] || simulation.joint_position()[joint] > dynamics.upper_limits()[joint]) ++joint_limit_violations;
      }
    }
    const std::size_t samples = tcp_position_errors.size();
    const auto percentile95 = [](std::vector<double>* values) { if (values->empty()) return 0.0; std::sort(values->begin(), values->end()); return (*values)[static_cast<std::size_t>(0.95 * static_cast<double>(values->size() - 1))]; };
    const double tcp_position_rms_m = samples == 0 ? INFINITY : std::sqrt(tcp_position_squared_error / static_cast<double>(samples));
    const double tcp_rotation_rms_rad = samples == 0 ? INFINITY : std::sqrt(tcp_rotation_squared_error / static_cast<double>(samples));
    const double joint_error_norm_rms_rad = samples == 0 ? INFINITY : std::sqrt(joint_squared_error / static_cast<double>(samples));
    const double saturation_fraction = samples == 0 ? 1.0 : static_cast<double>(saturation_count) / static_cast<double>(samples * 7);
    const bool passed = !aborted && tcp_position_rms_m <= 0.005 && tcp_rotation_rms_rad <= 0.08 && joint_error_norm_rms_rad <= 0.05 && saturation_fraction <= 0.01 && joint_limit_violations == 0;
    pass_count += passed ? 1 : 0;
    fail_count += passed ? 0 : 1;
    worst_tcp_position_rms_m = std::max(worst_tcp_position_rms_m, tcp_position_rms_m);
    worst_tcp_rotation_rms_rad = std::max(worst_tcp_rotation_rms_rad, tcp_rotation_rms_rad);
    output << "{\"type\":\"trajectory\",\"cycle\":1,\"spring_stiffness_nm_rad\":" << spring_stiffness_nm_rad << ",\"damping_nm_s_rad\":" << damping_nm_s_rad << ",\"trajectory_index\":" << trajectory_index << ",\"seed\":" << (seed + static_cast<unsigned int>(trajectory_index)) << ",\"passed\":" << (passed ? "true" : "false") << ",\"aborted\":" << (aborted ? "true" : "false") << ",\"samples\":" << samples << ",\"tcp_position_rms_m\":" << tcp_position_rms_m << ",\"tcp_position_p95_m\":" << percentile95(&tcp_position_errors) << ",\"tcp_rotation_rms_rad\":" << tcp_rotation_rms_rad << ",\"tcp_rotation_p95_rad\":" << percentile95(&tcp_rotation_errors) << ",\"joint_error_norm_rms_rad\":" << joint_error_norm_rms_rad << ",\"joint_error_norm_max_rad\":" << joint_error_max << ",\"saturation_fraction\":" << saturation_fraction << ",\"actuator_clip_fraction\":0,\"joint_limit_violations\":" << joint_limit_violations << "}\n";
    std::cout << "ROBUST seed=" << (seed + static_cast<unsigned int>(trajectory_index)) << " passed=" << passed << " tcp_rms=" << tcp_position_rms_m << "m rot_rms=" << tcp_rotation_rms_rad << "rad saturation=" << saturation_fraction << "\n";
  }
  const bool suite_passed = fail_count == 0;
  output << "{\"type\":\"suite\",\"cycle\":1,\"spring_stiffness_nm_rad\":" << spring_stiffness_nm_rad << ",\"damping_nm_s_rad\":" << damping_nm_s_rad << ",\"trajectory_count\":" << trajectories << ",\"passed\":" << (suite_passed ? "true" : "false") << ",\"pass_count\":" << pass_count << ",\"fail_count\":" << fail_count << ",\"worst_tcp_position_rms_m\":" << worst_tcp_position_rms_m << ",\"worst_tcp_rotation_rms_rad\":" << worst_tcp_rotation_rms_rad << "}\n";
  std::cout << "MuJoCo robustness suite completed: pass=" << pass_count << " fail=" << fail_count << " jsonl=" << output_path.string() << "\n";
  std::cout << "No SocketCAN interface was opened.\n";
  return suite_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif

#ifdef XROBOT_CPP_WITH_MUJOCO_VIEWER
int run_mujoco_viewer(int argc, char** argv) {
  double seconds = 0.0;
  double gripper_width_m = 0.060;
  double payload_mass_kg = 0.0;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--seconds" && index + 1 < argc) seconds = std::stod(argv[++index]);
    else if (argument == "--gripper-width-m" && index + 1 < argc) gripper_width_m = std::stod(argv[++index]);
    else if (argument == "--payload-mass-kg" && index + 1 < argc) payload_mass_kg = std::stod(argv[++index]);
    else throw std::invalid_argument("usage: --mujoco-viewer [--seconds S] [--gripper-width-m M] [--payload-mass-kg KG]");
  }
  if (!std::isfinite(seconds) || seconds < 0.0 || seconds > 3600.0) throw std::invalid_argument("--seconds must be in [0, 3600]");
  if (!std::isfinite(gripper_width_m) || gripper_width_m < 0.0 || gripper_width_m > 0.100) throw std::invalid_argument("--gripper-width-m must be in [0, 0.100]");
  if (!std::isfinite(payload_mass_kg) || payload_mass_kg < 0.0 || payload_mass_kg > 3.0) throw std::invalid_argument("--payload-mass-kg must be in [0, 3]");
  const std::string root(XROBOT_CPP_SOURCE_DIR);
  xrobot_cpp::NeroDynamicsModel dynamics(root + "/assets/urdf/nero_control_tcp.urdf", "base_link", "gripper_tcp");
  xrobot_cpp::NeroMujocoSimulation simulation(root + "/assets/mujoco/nero_torque.xml");
  const xrobot_cpp::Vec7 initial = (xrobot_cpp::Vec7() << 0.0, -0.20, 0.30, 0.20, -0.10, 0.10, 0.0).finished();
  const xrobot_cpp::Vec7 reachable_goal = initial + (xrobot_cpp::Vec7() << 0.08, -0.06, 0.04, 0.03, -0.03, 0.02, 0.04).finished();
  const xrobot_cpp::JointTrajectoryState desired{.position = dynamics.solve_ik(dynamics.forward_kinematics(reachable_goal), initial), .velocity = xrobot_cpp::Vec7::Zero(), .acceleration = xrobot_cpp::Vec7::Zero()};
  xrobot_cpp::ComputedTorqueController controller(xrobot_cpp::Vec7::Constant(kNeroLowSpeedSpringStiffnessNmRad), xrobot_cpp::Vec7::Constant(kNeroLowSpeedDampingNmSRad), (xrobot_cpp::Vec7() << 6.0, 6.0, 5.0, 5.0, 4.0, 3.0, 2.0).finished(), xrobot_cpp::JointDriveModel::mujoco_default());
  simulation.reset(initial);
  simulation.set_payload_mass_kg(payload_mass_kg);
  simulation.set_gripper_width(gripper_width_m);
  std::string glfw_error;
  glfwSetErrorCallback([](int, const char* message) { std::cerr << "GLFW: " << message << "\n"; });
  if (!glfwInit()) throw std::runtime_error("GLFW initialization failed; run this command in a desktop session");
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  GLFWwindow* window = glfwCreateWindow(1280, 900, "NERO MuJoCo - torque simulation", nullptr, nullptr);
  if (window == nullptr) { glfwTerminate(); throw std::runtime_error("cannot create MuJoCo viewer window; see GLFW error above"); }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  glfwSetKeyCallback(window, [](GLFWwindow* target, int key, int, int action, int) { if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(target, GLFW_TRUE); });
  mjvCamera camera; mjv_defaultCamera(&camera);
  camera.type = mjCAMERA_FREE; camera.azimuth = 135.0; camera.elevation = -25.0; camera.distance = 1.7;
  camera.lookat[0] = 0.0; camera.lookat[1] = 0.0; camera.lookat[2] = 0.75;
  mjvOption option; mjv_defaultOption(&option);
  mjvScene scene; mjv_defaultScene(&scene); mjv_makeScene(simulation.model(), &scene, 2000);
  mjrContext context; mjr_defaultContext(&context); mjr_makeContext(simulation.model(), &context, mjFONTSCALE_150);
  const auto start = std::chrono::steady_clock::now();
  std::cout << "MuJoCo Viewer started. This is simulation only: no SocketCAN interface is opened. Press Esc or close the window to stop.\n";
  while (!glfwWindowShouldClose(window)) {
    if (seconds > 0.0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >= seconds) break;
    for (int physics_step = 0; physics_step < 16; ++physics_step) {
      const auto command = controller.compute(simulation.dynamics_terms(), simulation.joint_position(), simulation.joint_velocity(), desired);
      simulation.set_joint_torque(command.feedforward_torque_nm);
      simulation.step();
    }
    mjrRect viewport{0, 0, 0, 0}; glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    mjv_updateScene(simulation.model(), simulation.data(), &option, nullptr, &camera, mjCAT_ALL, &scene);
    mjr_render(viewport, &scene, &context);
    glfwSwapBuffers(window); glfwPollEvents();
  }
  mjr_freeContext(&context); mjv_freeScene(&scene); glfwDestroyWindow(window); glfwTerminate();
  std::cout << "MuJoCo Viewer stopped at sim_time_s=" << simulation.time_s() << ". No SocketCAN interface was opened.\n";
  return EXIT_SUCCESS;
}
#endif

int run_dynamics_dry_run(int argc, char** argv) {
  std::string urdf = "../xrobot/third_party/agx_arm_urdf/nero/urdf/nero_description.urdf";
  for (int index = 2; index < argc; ++index) {
    if (std::string_view(argv[index]) == "--urdf" && index + 1 < argc) {
      urdf = argv[++index];
    } else {
      throw std::invalid_argument("usage: --dynamics-dry-run [--urdf PATH]");
    }
  }
  if (!std::filesystem::exists(urdf)) throw std::runtime_error("NERO URDF does not exist: " + urdf);
  xrobot_cpp::NeroDynamicsModel model(urdf);
  const xrobot_cpp::Vec7 measured = (xrobot_cpp::Vec7() << 0.0, -0.20, 0.30, 0.20, -0.10, 0.10, 0.0).finished();
  const xrobot_cpp::Vec7 known_reachable = measured + (xrobot_cpp::Vec7() << 0.010, -0.008, 0.006, 0.004, -0.004, 0.003, 0.002).finished();
  const xrobot_cpp::Pose tcp_target = model.forward_kinematics(known_reachable);
  const xrobot_cpp::Vec7 ik_goal = model.solve_ik(tcp_target, measured);
  const xrobot_cpp::JointTrajectoryState desired{.position = ik_goal, .velocity = xrobot_cpp::Vec7::Zero(), .acceleration = xrobot_cpp::Vec7::Zero()};
  xrobot_cpp::ComputedTorqueController controller(xrobot_cpp::Vec7::Constant(kNeroLowSpeedSpringStiffnessNmRad), xrobot_cpp::Vec7::Constant(kNeroLowSpeedDampingNmSRad),
                                                   (xrobot_cpp::Vec7() << 6.0, 6.0, 5.0, 5.0, 4.0, 3.0, 2.0).finished(), xrobot_cpp::JointDriveModel::mujoco_default());
  const xrobot_cpp::ComputedTorqueCommand command = controller.compute(model, measured, xrobot_cpp::Vec7::Zero(), desired);
  std::cout << "Dynamics dry run completed. No CAN interface was opened and no motor command was sent.\n";
  std::cout << "q_measured=[" << measured.transpose() << "]\n";
  std::cout << "q_target_ik=[" << ik_goal.transpose() << "]\n";
  std::cout << "q_target=[" << command.desired.position.transpose() << "]\n";
  std::cout << "qd_target=[" << command.desired.velocity.transpose() << "]\n";
  std::cout << "qdd_target=[" << command.commanded_acceleration.transpose() << "]\n";
  std::cout << "tau_spring_nm=[" << command.virtual_spring_torque_nm.transpose() << "]\n";
  std::cout << "tau_damping_nm=[" << command.virtual_damping_torque_nm.transpose() << "]\n";
  std::cout << "M_diag=[" << command.dynamics.mass.diagonal().transpose() << "]\n";
  std::cout << "C=[" << command.dynamics.coriolis.transpose() << "]\n";
  std::cout << "G=[" << command.dynamics.gravity.transpose() << "]\n";
  std::cout << "tau_ff_nm=[" << command.feedforward_torque_nm.transpose() << "]\n";
  for (std::uint8_t joint = 1; joint <= 7; ++joint) {
    const xrobot_cpp::MitSetpoint setpoint = controller.mit_setpoint(command, joint - 1);
    const xrobot_cpp::NeroMitCommand mit{.position_rad = setpoint.position_rad, .velocity_rad_s = setpoint.velocity_rad_s,
                                         .kp_nm_rad = setpoint.kp_nm_rad, .kd_nm_s_rad = setpoint.kd_nm_s_rad,
                                         .feedforward_torque_nm = setpoint.feedforward_torque_nm};
    const xrobot_cpp::CanFrame frame = xrobot_cpp::NeroCanProtocolV111::joint_mit(joint, mit);
    std::cout << "MIT J" << static_cast<unsigned>(joint) << " id=0x" << std::hex << frame.id << std::dec << " data=";
    for (const std::uint8_t byte : frame.data) std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
    std::cout << std::dec << "\n";
  }
  return EXIT_SUCCESS;
}

int run_can_protocol_test() {
  const xrobot_cpp::CanFrame mode = xrobot_cpp::NeroCanProtocolV111::mode_joint(50, true);
  const auto joints = xrobot_cpp::NeroCanProtocolV111::joint_target(xrobot_cpp::Vec7::Zero());
  if (mode.id != xrobot_cpp::NeroCanProtocolV111::kModeControl || mode.data[0] != 0x01 || mode.data[1] != 0x01 || mode.data[2] != 50 || mode.data[6] != 0x01 || joints[3].id != xrobot_cpp::NeroCanProtocolV111::kJoint7) {
    throw std::runtime_error("NERO v1.11 protocol self-test failed");
  }
  std::cout << "NERO v1.11 protocol self-test passed. No CAN interface was opened.\n";
  return EXIT_SUCCESS;
}

int run_can_monitor(std::string_view interface_name) {
  xrobot_cpp::NeroSocketCan can{std::string(interface_name)};
  can.open_read_only();
  std::cout << "SocketCAN monitor on " << interface_name << ". Receive only: no NERO command will be transmitted. Ctrl+C to stop.\n";
  while (true) {
    if (const auto frame = can.read(std::chrono::milliseconds(250))) {
      const xrobot_cpp::NeroFeedback feedback = can.feedback();
      std::cout << "CAN id=0x" << std::hex << frame->id << std::dec << " q_valid=" << std::count(feedback.joint_valid.begin(), feedback.joint_valid.end(), true)
                << " driver_valid=" << std::count(feedback.driver_valid.begin(), feedback.driver_valid.end(), true)
                << " enabled=" << std::count(feedback.joint_enabled.begin(), feedback.joint_enabled.end(), true)
                << " faults=" << std::count(feedback.joint_fault.begin(), feedback.joint_fault.end(), true);
      if (feedback.joint_valid[0]) std::cout << " q=[" << std::fixed << std::setprecision(4) << feedback.joints.transpose() << "]";
      std::cout << "\n";
    }
  }
}

int run_feedback_request(int argc, char** argv) {
  if (argc != 3 || std::string_view(argv[2]) != "--can-push-unlock") {
    throw std::invalid_argument("refusing CAN transmit: use --request-feedback --can-push-unlock");
  }
  xrobot_cpp::NeroSocketCan can{"can0"};
  can.open_read_only();
  can.send_normal_single_arm_config();
  can.send_normal_mode_feedback(50);
  std::cout << "Sent normal single-arm config (0x470) plus CAN feedback request (0x151); no joint, gripper, or enable frame was sent.\n";
  return EXIT_SUCCESS;
}

int run_enable_arm(int argc, char** argv) {
  if (argc != 3 || std::string_view(argv[2]) != "--arm-command-unlock") {
    throw std::invalid_argument("refusing CAN transmit: use --enable-arm --arm-command-unlock");
  }
  xrobot_cpp::NeroSocketCan can{"can0"};
  can.open_read_only();
  // Match the controller feedback bootstrap before requesting all-joint enable.
  can.send_normal_single_arm_config();
  can.send_normal_mode_feedback(50);
  const auto bootstrap_deadline = xrobot_cpp::Clock::now() + std::chrono::milliseconds(500);
  while (xrobot_cpp::Clock::now() < bootstrap_deadline) {
    (void)can.read(std::chrono::milliseconds(20));
  }
  const auto deadline = xrobot_cpp::Clock::now() + std::chrono::seconds(2);
  int attempts = 0;
  while (xrobot_cpp::Clock::now() < deadline) {
    can.send_motor_enable(8, true);
    ++attempts;
    for (int read_index = 0; read_index < 2; ++read_index) {
      (void)can.read(std::chrono::milliseconds(50));
      const xrobot_cpp::NeroFeedback feedback = can.feedback();
      const int enabled = std::count(feedback.joint_enabled.begin(), feedback.joint_enabled.end(), true);
      const int faults = std::count(feedback.joint_fault.begin(), feedback.joint_fault.end(), true);
      if (enabled == 7 && faults == 0) {
        std::cout << "NERO enable confirmed after " << attempts << " request(s): enabled=7 faults=0. No joint target was sent.\n";
        return EXIT_SUCCESS;
      }
    }
  }
  throw std::runtime_error("NERO enable was not confirmed by fresh driver feedback");
}

int run_single_joint_test(int argc, char** argv) {
  bool unlocked = false;
  int joint_number = 0;
  double delta_rad = 0.0;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--arm-command-unlock") {
      unlocked = true;
    } else if (argument == "--joint" && index + 1 < argc) {
      joint_number = std::stoi(argv[++index]);
    } else if (argument == "--delta-rad" && index + 1 < argc) {
      delta_rad = std::stod(argv[++index]);
    } else {
      throw std::invalid_argument("unknown or incomplete single-joint-test argument");
    }
  }
  if (!unlocked) throw std::invalid_argument("refusing CAN transmit: --arm-command-unlock is required");
  if (joint_number < 1 || joint_number > 7) throw std::invalid_argument("--joint must be in [1, 7]");
  if (!std::isfinite(delta_rad) || std::abs(delta_rad) < 1e-6 || std::abs(delta_rad) > 0.05) {
    throw std::invalid_argument("--delta-rad must be non-zero and within +/-0.05 rad");
  }

  const xrobot_cpp::Vec7 lower = (xrobot_cpp::Vec7() << -2.70526, -1.74, -2.75, -1.01, -2.75, -0.73, -1.5707963).finished();
  const xrobot_cpp::Vec7 upper = (xrobot_cpp::Vec7() << 2.70526, 1.74, 2.75, 2.14, 2.75, 0.95, 1.5707963).finished();
  xrobot_cpp::NeroCommandInterlock interlock(lower, upper, 0.25, 0.05);
  xrobot_cpp::NeroSocketCan can{"can0"};
  can.open_read_only();
  const auto deadline = xrobot_cpp::Clock::now() + std::chrono::seconds(2);
  xrobot_cpp::NeroFeedback feedback;
  while (xrobot_cpp::Clock::now() < deadline) {
    (void)can.read(std::chrono::milliseconds(50));
    feedback = can.feedback();
    const auto now_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(xrobot_cpp::Clock::now().time_since_epoch()).count());
    interlock.notify_control_input(now_ns);
    if (interlock.observe(feedback, now_ns) == xrobot_cpp::NeroSafetyState::kReady) break;
  }
  if (interlock.state() != xrobot_cpp::NeroSafetyState::kReady) {
    throw std::runtime_error("interlock rejected command: no fresh, enabled, fault-free feedback for all seven joints");
  }

  xrobot_cpp::Vec7 target = feedback.joints;
  target[joint_number - 1] += delta_rad;
  std::string reason;
  const auto now_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(xrobot_cpp::Clock::now().time_since_epoch()).count());
  interlock.notify_control_input(now_ns);
  if (!interlock.permits(target, feedback, now_ns, &reason)) {
    throw std::runtime_error("interlock rejected command: " + reason);
  }

  std::cout << "WARNING: sending one NERO MOVE J target only: J" << joint_number << " delta=" << delta_rad
            << " rad. No enable, gripper, home, or follow-up command is sent.\n";
  can.send_mode_joint(10, false);
  can.send_joint_target(target);
  std::cout << "Sent 5 frames. Set XROBOT_CPP_TX_AUDIT_PATH before launch to retain the C++ TX audit.\n";
  return EXIT_SUCCESS;
}

int run_gravity_envelope_scan(int argc, char** argv) {
  std::size_t samples = 100000;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--samples" && index + 1 < argc) samples = static_cast<std::size_t>(std::stoull(argv[++index]));
    else throw std::invalid_argument("usage: --gravity-envelope-scan [--samples N]");
  }
  if (samples < 1000 || samples > 1000000) throw std::invalid_argument("--samples must be in [1000, 1000000]");
  const std::string urdf = std::string(XROBOT_CPP_SOURCE_DIR) + "/assets/urdf/nero_control_tcp.urdf";
  xrobot_cpp::NeroDynamicsModel model(urdf, "base_link", "gripper_tcp");
  const xrobot_cpp::Vec7 lower = model.lower_limits();
  const xrobot_cpp::Vec7 upper = model.upper_limits();
  std::mt19937_64 rng(20260812);
  xrobot_cpp::Vec7 best_global_q = xrobot_cpp::Vec7::Zero();
  xrobot_cpp::Vec7 best_lower_q = xrobot_cpp::Vec7::Zero();
  xrobot_cpp::Vec7 best_upper_q = xrobot_cpp::Vec7::Zero();
  double best_global = -1.0, best_lower = -1.0, best_upper = -1.0;
  auto sample_q = [&] {
    xrobot_cpp::Vec7 q;
    for (Eigen::Index joint = 0; joint < 7; ++joint) q[joint] = std::uniform_real_distribution<double>(lower[joint], upper[joint])(rng);
    return q;
  };
  auto evaluate = [&](const xrobot_cpp::Vec7& q, double* best, xrobot_cpp::Vec7* best_q) {
    const double magnitude = std::abs(model.dynamics(q, xrobot_cpp::Vec7::Zero()).gravity[1]);
    if (magnitude > *best) { *best = magnitude; *best_q = q; }
  };
  for (std::size_t sample = 0; sample < samples; ++sample) {
    xrobot_cpp::Vec7 q = sample_q();
    evaluate(q, &best_global, &best_global_q);
    q[1] = lower[1]; evaluate(q, &best_lower, &best_lower_q);
    q[1] = upper[1]; evaluate(q, &best_upper, &best_upper_q);
  }
  auto refine = [&](double* best, xrobot_cpp::Vec7* best_q, std::optional<double> fixed_j2) {
    for (double step : {0.20, 0.05, 0.01, 0.002}) {
      bool improved = true;
      while (improved) {
        improved = false;
        for (Eigen::Index joint = 0; joint < 7; ++joint) {
          if (fixed_j2 && joint == 1) continue;
          for (double sign : {-1.0, 1.0}) {
            xrobot_cpp::Vec7 candidate = *best_q;
            candidate[joint] = std::clamp(candidate[joint] + sign * step, lower[joint], upper[joint]);
            const double value = std::abs(model.dynamics(candidate, xrobot_cpp::Vec7::Zero()).gravity[1]);
            if (value > *best + 1e-10) { *best = value; *best_q = candidate; improved = true; }
          }
        }
      }
    }
  };
  refine(&best_global, &best_global_q, std::nullopt);
  refine(&best_lower, &best_lower_q, lower[1]);
  refine(&best_upper, &best_upper_q, upper[1]);
  std::cout << "J2 static gravity envelope from URDF/KDL; no CAN interface was opened. samples=" << samples << "\n";
  std::cout << "J2 lower_limit_rad=" << lower[1] << " max_abs_G2_nm=" << best_lower << " q_rad=[" << best_lower_q.transpose() << "]\n";
  std::cout << "J2 upper_limit_rad=" << upper[1] << " max_abs_G2_nm=" << best_upper << " q_rad=[" << best_upper_q.transpose() << "]\n";
  std::cout << "all_joint_global max_abs_G2_nm=" << best_global << " q_rad=[" << best_global_q.transpose() << "]\n";
  return EXIT_SUCCESS;
}

int run_real_gravity_preview(int argc, char** argv) {
  double seconds = 3.0;
  std::filesystem::path output_path;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--seconds" && index + 1 < argc) seconds = std::stod(argv[++index]);
    else if (argument == "--output" && index + 1 < argc) output_path = argv[++index];
    else throw std::invalid_argument("usage: --real-gravity-preview [--seconds S] [--output PATH]");
  }
  if (!std::isfinite(seconds) || seconds <= 0.0 || seconds > 10.0) throw std::invalid_argument("--seconds must be in (0, 10]");
  const std::string urdf = std::string(XROBOT_CPP_SOURCE_DIR) + "/assets/urdf/nero_control_tcp.urdf";
  xrobot_cpp::NeroDynamicsModel model(urdf, "base_link", "gripper_tcp");
  xrobot_cpp::NeroSocketCan can{"can0"};
  can.open_read_only();
  xrobot_cpp::JointVelocityEstimator velocity_estimator(12.0);
  const auto deadline = xrobot_cpp::Clock::now() + std::chrono::duration_cast<xrobot_cpp::Clock::duration>(std::chrono::duration<double>(seconds));
  xrobot_cpp::NeroFeedback feedback;
  bool valid = false;
  while (xrobot_cpp::Clock::now() < deadline) {
    (void)can.read(std::chrono::milliseconds(20));
    feedback = can.feedback();
    const auto now = xrobot_cpp::Clock::now();
    const auto now_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
    xrobot_cpp::NeroTorqueSafetyLimits limits;
    limits.max_velocity_rad_s = xrobot_cpp::Vec7::Constant(1.0);
    limits.max_torque_nm = xrobot_cpp::Vec7::Constant(16.0);
    limits.max_tracking_error_rad = 0.10;
    xrobot_cpp::NeroTorqueSafetyGate gate(model.lower_limits(), model.upper_limits(), 0.25, limits);
    gate.notify_keepalive(now_ns);
    if (gate.observe(feedback, now_ns) == xrobot_cpp::NeroSafetyState::kReady) {
      velocity_estimator.reset(feedback.joints, now);
      valid = true;
      break;
    }
  }
  if (!valid) throw std::runtime_error("gravity preview requires fresh, enabled, fault-free feedback");
  const auto sample_now = xrobot_cpp::Clock::now();
  for (int index = 0; index < 8; ++index) (void)can.read(std::chrono::milliseconds(10));
  feedback = can.feedback();
  const xrobot_cpp::Vec7 velocity = velocity_estimator.update(feedback.joints, sample_now);
  const xrobot_cpp::DynamicsTerms terms = model.dynamics(feedback.joints, velocity);
  std::ofstream output;
  if (!output_path.empty()) {
    if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
    output.open(output_path);
    if (!output) throw std::runtime_error("cannot open gravity report: " + output_path.string());
    output << "{\"event\":\"gravity_feedforward_check\",\"q_rad\":" << json_vector(feedback.joints)
           << ",\"qd_rad_s\":" << json_vector(velocity)
           << ",\"mass_diag_kg_m2\":" << json_vector(terms.mass.diagonal())
           << ",\"coriolis_nm\":" << json_vector(terms.coriolis)
           << ",\"gravity_nm\":" << json_vector(terms.gravity)
           << ",\"tau_ff_plus_g_nm\":" << json_vector(terms.gravity)
           << ",\"tau_ff_minus_g_nm\":" << json_vector(-terms.gravity) << "}\n";
  }
  std::cout << "Real gravity preview: receive-only. No CAN command was sent.\n";
  std::cout << "q_rad=[" << feedback.joints.transpose() << "]\n";
  std::cout << "qd_rad_s=[" << velocity.transpose() << "]\n";
  std::cout << "gravity_nm=[" << terms.gravity.transpose() << "]\n";
  std::cout << "max_abs_gravity_nm=" << terms.gravity.cwiseAbs().maxCoeff() << "\n";
  for (std::uint8_t joint = 1; joint <= 7; ++joint) {
    const auto plus = xrobot_cpp::NeroCanProtocolV111::joint_mit(joint, {.position_rad = feedback.joints[joint - 1], .velocity_rad_s = 0.0, .kp_nm_rad = 0.0, .kd_nm_s_rad = 0.0, .feedforward_torque_nm = terms.gravity[joint - 1]});
    std::cout << "J" << static_cast<unsigned>(joint) << " +G MIT id=0x" << std::hex << plus.id << " data=";
    for (const auto byte : plus.data) std::cout << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
    std::cout << std::dec << "\n";
  }
  if (!output_path.empty()) std::cout << "report=" << output_path.string() << "\n";
  return EXIT_SUCCESS;
}

int run_cartesian_ik_preview(int argc, char** argv) {
  std::string axis;
  double delta_m = 0.0;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--axis" && index + 1 < argc) axis = argv[++index];
    else if (argument == "--delta-m" && index + 1 < argc) delta_m = std::stod(argv[++index]);
    else throw std::invalid_argument("usage: --cartesian-ik-preview --axis x|y|z --delta-m +/-0.02");
  }
  if (axis != "x" && axis != "y" && axis != "z") throw std::invalid_argument("--axis must be x, y, or z");
  if (!std::isfinite(delta_m) || std::abs(delta_m) < 1e-6 || std::abs(delta_m) > 0.02) throw std::invalid_argument("--delta-m must be non-zero and within +/-0.02m");
  const std::string urdf = std::string(XROBOT_CPP_SOURCE_DIR) + "/assets/urdf/nero_control_tcp.urdf";
  xrobot_cpp::NeroDynamicsModel model(urdf, "base_link", "gripper_tcp");
  xrobot_cpp::NeroSocketCan can{"can0"};
  can.open_read_only();
  xrobot_cpp::NeroFeedback feedback;
  bool complete_feedback = false;
  const auto deadline = xrobot_cpp::Clock::now() + std::chrono::seconds(3);
  while (xrobot_cpp::Clock::now() < deadline) {
    (void)can.read(std::chrono::milliseconds(20));
    feedback = can.feedback();
    complete_feedback = std::all_of(feedback.joint_valid.begin(), feedback.joint_valid.end(), [](bool value) { return value; });
    if (complete_feedback) break;
  }
  if (!complete_feedback) throw std::runtime_error("IK preview requires seven fresh joint feedback values");
  const xrobot_cpp::Pose start = model.forward_kinematics(feedback.joints);
  xrobot_cpp::Pose target = start;
  const int axis_index = axis == "x" ? 0 : (axis == "y" ? 1 : 2);
  target.position[axis_index] += delta_m;
  std::cout << "Cartesian IK preview: receive-only. No CAN command was sent.\n";
  std::cout << "q_rad=[" << feedback.joints.transpose() << "]\n";
  std::cout << "lower_margin_rad=[" << (feedback.joints - model.lower_limits()).transpose() << "]\n";
  std::cout << "upper_margin_rad=[" << (model.upper_limits() - feedback.joints).transpose() << "]\n";
  std::cout << "tcp_start_m=[" << start.position.transpose() << "] target_m=[" << target.position.transpose() << "] orientation_soft_limit_deg=2\n";
  try {
    constexpr double kOrientationToleranceRad = 2.0 * std::numbers::pi / 180.0;
    constexpr double kPreviewContinuityLimitRad = 0.25;
    xrobot_cpp::Vec7 staged_goal = feedback.joints;
    constexpr double kWaypointM = 0.0001;
    constexpr double kPerIterationStepRad = 0.002;
    constexpr double kWaypointToleranceM = 1e-4;
    const int staged_waypoints = std::max(1, static_cast<int>(std::ceil(std::abs(delta_m) / kWaypointM)));
    bool staged_success = true;
    for (int waypoint_index = 1; waypoint_index <= staged_waypoints; ++waypoint_index) {
      xrobot_cpp::Pose waypoint = start;
      waypoint.position = start.position + (target.position - start.position) *
          (static_cast<double>(waypoint_index) / staged_waypoints);
      bool waypoint_reached = false;
      for (int iteration = 0; iteration < 64; ++iteration) {
        const xrobot_cpp::Pose achieved = model.forward_kinematics(staged_goal);
        if ((waypoint.position - achieved.position).norm() <= kWaypointToleranceM &&
            waypoint.orientation.angularDistance(achieved.orientation) <= kOrientationToleranceRad) {
          waypoint_reached = true;
          break;
        }
        const xrobot_cpp::Vec7 next = model.solve_ik_step(waypoint, staged_goal, kPerIterationStepRad);
        if ((next - staged_goal).cwiseAbs().maxCoeff() < 1e-7 ||
            (next - feedback.joints).cwiseAbs().maxCoeff() > kPreviewContinuityLimitRad) break;
        staged_goal = next;
      }
      if (!waypoint_reached) {
        const xrobot_cpp::Pose achieved = model.forward_kinematics(staged_goal);
        std::cout << "IK PREVIEW staged differential path rejected at waypoint=" << waypoint_index
                  << " position_error_m=" << (waypoint.position - achieved.position).norm()
                  << " orientation_error_rad=" << waypoint.orientation.angularDistance(achieved.orientation)
                  << " cumulative_delta_rad=" << (staged_goal - feedback.joints).cwiseAbs().maxCoeff() << "\n";
        staged_success = false;
        break;
      }
    }
    const xrobot_cpp::Pose staged_achieved = model.forward_kinematics(staged_goal);
    const double staged_position_error = (target.position - staged_achieved.position).norm();
    const double staged_orientation_error = target.orientation.angularDistance(staged_achieved.orientation);
    const double staged_max_joint_delta = (staged_goal - feedback.joints).cwiseAbs().maxCoeff();
    if (staged_success && staged_position_error <= 5e-4 && staged_orientation_error <= kOrientationToleranceRad &&
        staged_max_joint_delta <= kPreviewContinuityLimitRad) {
      std::cout << "IK PREVIEW PASS (0.1 mm staged differential path) q_goal=[" << staged_goal.transpose()
                << "] max_joint_delta_rad=" << staged_max_joint_delta
                << " tcp_position_error_m=" << staged_position_error
                << " orientation_error_rad=" << staged_orientation_error << "\n";
      return EXIT_SUCCESS;
    }
    xrobot_cpp::Vec7 local_goal = feedback.joints;
    for (int iteration = 0; iteration < 600; ++iteration) {
      const xrobot_cpp::Pose local_pose = model.forward_kinematics(local_goal);
      const double position_error = (target.position - local_pose.position).norm();
      const double orientation_error = target.orientation.angularDistance(local_pose.orientation);
      if (position_error <= 5e-4 && orientation_error <= kOrientationToleranceRad) break;
      const xrobot_cpp::Vec7 next = model.solve_ik_step(target, local_goal, 0.004);
      if ((next - local_goal).cwiseAbs().maxCoeff() < 1e-7) break;
      local_goal = next;
    }
    const xrobot_cpp::Pose local_achieved = model.forward_kinematics(local_goal);
    const double local_position_error = (target.position - local_achieved.position).norm();
    const double local_orientation_error = target.orientation.angularDistance(local_achieved.orientation);
    const double local_max_joint_delta = (local_goal - feedback.joints).cwiseAbs().maxCoeff();
    if (local_position_error <= 5e-4 && local_orientation_error <= kOrientationToleranceRad &&
        local_max_joint_delta <= kPreviewContinuityLimitRad) {
      std::cout << "IK PREVIEW PASS (bounded differential path) q_goal=[" << local_goal.transpose()
                << "] max_joint_delta_rad=" << local_max_joint_delta
                << " tcp_position_error_m=" << local_position_error
                << " orientation_error_rad=" << local_orientation_error << "\n";
      return EXIT_SUCCESS;
    }
    std::cout << "IK PREVIEW local path did not meet constraints: position_error_m=" << local_position_error
              << " orientation_error_rad=" << local_orientation_error
              << " max_joint_delta_rad=" << local_max_joint_delta << "\n";
    const xrobot_cpp::Vec7 goal = model.solve_ik(target, feedback.joints, kOrientationToleranceRad);
    const xrobot_cpp::Pose achieved = model.forward_kinematics(goal);
    const double max_joint_delta = (goal - feedback.joints).cwiseAbs().maxCoeff();
    if (max_joint_delta > kPreviewContinuityLimitRad) {
      std::cout << "IK PREVIEW FAIL: reachable only through a discontinuous branch: max_joint_delta_rad="
                << max_joint_delta << " limit_rad=" << kPreviewContinuityLimitRad
                << " tcp_position_error_m=" << (target.position - achieved.position).norm() << "\n";
      return EXIT_FAILURE;
    }
    std::cout << "IK PREVIEW PASS q_goal=[" << goal.transpose() << "] max_joint_delta_rad=" << max_joint_delta
              << " tcp_position_error_m=" << (target.position - achieved.position).norm() << "\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cout << "IK PREVIEW FAIL: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}

int run_mit_single_joint_hold(int argc, char** argv) {
  bool unlocked = false, dry_run = false, supported_joint = false;
  int joint_number = 0;
  double torque_nm = 0.0, seconds = 1.0;
  std::filesystem::path telemetry_path;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--arm-command-unlock") unlocked = true;
    else if (argument == "--dry-run") dry_run = true;
    else if (argument == "--supported-joint") supported_joint = true;
    else if (argument == "--telemetry-log" && index + 1 < argc) telemetry_path = argv[++index];
    else if (argument == "--joint" && index + 1 < argc) joint_number = std::stoi(argv[++index]);
    else if (argument == "--torque-nm" && index + 1 < argc) torque_nm = std::stod(argv[++index]);
    else if (argument == "--seconds" && index + 1 < argc) seconds = std::stod(argv[++index]);
    else throw std::invalid_argument("usage: --mit-single-joint-hold --arm-command-unlock --supported-joint --joint 7 --torque-nm +/-1.5 --seconds <=0.1 [--telemetry-log PATH] [--dry-run]");
  }
  if (!unlocked) throw std::invalid_argument("refusing MIT transmit: --arm-command-unlock is required");
  if (!dry_run && !supported_joint) throw std::invalid_argument("single-joint direction check requires --supported-joint confirmation");
  if (joint_number < 1 || joint_number > 7) throw std::invalid_argument("--joint must be in [1, 7]");
  if (!std::isfinite(torque_nm) || std::abs(torque_nm) < 1e-6 || std::abs(torque_nm) > 1.5) {
    throw std::invalid_argument("--torque-nm must be non-zero and within +/-1.5 Nm");
  }
  if (!std::isfinite(seconds) || seconds <= 0.0 || seconds > 0.10) {
    throw std::invalid_argument("--seconds must be in (0, 0.10]");
  }
  if (dry_run) {
    const xrobot_cpp::NeroMitCommand sample{.position_rad = 0.0, .velocity_rad_s = 0.0, .kp_nm_rad = 0.0, .kd_nm_s_rad = 0.0, .feedforward_torque_nm = torque_nm};
    const xrobot_cpp::CanFrame mode = xrobot_cpp::NeroCanProtocolV111::mode_mit(10, true);
    const xrobot_cpp::CanFrame command = xrobot_cpp::NeroCanProtocolV111::joint_mit(static_cast<std::uint8_t>(joint_number), sample);
    std::cout << "MIT dry run passed: mode_id=0x" << std::hex << mode.id << " joint_id=0x" << command.id << std::dec
              << " J" << joint_number << " torque=" << torque_nm << "Nm. No SocketCAN interface was opened.\n";
    return EXIT_SUCCESS;
  }

  const std::string urdf = std::string(XROBOT_CPP_SOURCE_DIR) + "/../xrobot/third_party/agx_arm_urdf/nero/urdf/nero_description.urdf";
  xrobot_cpp::NeroDynamicsModel model(urdf);
  xrobot_cpp::NeroTorqueSafetyLimits limits;
  limits.max_velocity_rad_s = xrobot_cpp::Vec7::Constant(0.20);
  limits.max_torque_nm = xrobot_cpp::Vec7::Constant(1e-6);
  limits.max_torque_nm[joint_number - 1] = std::abs(torque_nm);
  limits.max_tracking_error_rad = 0.02;
  xrobot_cpp::NeroTorqueSafetyGate safety(model.lower_limits(), model.upper_limits(), 0.25, limits);
  xrobot_cpp::JointVelocityEstimator velocity_estimator(12.0);
  xrobot_cpp::NeroSocketCan can{"can0"};
  can.open_read_only();
  xrobot_cpp::NeroFeedback feedback;
  const auto feedback_deadline = xrobot_cpp::Clock::now() + std::chrono::seconds(2);
  while (xrobot_cpp::Clock::now() < feedback_deadline) {
    (void)can.read(std::chrono::milliseconds(20));
    feedback = can.feedback();
    const auto now = xrobot_cpp::Clock::now();
    const auto now_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
    safety.notify_keepalive(now_ns);
    if (safety.observe(feedback, now_ns) == xrobot_cpp::NeroSafetyState::kReady) break;
  }
  const auto ready_now = xrobot_cpp::Clock::now();
  const auto ready_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(ready_now.time_since_epoch()).count());
  safety.notify_keepalive(ready_ns);
  if (safety.observe(feedback, ready_ns) != xrobot_cpp::NeroSafetyState::kReady) {
    throw std::runtime_error("MIT test blocked: fresh, enabled, fault-free feedback is required");
  }
  velocity_estimator.reset(feedback.joints, ready_now);
  xrobot_cpp::JointTrajectoryState hold{.position = feedback.joints, .velocity = xrobot_cpp::Vec7::Zero(), .acceleration = xrobot_cpp::Vec7::Zero()};
  xrobot_cpp::Vec7 torque = xrobot_cpp::Vec7::Zero();
  torque[joint_number - 1] = torque_nm;
  const double initial_joint_position_rad = feedback.joints[joint_number - 1];
  double peak_abs_velocity_rad_s = 0.0;
  xrobot_cpp::NeroMitCommand command{.position_rad = hold.position[joint_number - 1], .velocity_rad_s = 0.0, .kp_nm_rad = 0.0, .kd_nm_s_rad = 0.0, .feedforward_torque_nm = torque_nm};
  std::ofstream telemetry;
  if (!telemetry_path.empty()) {
    if (!telemetry_path.parent_path().empty()) std::filesystem::create_directories(telemetry_path.parent_path());
    telemetry.open(telemetry_path);
    if (!telemetry) throw std::runtime_error("cannot open MIT telemetry: " + telemetry_path.string());
  }
  bool mit_mode_entered = false;
  auto restore_joint_mode = [&] {
    if (!mit_mode_entered) return;
    try {
      can.send_mode_joint(10, true);
      for (int repeat = 0; repeat < 3; ++repeat) can.send_joint_target(feedback.joints);
      std::cout << "MIT STOP: restored joint mode and sent measured-position hold targets; no zero-torque MIT frame was sent.\n";
    } catch (const std::exception& error) {
      std::cerr << "MIT STOP WARNING: failed to restore joint mode: " << error.what() << "\n";
    }
    mit_mode_entered = false;
  };
  try {
    g_mit_stop_requested = 0;
    std::signal(SIGINT, request_mit_stop);
    std::cout << "WARNING: entering supported MIT direction check for J" << joint_number << ", torque=" << torque_nm << "Nm, duration<=" << seconds
              << "s. The arm must be physically supported. Ctrl+C requests joint-mode recovery.\n";
    can.send_mode_mit(10, true);
    mit_mode_entered = true;
    const auto deadline = xrobot_cpp::Clock::now() + std::chrono::duration_cast<xrobot_cpp::Clock::duration>(std::chrono::duration<double>(seconds));
    auto next_cycle = xrobot_cpp::Clock::now();
    std::size_t sent_cycles = 0;
    while (xrobot_cpp::Clock::now() < deadline && !g_mit_stop_requested) {
      for (int read_index = 0; read_index < 32; ++read_index) {
        if (!can.read(read_index == 0 ? std::chrono::milliseconds(2) : std::chrono::milliseconds(0))) break;
      }
      const auto now = xrobot_cpp::Clock::now();
      const auto now_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
      feedback = can.feedback();
      const xrobot_cpp::Vec7 measured_velocity = velocity_estimator.update(feedback.joints, now);
      const xrobot_cpp::DynamicsTerms dynamics = model.dynamics(feedback.joints, measured_velocity);
      peak_abs_velocity_rad_s = std::max(peak_abs_velocity_rad_s, std::abs(measured_velocity[joint_number - 1]));
      write_mit_telemetry(&telemetry, now_ns, feedback, measured_velocity, hold, dynamics, torque,
                          xrobot_cpp::Vec7::Zero(), xrobot_cpp::Vec7::Zero(), "single_joint_direction_check");
      safety.notify_keepalive(now_ns);
      std::string reason;
      if (!safety.permits(hold, measured_velocity, torque, feedback, now_ns, &reason)) {
        throw std::runtime_error("MIT safety stop: " + reason);
      }
      can.send_joint_mit(static_cast<std::uint8_t>(joint_number), command);
      ++sent_cycles;
      next_cycle += std::chrono::milliseconds(10);
      std::this_thread::sleep_until(next_cycle);
    }
    restore_joint_mode();
    const double final_joint_position_rad = feedback.joints[joint_number - 1];
    const double joint_delta_rad = final_joint_position_rad - initial_joint_position_rad;
    const char* response_direction = std::abs(joint_delta_rad) < 1e-4 ? "inconclusive" : (joint_delta_rad > 0.0 ? "+joint" : "-joint");
    if (telemetry) {
      telemetry << "{\"event\":\"single_joint_direction_summary\",\"joint\":" << joint_number
                << ",\"commanded_torque_nm\":" << torque_nm
                << ",\"initial_joint_position_rad\":" << initial_joint_position_rad
                << ",\"final_joint_position_rad\":" << final_joint_position_rad
                << ",\"joint_delta_rad\":" << joint_delta_rad
                << ",\"peak_abs_velocity_rad_s\":" << peak_abs_velocity_rad_s
                << ",\"response_direction\":\"" << response_direction << "\"}\n";
      telemetry.flush();
    }
    std::cout << "MIT single-joint direction check completed: joint=" << joint_number << " sent_cycles=" << sent_cycles
              << " interrupted=" << (g_mit_stop_requested ? "true" : "false")
              << " delta_rad=" << joint_delta_rad << " response_direction=" << response_direction
              << " peak_velocity_rad_s=" << peak_abs_velocity_rad_s << ".\n";
  } catch (...) {
    restore_joint_mode();
    throw;
  }
  return EXIT_SUCCESS;
}

int run_mit_effective_torque_collect(int argc, char** argv) {
  bool unlocked = false, dry_run = false, supported_arm = false;
  int joint_number = 0;
  double amplitude_rad = 0.010, probe_torque_nm = 0.25, frequency_hz = 0.15, seconds = 20.0;
  std::filesystem::path telemetry_path;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--arm-command-unlock") unlocked = true;
    else if (argument == "--dry-run") dry_run = true;
    else if (argument == "--supported-arm") supported_arm = true;
    else if (argument == "--joint" && index + 1 < argc) joint_number = std::stoi(argv[++index]);
    else if (argument == "--amplitude-rad" && index + 1 < argc) amplitude_rad = std::stod(argv[++index]);
    else if (argument == "--probe-torque-nm" && index + 1 < argc) probe_torque_nm = std::stod(argv[++index]);
    else if (argument == "--frequency-hz" && index + 1 < argc) frequency_hz = std::stod(argv[++index]);
    else if (argument == "--seconds" && index + 1 < argc) seconds = std::stod(argv[++index]);
    else if (argument == "--telemetry-log" && index + 1 < argc) telemetry_path = argv[++index];
    else throw std::invalid_argument("usage: --mit-effective-torque-collect --arm-command-unlock --supported-arm --joint 2 --telemetry-log PATH [--amplitude-rad 0.010] [--probe-torque-nm 0.25] [--frequency-hz 0.15] [--seconds 20] [--dry-run]");
  }
  if (!unlocked) throw std::invalid_argument("refusing MIT transmit: --arm-command-unlock is required");
  if (!dry_run && !supported_arm) throw std::invalid_argument("effective torque collection requires physical arm support and --supported-arm");
  if (joint_number < 1 || joint_number > 7) throw std::invalid_argument("--joint must be in [1, 7]");
  if (!std::isfinite(amplitude_rad) || amplitude_rad < 0.005 || amplitude_rad > 0.020) throw std::invalid_argument("--amplitude-rad must be in [0.005, 0.020]");
  if (!std::isfinite(probe_torque_nm) || probe_torque_nm < 0.05 || probe_torque_nm > 0.50) throw std::invalid_argument("--probe-torque-nm must be in [0.05, 0.50]");
  if (!std::isfinite(frequency_hz) || frequency_hz < 0.05 || frequency_hz > 0.25) throw std::invalid_argument("--frequency-hz must be in [0.05, 0.25]");
  if (!std::isfinite(seconds) || seconds < 12.0 || seconds > 30.0) throw std::invalid_argument("--seconds must be in [12, 30]");
  if (telemetry_path.empty()) throw std::invalid_argument("--telemetry-log is required");
  if (dry_run) {
    std::cout << "Effective torque collection dry run passed: J" << joint_number << " +/-" << amplitude_rad
              << "rad with +/-" << probe_torque_nm << "Nm torque probe at " << frequency_hz << "Hz for " << seconds << "s. No SocketCAN interface was opened.\n";
    return EXIT_SUCCESS;
  }

  const std::string urdf = std::string(XROBOT_CPP_SOURCE_DIR) + "/assets/urdf/nero_control_tcp.urdf";
  xrobot_cpp::NeroDynamicsModel model(urdf, "base_link", "gripper_tcp");
  xrobot_cpp::NeroTorqueSafetyLimits limits;
  limits.max_velocity_rad_s = xrobot_cpp::Vec7::Constant(0.50);
  limits.max_torque_nm = mit_support_torque_limits();
  limits.max_tracking_error_rad = 0.08;
  xrobot_cpp::NeroTorqueSafetyGate safety(model.lower_limits(), model.upper_limits(), 0.25, limits);
  xrobot_cpp::JointVelocityEstimator velocity_estimator(8.0);
  xrobot_cpp::NeroSocketCan can{"can0"};
  can.open_read_only();
  xrobot_cpp::NeroFeedback feedback;
  const auto feedback_deadline = xrobot_cpp::Clock::now() + std::chrono::seconds(2);
  while (xrobot_cpp::Clock::now() < feedback_deadline) {
    (void)can.read(std::chrono::milliseconds(20));
    feedback = can.feedback();
    const auto now = xrobot_cpp::Clock::now();
    const auto now_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
    safety.notify_keepalive(now_ns);
    if (safety.observe(feedback, now_ns) == xrobot_cpp::NeroSafetyState::kReady) break;
  }
  const auto ready_now = xrobot_cpp::Clock::now();
  const auto ready_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(ready_now.time_since_epoch()).count());
  safety.notify_keepalive(ready_ns);
  if (safety.observe(feedback, ready_ns) != xrobot_cpp::NeroSafetyState::kReady) {
    throw std::runtime_error("effective torque collection blocked: fresh, enabled, fault-free feedback is required");
  }
  const Eigen::Index selected = joint_number - 1;
  if (feedback.joints[selected] - model.lower_limits()[selected] < amplitude_rad + 0.03 ||
      model.upper_limits()[selected] - feedback.joints[selected] < amplitude_rad + 0.03) {
    throw std::runtime_error("effective torque collection blocked: selected joint is too near its URDF limit");
  }
  velocity_estimator.reset(feedback.joints, ready_now);
  const xrobot_cpp::JointTrajectoryState hold{.position = feedback.joints, .velocity = xrobot_cpp::Vec7::Zero(), .acceleration = xrobot_cpp::Vec7::Zero()};
  const xrobot_cpp::Vec7 kp = xrobot_cpp::Vec7::Constant(2.0);
  const xrobot_cpp::Vec7 kd = xrobot_cpp::Vec7::Constant(0.40);
  const xrobot_cpp::DynamicsTerms initial_dynamics = model.dynamics(hold.position, xrobot_cpp::Vec7::Zero());
  const double requested_peak_torque_nm = std::abs(initial_dynamics.gravity[selected]) + probe_torque_nm +
      kp[selected] * amplitude_rad + kd[selected] * amplitude_rad * (2.0 * std::numbers::pi * frequency_hz);
  constexpr double kCollectionTorqueReserveNm = 0.40;
  if (requested_peak_torque_nm > limits.max_torque_nm[selected] - kCollectionTorqueReserveNm) {
    std::ostringstream error;
    error << "effective torque collection blocked before MIT entry: J" << joint_number
          << " has insufficient torque headroom at this pose (predicted peak=" << requested_peak_torque_nm
          << "Nm, required limit<=" << (limits.max_torque_nm[selected] - kCollectionTorqueReserveNm)
          << "Nm after " << kCollectionTorqueReserveNm << "Nm reserve)";
    throw std::runtime_error(error.str());
  }
  if (!telemetry_path.parent_path().empty()) std::filesystem::create_directories(telemetry_path.parent_path());
  std::ofstream telemetry(telemetry_path);
  if (!telemetry) throw std::runtime_error("cannot open effective torque telemetry: " + telemetry_path.string());

  bool mit_mode_entered = false;
  auto restore_joint_mode = [&] {
    if (!mit_mode_entered) return;
    try {
      can.send_mode_joint(10, true);
      const xrobot_cpp::Vec7 hold_target = feedback.joints;
      for (int repeat = 0; repeat < 3; ++repeat) can.send_joint_target(hold_target);
      std::cout << "MIT STOP: restored joint mode and sent three measured-position hold targets; no zero-torque MIT frame was sent.\n";
    } catch (const std::exception& error) { std::cerr << "MIT STOP WARNING: " << error.what() << "\n"; }
    mit_mode_entered = false;
  };
  try {
    g_mit_stop_requested = 0;
    std::signal(SIGINT, request_mit_stop);
    std::cout << "WARNING: effective torque collection drives J" << joint_number << " through a symmetric +/-" << amplitude_rad
              << "rad position sine plus an independent +/-" << probe_torque_nm << "Nm torque sine at " << frequency_hz << "Hz for <=" << seconds
              << "s. Other joints hold with G(q)+Kp=2.0,Kd=0.40. The arm must remain physically supported.\n";
    can.send_mode_mit(10, true);
    mit_mode_entered = true;
    // Precharge all seven axes immediately so MIT entry never creates a zero-torque interval.
    const xrobot_cpp::Vec7 initial_equivalent_torque = initial_dynamics.gravity;
    std::string initial_reason;
    if (!safety.permits(hold, xrobot_cpp::Vec7::Zero(), initial_equivalent_torque, feedback, ready_ns, &initial_reason)) {
      throw std::runtime_error("effective torque collection blocked: " + initial_reason);
    }
    for (std::uint8_t joint = 1; joint <= 7; ++joint) {
      can.send_joint_mit(joint, {.position_rad = hold.position[joint - 1], .velocity_rad_s = 0.0,
                                 .kp_nm_rad = kp[joint - 1], .kd_nm_s_rad = kd[joint - 1],
                                 .feedforward_torque_nm = initial_dynamics.gravity[joint - 1]});
    }
    const auto start = xrobot_cpp::Clock::now();
    const auto deadline = start + std::chrono::duration_cast<xrobot_cpp::Clock::duration>(std::chrono::duration<double>(seconds));
    auto next_cycle = start;
    std::size_t sent_cycles = 1;
    while (xrobot_cpp::Clock::now() < deadline && !g_mit_stop_requested) {
      for (int read_index = 0; read_index < 32; ++read_index) {
        if (!can.read(read_index == 0 ? std::chrono::milliseconds(2) : std::chrono::milliseconds(0))) break;
      }
      const auto now = xrobot_cpp::Clock::now();
      const double elapsed_s = std::chrono::duration<double>(now - start).count();
      const auto now_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
      feedback = can.feedback();
      const xrobot_cpp::Vec7 velocity = velocity_estimator.update(feedback.joints, now);
      xrobot_cpp::JointTrajectoryState desired = hold;
      const double omega = 2.0 * std::numbers::pi * frequency_hz;
      // The 90-degree phase offset prevents the position impedance and t_ff probe from being collinear.
      desired.position[selected] += amplitude_rad * std::cos(omega * elapsed_s);
      desired.velocity[selected] = -amplitude_rad * omega * std::sin(omega * elapsed_s);
      desired.acceleration[selected] = -amplitude_rad * omega * omega * std::cos(omega * elapsed_s);
      const xrobot_cpp::DynamicsTerms dynamics = model.dynamics(feedback.joints, velocity);
      xrobot_cpp::Vec7 feedforward = dynamics.gravity;
      feedforward[selected] += probe_torque_nm * std::sin(omega * elapsed_s);
      const xrobot_cpp::Vec7 equivalent_torque = feedforward + kp.cwiseProduct(desired.position - feedback.joints) +
          kd.cwiseProduct(desired.velocity - velocity);
      safety.notify_keepalive(now_ns);
      std::string reason;
      if (!safety.permits(desired, velocity, equivalent_torque, feedback, now_ns, &reason)) {
        throw std::runtime_error("effective torque collection safety stop: " + reason);
      }
      write_mit_telemetry(&telemetry, now_ns, feedback, velocity, desired, dynamics, feedforward, kp, kd, "effective_torque_collect");
      for (std::uint8_t joint = 1; joint <= 7; ++joint) {
        can.send_joint_mit(joint, {.position_rad = desired.position[joint - 1], .velocity_rad_s = desired.velocity[joint - 1],
                                   .kp_nm_rad = kp[joint - 1], .kd_nm_s_rad = kd[joint - 1],
                                   .feedforward_torque_nm = feedforward[joint - 1]});
      }
      ++sent_cycles;
      next_cycle += std::chrono::milliseconds(10);
      std::this_thread::sleep_until(next_cycle);
    }
    restore_joint_mode();
    std::cout << "Effective torque collection completed: joint=" << joint_number << " sent_cycles=" << sent_cycles
              << " interrupted=" << (g_mit_stop_requested ? "true" : "false") << " telemetry=" << telemetry_path.string() << "\n";
  } catch (...) { restore_joint_mode(); throw; }
  return EXIT_SUCCESS;
}

int run_mit_seven_axis_gravity_hold(int argc, char** argv) {
  bool unlocked = false, dry_run = false, supported_arm = false;
  double gravity_scale = 1.0, seconds = 0.10;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--arm-command-unlock") unlocked = true;
    else if (argument == "--dry-run") dry_run = true;
    else if (argument == "--supported-arm") supported_arm = true;
    else if (argument == "--gravity-scale" && index + 1 < argc) gravity_scale = std::stod(argv[++index]);
    else if (argument == "--seconds" && index + 1 < argc) seconds = std::stod(argv[++index]);
    else throw std::invalid_argument("usage: --mit-seven-axis-gravity-hold --arm-command-unlock --supported-arm [--gravity-scale <=1.00] [--seconds <=0.20] [--dry-run]");
  }
  if (!unlocked) throw std::invalid_argument("refusing MIT transmit: --arm-command-unlock is required");
  if (!dry_run && !supported_arm) throw std::invalid_argument("refusing full-gravity MIT test: physically support the arm and pass --supported-arm");
  if (!std::isfinite(gravity_scale) || gravity_scale <= 0.0 || gravity_scale > 1.00) throw std::invalid_argument("--gravity-scale must be in (0, 1.00]");
  if (!std::isfinite(seconds) || seconds <= 0.0 || seconds > 0.10) throw std::invalid_argument("--seconds must be in (0, 0.10]");
  if (dry_run) {
    std::cout << "Seven-axis gravity-hold dry run passed: scale=" << gravity_scale
              << ", per-axis torque limits=[" << (gravity_scale * mit_support_torque_limits()).transpose() << "]Nm. No SocketCAN interface was opened.\n";
    return EXIT_SUCCESS;
  }
  const std::string urdf = std::string(XROBOT_CPP_SOURCE_DIR) + "/assets/urdf/nero_control_tcp.urdf";
  xrobot_cpp::NeroDynamicsModel model(urdf, "base_link", "gripper_tcp");
  xrobot_cpp::NeroTorqueSafetyLimits limits;
  limits.max_velocity_rad_s = xrobot_cpp::Vec7::Constant(3.0);
  limits.max_torque_nm = gravity_scale * mit_support_torque_limits();
  limits.max_tracking_error_rad = 0.25;
  xrobot_cpp::NeroTorqueSafetyGate safety(model.lower_limits(), model.upper_limits(), 0.25, limits);
  xrobot_cpp::ComputedTorqueController controller(
      xrobot_cpp::Vec7::Zero(), xrobot_cpp::Vec7::Zero(), limits.max_torque_nm);
  xrobot_cpp::JointVelocityEstimator velocity_estimator(12.0);
  xrobot_cpp::NeroSocketCan can{"can0"};
  can.open_read_only();
  xrobot_cpp::NeroFeedback feedback;
  const auto feedback_deadline = xrobot_cpp::Clock::now() + std::chrono::seconds(2);
  while (xrobot_cpp::Clock::now() < feedback_deadline) {
    (void)can.read(std::chrono::milliseconds(20));
    feedback = can.feedback();
    const auto now = xrobot_cpp::Clock::now();
    const auto now_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
    safety.notify_keepalive(now_ns);
    if (safety.observe(feedback, now_ns) == xrobot_cpp::NeroSafetyState::kReady) break;
  }
  const auto ready_now = xrobot_cpp::Clock::now();
  const auto ready_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(ready_now.time_since_epoch()).count());
  safety.notify_keepalive(ready_ns);
  if (safety.observe(feedback, ready_ns) != xrobot_cpp::NeroSafetyState::kReady) throw std::runtime_error("seven-axis MIT test blocked: fresh, enabled, fault-free feedback is required");
  velocity_estimator.reset(feedback.joints, ready_now);
  xrobot_cpp::JointTrajectoryState hold{.position = feedback.joints, .velocity = xrobot_cpp::Vec7::Zero(), .acceleration = xrobot_cpp::Vec7::Zero()};
  const xrobot_cpp::Vec7 initial_torque = (gravity_scale * model.dynamics(feedback.joints, xrobot_cpp::Vec7::Zero()).gravity)
      .cwiseMax(-limits.max_torque_nm).cwiseMin(limits.max_torque_nm);
  std::string initial_reason;
  if (!safety.permits(hold, xrobot_cpp::Vec7::Zero(), initial_torque, feedback, ready_ns, &initial_reason)) {
    throw std::runtime_error("seven-axis MIT test blocked: " + initial_reason);
  }
  bool mit_mode_entered = false;
  auto restore_joint_mode = [&] {
    if (!mit_mode_entered) return;
    try {
      can.send_mode_joint(10, true);
      const xrobot_cpp::Vec7 hold_target = feedback.joints;
      for (int repeat = 0; repeat < 3; ++repeat) can.send_joint_target(hold_target);
      std::cout << "MIT STOP: restored joint mode and sent three measured-position hold targets; no zero-torque MIT frame was sent.\n";
    } catch (const std::exception& error) { std::cerr << "MIT STOP WARNING: " << error.what() << "\n"; }
    mit_mode_entered = false;
  };
  try {
    g_mit_stop_requested = 0;
    std::signal(SIGINT, request_mit_stop);
    std::cout << "WARNING: supported seven-axis MIT gravity test, scale=" << gravity_scale
              << ", torque hard limits=[" << limits.max_torque_nm.transpose() << "]Nm, duration<=" << seconds
              << "s, first frame is precharged G(q) plus Kp=2.0, Kd=0.40.\n";
    can.send_mode_mit(10, true);
    mit_mode_entered = true;
    // The first frame after entering MIT already carries gravity support.
    for (std::uint8_t joint = 1; joint <= 7; ++joint) {
      can.send_joint_mit(joint, {.position_rad = hold.position[joint - 1], .velocity_rad_s = 0.0,
                                 .kp_nm_rad = 2.0, .kd_nm_s_rad = 0.40,
                                 .feedforward_torque_nm = initial_torque[joint - 1]});
    }
    const auto deadline = xrobot_cpp::Clock::now() + std::chrono::duration_cast<xrobot_cpp::Clock::duration>(std::chrono::duration<double>(seconds));
    auto next_cycle = xrobot_cpp::Clock::now();
    std::size_t sent_cycles = 1;
    while (xrobot_cpp::Clock::now() < deadline && !g_mit_stop_requested) {
      for (int read_index = 0; read_index < 32; ++read_index) if (!can.read(read_index == 0 ? std::chrono::milliseconds(2) : std::chrono::milliseconds(0))) break;
      const auto now = xrobot_cpp::Clock::now();
      const auto now_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
      feedback = can.feedback();
      const xrobot_cpp::Vec7 velocity = velocity_estimator.update(feedback.joints, now);
      const xrobot_cpp::ComputedTorqueCommand raw_command = controller.compute(model, feedback.joints, velocity, hold);
      const xrobot_cpp::Vec7 torque = gravity_scale * raw_command.feedforward_torque_nm;
      safety.notify_keepalive(now_ns);
      std::string reason;
      if (!safety.permits(hold, velocity, torque, feedback, now_ns, &reason)) throw std::runtime_error("seven-axis MIT safety stop: " + reason);
      for (std::uint8_t joint = 1; joint <= 7; ++joint) {
        can.send_joint_mit(joint, {.position_rad = hold.position[joint - 1], .velocity_rad_s = 0.0, .kp_nm_rad = 2.0, .kd_nm_s_rad = 0.40, .feedforward_torque_nm = torque[joint - 1]});
      }
      ++sent_cycles;
      next_cycle += std::chrono::milliseconds(10);
      std::this_thread::sleep_until(next_cycle);
    }
    restore_joint_mode();
    std::cout << "Seven-axis gravity hold completed: sent_cycles=" << sent_cycles
              << " interrupted=" << (g_mit_stop_requested ? "true" : "false") << ".\n";
  } catch (...) { restore_joint_mode(); throw; }
  return EXIT_SUCCESS;
}

int run_mit_j7_scurve_step(int argc, char** argv) {
  bool unlocked = false, dry_run = false;
  double delta_rad = 0.0, seconds = 1.0;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--arm-command-unlock") unlocked = true;
    else if (argument == "--dry-run") dry_run = true;
    else if (argument == "--delta-rad" && index + 1 < argc) delta_rad = std::stod(argv[++index]);
    else if (argument == "--seconds" && index + 1 < argc) seconds = std::stod(argv[++index]);
    else throw std::invalid_argument("usage: --mit-j7-scurve-step --arm-command-unlock --delta-rad +/-0.01 [--seconds <=1] [--dry-run]");
  }
  if (!unlocked) throw std::invalid_argument("refusing MIT transmit: --arm-command-unlock is required");
  if (!std::isfinite(delta_rad) || std::abs(delta_rad) < 1e-6 || std::abs(delta_rad) > 0.01) throw std::invalid_argument("--delta-rad must be non-zero and within +/-0.01 rad");
  if (!std::isfinite(seconds) || seconds <= 0.0 || seconds > 2.0) throw std::invalid_argument("--seconds must be in (0, 2]");
  if (dry_run) {
    std::cout << "J7 S-curve dry run passed: delta=" << delta_rad << "rad, v_max=0.20rad/s, torque_limit=0.20Nm. No SocketCAN interface was opened.\n";
    return EXIT_SUCCESS;
  }
  const std::string urdf = std::string(XROBOT_CPP_SOURCE_DIR) + "/../xrobot/third_party/agx_arm_urdf/nero/urdf/nero_description.urdf";
  xrobot_cpp::NeroDynamicsModel model(urdf);
  xrobot_cpp::NeroTorqueSafetyLimits limits;
  limits.max_velocity_rad_s = xrobot_cpp::Vec7::Constant(0.20);
  limits.max_torque_nm = xrobot_cpp::Vec7::Constant(0.20);
  limits.max_tracking_error_rad = 0.02;
  xrobot_cpp::NeroTorqueSafetyGate safety(model.lower_limits(), model.upper_limits(), 0.25, limits);
  xrobot_cpp::ComputedTorqueController controller(
      xrobot_cpp::Vec7::Constant(kNeroLowSpeedSpringStiffnessNmRad),
      xrobot_cpp::Vec7::Constant(kNeroLowSpeedDampingNmSRad), limits.max_torque_nm);
  xrobot_cpp::JointVelocityEstimator velocity_estimator(12.0);
  xrobot_cpp::NeroSocketCan can{"can0"};
  can.open_read_only();
  xrobot_cpp::NeroFeedback feedback;
  const auto feedback_deadline = xrobot_cpp::Clock::now() + std::chrono::seconds(2);
  while (xrobot_cpp::Clock::now() < feedback_deadline) {
    (void)can.read(std::chrono::milliseconds(20));
    feedback = can.feedback();
    const auto now = xrobot_cpp::Clock::now();
    const auto now_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
    safety.notify_keepalive(now_ns);
    if (safety.observe(feedback, now_ns) == xrobot_cpp::NeroSafetyState::kReady) break;
  }
  const auto ready_now = xrobot_cpp::Clock::now();
  const auto ready_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(ready_now.time_since_epoch()).count());
  safety.notify_keepalive(ready_ns);
  if (safety.observe(feedback, ready_ns) != xrobot_cpp::NeroSafetyState::kReady) throw std::runtime_error("J7 S-curve test blocked: fresh, enabled, fault-free feedback is required");
  velocity_estimator.reset(feedback.joints, ready_now);
  xrobot_cpp::JerkLimitedJointPlanner trajectory(
      xrobot_cpp::Vec7::Constant(0.20), xrobot_cpp::Vec7::Constant(0.50), xrobot_cpp::Vec7::Constant(5.0));
  trajectory.reset(feedback.joints);
  xrobot_cpp::Vec7 goal = feedback.joints;
  goal[6] += delta_rad;
  if (goal[6] < model.lower_limits()[6] || goal[6] > model.upper_limits()[6]) throw std::runtime_error("J7 S-curve goal exceeds URDF joint limit");
  trajectory.set_goal(goal);
  xrobot_cpp::JointTrajectoryState desired = trajectory.state();
  bool mit_mode_entered = false;
  auto restore_joint_mode = [&] {
    if (!mit_mode_entered) return;
    try {
      can.send_mode_joint(10, true);
      const xrobot_cpp::Vec7 hold_target = feedback.joints;
      for (int repeat = 0; repeat < 3; ++repeat) can.send_joint_target(hold_target);
      std::cout << "MIT STOP: restored joint mode and sent three measured-position hold targets; no zero-torque MIT frame was sent.\n";
    } catch (const std::exception& error) { std::cerr << "MIT STOP WARNING: " << error.what() << "\n"; }
    mit_mode_entered = false;
  };
  try {
    g_mit_stop_requested = 0;
    std::signal(SIGINT, request_mit_stop);
    std::cout << "WARNING: J7 S-curve MIT step delta=" << delta_rad << "rad, v_max=0.20rad/s, duration<=" << seconds << "s.\n";
    can.send_mode_mit(10, true);
    mit_mode_entered = true;
    const auto deadline = xrobot_cpp::Clock::now() + std::chrono::duration_cast<xrobot_cpp::Clock::duration>(std::chrono::duration<double>(seconds));
    auto next_cycle = xrobot_cpp::Clock::now();
    std::size_t sent_cycles = 0;
    while (xrobot_cpp::Clock::now() < deadline && !g_mit_stop_requested) {
      for (int read_index = 0; read_index < 32; ++read_index) if (!can.read(read_index == 0 ? std::chrono::milliseconds(2) : std::chrono::milliseconds(0))) break;
      const auto now = xrobot_cpp::Clock::now();
      const auto now_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
      feedback = can.feedback();
      const xrobot_cpp::Vec7 velocity = velocity_estimator.update(feedback.joints, now);
      desired = trajectory.step(0.010);
      const xrobot_cpp::ComputedTorqueCommand command = controller.compute(model, feedback.joints, velocity, desired);
      safety.notify_keepalive(now_ns);
      std::string reason;
      if (!safety.permits(desired, velocity, command.feedforward_torque_nm, feedback, now_ns, &reason)) throw std::runtime_error("J7 S-curve safety stop: " + reason);
      for (std::uint8_t joint = 1; joint <= 7; ++joint) can.send_joint_mit(joint, {.position_rad = desired.position[joint - 1], .velocity_rad_s = desired.velocity[joint - 1], .kp_nm_rad = 0.0, .kd_nm_s_rad = 0.0, .feedforward_torque_nm = command.feedforward_torque_nm[joint - 1]});
      ++sent_cycles;
      next_cycle += std::chrono::milliseconds(10);
      std::this_thread::sleep_until(next_cycle);
    }
    restore_joint_mode();
    std::cout << "J7 S-curve MIT step completed: sent_cycles=" << sent_cycles << " final_target_j7=" << desired.position[6]
              << " interrupted=" << (g_mit_stop_requested ? "true" : "false") << ".\n";
  } catch (...) { restore_joint_mode(); throw; }
  return EXIT_SUCCESS;
}

int run_mit_cartesian_scurve_step(int argc, char** argv) {
  bool unlocked = false, dry_run = false, hold_current = false, supported_arm = false;
  std::string axis;
  double delta_m = 0.0, seconds = 1.0, settle_seconds = 0.0, impedance_kp_nm_rad = 6.0, impedance_kd_nm_s_rad = 0.80;
  std::filesystem::path telemetry_path;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--arm-command-unlock") unlocked = true;
    else if (argument == "--dry-run") dry_run = true;
    else if (argument == "--hold-current") hold_current = true;
    else if (argument == "--supported-arm") supported_arm = true;
    else if (argument == "--telemetry-log" && index + 1 < argc) telemetry_path = argv[++index];
    else if (argument == "--axis" && index + 1 < argc) axis = argv[++index];
    else if (argument == "--delta-m" && index + 1 < argc) delta_m = std::stod(argv[++index]);
    else if (argument == "--seconds" && index + 1 < argc) seconds = std::stod(argv[++index]);
    else if (argument == "--settle-seconds" && index + 1 < argc) settle_seconds = std::stod(argv[++index]);
    else if (argument == "--kp" && index + 1 < argc) impedance_kp_nm_rad = std::stod(argv[++index]);
    else if (argument == "--kd" && index + 1 < argc) impedance_kd_nm_s_rad = std::stod(argv[++index]);
    else throw std::invalid_argument("usage: --mit-cartesian-scurve-step --arm-command-unlock --supported-arm --axis x|y|z --delta-m +/-0.02 [--hold-current for zero displacement] [--seconds trajectory-window] [--settle-seconds S] [--kp K<=20] [--kd D<=2] [--telemetry-log PATH] [--dry-run]");
  }
  if (!unlocked) throw std::invalid_argument("refusing MIT transmit: --arm-command-unlock is required");
  if (!dry_run && !supported_arm) throw std::invalid_argument("refusing Cartesian MIT test: physically support the arm and pass --supported-arm");
  if (axis != "x" && axis != "y" && axis != "z") throw std::invalid_argument("--axis must be x, y, or z in the fixed NERO base frame");
  if (!std::isfinite(delta_m) || std::abs(delta_m) > 0.02 ||
      (!hold_current && std::abs(delta_m) < 1e-6) ||
      (hold_current && std::abs(delta_m) >= 1e-6)) {
    throw std::invalid_argument(hold_current ? "--hold-current requires --delta-m 0" : "--delta-m must be non-zero and within +/-0.02 m");
  }
  if (!std::isfinite(seconds) || seconds <= 0.0 || seconds > 2.0 || !std::isfinite(settle_seconds) || settle_seconds < 0.0 || seconds + settle_seconds > 4.0) {
    throw std::invalid_argument("--seconds must be in (0, 2], --settle-seconds must be non-negative, and their sum must be <=4");
  }
  if (!std::isfinite(impedance_kp_nm_rad) || impedance_kp_nm_rad <= 0.0 || impedance_kp_nm_rad > 20.0 ||
      !std::isfinite(impedance_kd_nm_s_rad) || impedance_kd_nm_s_rad <= 0.0 || impedance_kd_nm_s_rad > 2.0) {
    throw std::invalid_argument("--kp must be in (0, 20] and --kd must be in (0, 2]");
  }
  if (dry_run) {
    std::cout << "Cartesian S-curve dry run passed: axis=" << axis << " delta=" << delta_m
              << "m, settle=" << settle_seconds << "s, TCP orientation locked, v_joint_max=0.20rad/s, torque_limits=[1,6,1,3,1,1,1]Nm. No SocketCAN interface was opened.\n";
    return EXIT_SUCCESS;
  }
  const std::string urdf = std::string(XROBOT_CPP_SOURCE_DIR) + "/assets/urdf/nero_control_tcp.urdf";
  xrobot_cpp::NeroDynamicsModel model(urdf, "base_link", "gripper_tcp");
  xrobot_cpp::NeroTorqueSafetyLimits limits;
  limits.max_velocity_rad_s = xrobot_cpp::Vec7::Constant(3.00);
  limits.max_torque_nm = mit_support_torque_limits();
  limits.max_tracking_error_rad = 0.25;
  xrobot_cpp::Vec7 safety_lower = model.lower_limits();
  xrobot_cpp::Vec7 safety_upper = model.upper_limits();
  safety_lower[6] -= 0.25;  // J7 measured-state allowance; IK targets retain vendor URDF limits.
  safety_upper[6] += 0.25;
  xrobot_cpp::NeroTorqueSafetyGate safety(safety_lower, safety_upper, 0.25, limits);
  xrobot_cpp::JointVelocityEstimator velocity_estimator(2.0);
  xrobot_cpp::NeroSocketCan can{"can0"};
  can.open_read_only();
  xrobot_cpp::NeroFeedback feedback;
  const auto feedback_deadline = xrobot_cpp::Clock::now() + std::chrono::seconds(2);
  while (xrobot_cpp::Clock::now() < feedback_deadline) {
    (void)can.read(std::chrono::milliseconds(20));
    feedback = can.feedback();
    const auto now = xrobot_cpp::Clock::now();
    const auto now_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
    safety.notify_keepalive(now_ns);
    if (safety.observe(feedback, now_ns) == xrobot_cpp::NeroSafetyState::kReady) break;
  }
  const auto ready_now = xrobot_cpp::Clock::now();
  const auto ready_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(ready_now.time_since_epoch()).count());
  safety.notify_keepalive(ready_ns);
  if (safety.observe(feedback, ready_ns) != xrobot_cpp::NeroSafetyState::kReady) throw std::runtime_error("Cartesian S-curve test blocked: fresh, enabled, fault-free feedback is required");
  const xrobot_cpp::Pose tcp_start = model.forward_kinematics(feedback.joints);
  xrobot_cpp::Pose tcp_target = tcp_start;
  const int axis_index = axis == "x" ? 0 : (axis == "y" ? 1 : 2);
  if (!hold_current) tcp_target.position[axis_index] += delta_m;
  const int waypoint_count = hold_current ? 1 : std::max(1, static_cast<int>(std::ceil(std::abs(delta_m) / 0.003)));
  xrobot_cpp::Vec7 goal = feedback.joints;
  for (int waypoint_index = 1; waypoint_index <= waypoint_count; ++waypoint_index) {
    xrobot_cpp::Pose waypoint = tcp_start;
    waypoint.position = tcp_start.position + (tcp_target.position - tcp_start.position) * (static_cast<double>(waypoint_index) / waypoint_count);
    for (int iteration = 0; iteration < 96; ++iteration) {
      const xrobot_cpp::Pose achieved = model.forward_kinematics(goal);
      if ((waypoint.position - achieved.position).norm() <= 0.0005) break;
      goal = model.solve_ik_step(waypoint, goal, 0.015);
    }
    if ((waypoint.position - model.forward_kinematics(goal).position).norm() > 0.003) {
      // The bounded resolved-rate iteration can stall after a limit clamp; retry this local waypoint with KDL position IK.
      goal = model.solve_ik(waypoint, goal, 2.0 * std::numbers::pi / 180.0);
    }
    if ((waypoint.position - model.forward_kinematics(goal).position).norm() > 0.003) {
      throw std::runtime_error("Cartesian target rejected: a waypoint did not reach 3 mm TCP accuracy");
    }
  }
  const double tcp_goal_error_m = (tcp_target.position - model.forward_kinematics(goal).position).norm();
  const double max_joint_delta = (goal - feedback.joints).cwiseAbs().maxCoeff();
  if (max_joint_delta > 3.50) throw std::runtime_error("Cartesian target rejected: continuous IK requires a joint change greater than 3.50 rad");
  const auto feedback_stamp = feedback.joint_state_monotonic_ns ? xrobot_cpp::Clock::time_point(std::chrono::nanoseconds(feedback.joint_state_monotonic_ns)) : ready_now;
  velocity_estimator.reset(feedback.joints, feedback_stamp);
  std::uint64_t last_joint_state_sequence = feedback.joint_state_sequence;
  xrobot_cpp::Vec7 filtered_velocity = xrobot_cpp::Vec7::Zero();
  // Precharge the first MIT frame with gravity compensation. Ramping from zero
  // would leave a hanging joint unsupported for more than one second at J2.
  xrobot_cpp::Vec7 applied_torque = model.dynamics(feedback.joints, xrobot_cpp::Vec7::Zero()).gravity
      .cwiseMax(-limits.max_torque_nm).cwiseMin(limits.max_torque_nm);
  xrobot_cpp::Vec7 applied_impedance_scale = xrobot_cpp::Vec7::Zero();
  xrobot_cpp::Vec7 gravity_trim_nm = xrobot_cpp::Vec7::Zero();
  const xrobot_cpp::Vec7 max_torque_delta_per_cycle = xrobot_cpp::Vec7::Constant(0.03);
  constexpr double kImpedanceGainRisePerCycle = 0.025;
  xrobot_cpp::JerkLimitedJointPlanner trajectory(
      xrobot_cpp::Vec7::Constant(0.20), xrobot_cpp::Vec7::Constant(0.50), xrobot_cpp::Vec7::Constant(5.0));
  trajectory.reset(feedback.joints);
  trajectory.set_goal(goal);
  xrobot_cpp::JointTrajectoryState desired = trajectory.state();
  std::string initial_reason;
  if (!safety.permits(desired, xrobot_cpp::Vec7::Zero(), applied_torque, feedback, ready_ns, &initial_reason)) {
    throw std::runtime_error("Cartesian MIT test blocked: " + initial_reason);
  }
  std::ofstream telemetry;
  if (!telemetry_path.empty()) {
    if (!telemetry_path.parent_path().empty()) std::filesystem::create_directories(telemetry_path.parent_path());
    telemetry.open(telemetry_path);
    if (!telemetry) throw std::runtime_error("cannot open MIT telemetry: " + telemetry_path.string());
  }
  bool mit_mode_entered = false;
  auto restore_joint_mode = [&] {
    if (!mit_mode_entered) return;
    try {
      // Do not send an all-zero MIT frame before mode handoff: it removes the
      // virtual spring, damper, and gravity support for one driver cycle.
      // The vendor API switches to MOVE J first, then sends the hold target.
      can.send_mode_joint(10, true);
      const xrobot_cpp::Vec7 hold_target = feedback.joints;
      for (int repeat = 0; repeat < 3; ++repeat) can.send_joint_target(hold_target);
      std::cout << "MIT STOP: restored joint mode and sent three measured-position hold targets; no zero-torque MIT frame was sent.\n";
    } catch (const std::exception& error) { std::cerr << "MIT STOP WARNING: " << error.what() << "\n"; }
    mit_mode_entered = false;
  };
  try {
    g_mit_stop_requested = 0;
    std::signal(SIGINT, request_mit_stop);
    std::cout << "WARNING: Cartesian MIT " << (hold_current ? "current-pose hold" : "S-curve step") << " base_" << axis << " delta=" << delta_m
              << "m, orientation locked, joint v_max=0.20rad/s, measured velocity limit=3.00rad/s, tracking limit=0.25rad, K=" << impedance_kp_nm_rad << "Nm/rad, D=" << impedance_kd_nm_s_rad << "Nms/rad, torque slew=0.03Nm/10ms, torque hard limits=[1,6,1,3,1,1,1]Nm, trajectory window=" << seconds << "s, settle window=" << settle_seconds << "s; then MOVE J measured-position hold is restored.\n";
    can.send_mode_mit(10, true);
    mit_mode_entered = true;
    // The first frame after entering MIT carries static gravity support.
    for (std::uint8_t joint = 1; joint <= 7; ++joint) {
      can.send_joint_mit(joint, {.position_rad = desired.position[joint - 1], .velocity_rad_s = 0.0,
                                 .kp_nm_rad = 2.0, .kd_nm_s_rad = 0.40,
                                 .feedforward_torque_nm = applied_torque[joint - 1]});
    }
    const auto session_deadline = xrobot_cpp::Clock::now() + std::chrono::duration_cast<xrobot_cpp::Clock::duration>(std::chrono::duration<double>(seconds + settle_seconds));
    auto next_cycle = xrobot_cpp::Clock::now();
    std::size_t sent_cycles = 1;
    bool completed_schedule = false;
    while (!g_mit_stop_requested) {
      for (int read_index = 0; read_index < 32; ++read_index) if (!can.read(read_index == 0 ? std::chrono::milliseconds(2) : std::chrono::milliseconds(0))) break;
      const auto now = xrobot_cpp::Clock::now();
      const auto now_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
      feedback = can.feedback();
      const xrobot_cpp::Vec7 velocity = [&] {
        if (feedback.joint_state_sequence != last_joint_state_sequence) {
          const auto frame_stamp = feedback.joint_state_monotonic_ns ? xrobot_cpp::Clock::time_point(std::chrono::nanoseconds(feedback.joint_state_monotonic_ns)) : now;
          filtered_velocity = velocity_estimator.update(feedback.joints, frame_stamp);
          last_joint_state_sequence = feedback.joint_state_sequence;
        }
        return filtered_velocity;
      }();
      desired = trajectory.step(0.010);
      const xrobot_cpp::DynamicsTerms dynamics = model.dynamics(feedback.joints, velocity);
      const xrobot_cpp::Vec7 position_error = desired.position - feedback.joints;
      const xrobot_cpp::Vec7 velocity_error = desired.velocity - velocity;
      update_gravity_trim(&gravity_trim_nm, position_error, velocity, 0.010);

      // Use G(q_measured) only for real-arm feedforward.  Desired trajectory
      // acceleration and Coriolis terms remain outside t_ff pending physical
      // torque calibration; native MIT Kp/Kd supplies the low impedance.
      const xrobot_cpp::Vec7 gravity = (nominal_gravity_feedforward(dynamics, limits.max_torque_nm) + gravity_trim_nm)
          .cwiseMax(-limits.max_torque_nm).cwiseMin(limits.max_torque_nm);
      const xrobot_cpp::Vec7 torque = applied_torque +
          (gravity - applied_torque)
              .cwiseMax(-max_torque_delta_per_cycle)
              .cwiseMin(max_torque_delta_per_cycle);
      applied_torque = torque;

      // Reserve feedforward headroom for the driver-side virtual spring and damper.
      const xrobot_cpp::Vec7 impedance_torque =
          xrobot_cpp::Vec7::Constant(impedance_kp_nm_rad).cwiseProduct(position_error) +
          xrobot_cpp::Vec7::Constant(impedance_kd_nm_s_rad).cwiseProduct(velocity_error);
      xrobot_cpp::Vec7 headroom_scale = xrobot_cpp::Vec7::Ones();
      for (Eigen::Index index = 0; index < 7; ++index) {
        const double available = std::max(0.0, limits.max_torque_nm[index] - std::abs(torque[index]));
        if (std::abs(impedance_torque[index]) > available && std::abs(impedance_torque[index]) > 1e-9) {
          headroom_scale[index] = available / std::abs(impedance_torque[index]);
        }
        // Drop gain immediately when torque headroom shrinks, but ramp it up
        // over 0.4 s after entering MIT mode or recovering from saturation.
        applied_impedance_scale[index] = headroom_scale[index] < applied_impedance_scale[index]
            ? headroom_scale[index]
            : std::min(headroom_scale[index], applied_impedance_scale[index] + kImpedanceGainRisePerCycle);
      }
      const xrobot_cpp::Vec7 kp = xrobot_cpp::Vec7::Constant(impedance_kp_nm_rad).cwiseProduct(applied_impedance_scale);
      const xrobot_cpp::Vec7 kd = xrobot_cpp::Vec7::Constant(impedance_kd_nm_s_rad).cwiseProduct(applied_impedance_scale);
      const xrobot_cpp::Vec7 predicted_total_torque = torque + applied_impedance_scale.cwiseProduct(impedance_torque);
      write_mit_telemetry(&telemetry, now_ns, feedback, velocity, desired, dynamics, torque, kp, kd,
                          hold_current ? "cartesian_current_pose_hold" : "cartesian_scurve");
      safety.notify_keepalive(now_ns);
      std::string reason;
      if (!safety.permits(desired, velocity, predicted_total_torque, feedback, now_ns, &reason)) throw std::runtime_error("Cartesian S-curve safety stop: " + reason);
      for (std::uint8_t joint = 1; joint <= 7; ++joint) {
        const Eigen::Index index = static_cast<Eigen::Index>(joint - 1);
        can.send_joint_mit(joint, {.position_rad = desired.position[index], .velocity_rad_s = desired.velocity[index],
                                   .kp_nm_rad = kp[index],
                                   .kd_nm_s_rad = kd[index],
                                   .feedforward_torque_nm = torque[index]});
      }
      ++sent_cycles;
      if (now >= session_deadline) {
        completed_schedule = true;
        break;
      }
      next_cycle += std::chrono::milliseconds(10);
      std::this_thread::sleep_until(next_cycle);
    }
    restore_joint_mode();
    std::cout << "Cartesian torque session " << (completed_schedule ? "completed scheduled window" : "stopped by explicit operator request")
              << ": sent_cycles=" << sent_cycles << " base_" << axis
              << "_target_delta_m=" << delta_m << " tcp_goal_error_m=" << tcp_goal_error_m << " max_ik_joint_delta_rad=" << max_joint_delta
              << " interrupted=" << (g_mit_stop_requested ? "true" : "false") << ".\n";
  } catch (...) { restore_joint_mode(); throw; }
  return EXIT_SUCCESS;
}

int run_real_control_preflight(int argc, char** argv) {
  double seconds = 5.0;
  std::string interface_name{"can0"};
  bool bootstrap_feedback = false;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--seconds" && index + 1 < argc) seconds = std::stod(argv[++index]);
    else if (argument == "--can" && index + 1 < argc) interface_name = argv[++index];
    else if (argument == "--bootstrap-feedback") bootstrap_feedback = true;
    else throw std::invalid_argument(
        "usage: --real-control-preflight [--seconds S] [--can can0] [--bootstrap-feedback]");
  }
  if (!std::isfinite(seconds) || seconds <= 0.0 || seconds > 30.0) {
    throw std::invalid_argument("--seconds must be in (0, 30]");
  }
  const std::string urdf = std::string(XROBOT_CPP_SOURCE_DIR) + "/../xrobot/third_party/agx_arm_urdf/nero/urdf/nero_description.urdf";
  xrobot_cpp::NeroDynamicsModel model(urdf);
  xrobot_cpp::NeroTorqueSafetyLimits limits;
  limits.max_velocity_rad_s = xrobot_cpp::Vec7::Constant(0.50);
  limits.max_torque_nm = xrobot_cpp::Vec7::Constant(2.00);
  limits.max_tracking_error_rad = 0.05;
  xrobot_cpp::NeroTorqueSafetyGate safety(model.lower_limits(), model.upper_limits(), 0.25, limits);
  xrobot_cpp::JointVelocityEstimator velocity_estimator(12.0);
  xrobot_cpp::ComputedTorqueController controller(xrobot_cpp::Vec7::Constant(3.0), xrobot_cpp::Vec7::Constant(0.5), limits.max_torque_nm);
  xrobot_cpp::NeroSocketCan can(interface_name);
  can.open_read_only();
  std::cout << "Real-control preflight on " << interface_name;
  if (bootstrap_feedback) {
    std::cout << ": applying the same non-motion feedback bootstrap as pyAgxArm "
              << "set_normal_mode() (0x470, then 0x151). No enable, MIT, joint, or gripper CAN frame will be sent.\n";
    can.send_normal_single_arm_config();
    can.send_normal_mode_feedback(50);
    const auto bootstrap_deadline = xrobot_cpp::Clock::now() + std::chrono::milliseconds(500);
    while (xrobot_cpp::Clock::now() < bootstrap_deadline) {
      (void)can.read(std::chrono::milliseconds(20));
    }
  } else {
    std::cout << ": receive-only. No mode, enable, MIT, joint, or gripper CAN frame will be sent.\n";
  }
  const auto deadline = xrobot_cpp::Clock::now() + std::chrono::duration_cast<xrobot_cpp::Clock::duration>(std::chrono::duration<double>(seconds));
  std::size_t permitted_cycles = 0;
  std::size_t rejected_cycles = 0;
  std::string last_reason;
  auto last_report = xrobot_cpp::Clock::now();
  while (xrobot_cpp::Clock::now() < deadline) {
    (void)can.read(std::chrono::milliseconds(10));
    const auto now = xrobot_cpp::Clock::now();
    const auto now_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
    const xrobot_cpp::NeroFeedback feedback = can.feedback();
    safety.notify_keepalive(now_ns);
    const xrobot_cpp::Vec7 measured_velocity = velocity_estimator.update(feedback.joints, now);
    xrobot_cpp::JointTrajectoryState hold{.position = feedback.joints, .velocity = xrobot_cpp::Vec7::Zero(), .acceleration = xrobot_cpp::Vec7::Zero()};
    const xrobot_cpp::ComputedTorqueCommand command = controller.compute(model, feedback.joints, measured_velocity, hold);
    std::string reason;
    if (safety.permits(hold, measured_velocity, command.feedforward_torque_nm, feedback, now_ns, &reason)) {
      ++permitted_cycles;
    } else {
      ++rejected_cycles;
      last_reason = reason;
    }
    if (now - last_report >= std::chrono::seconds(1)) {
      std::cout << "PRECHECK q_valid=" << std::count(feedback.joint_valid.begin(), feedback.joint_valid.end(), true)
                << " driver_valid=" << std::count(feedback.driver_valid.begin(), feedback.driver_valid.end(), true)
                << " enabled=" << std::count(feedback.joint_enabled.begin(), feedback.joint_enabled.end(), true)
                << " faults=" << std::count(feedback.joint_fault.begin(), feedback.joint_fault.end(), true)
                << " state=" << static_cast<int>(safety.observe(feedback, now_ns));
      if (!last_reason.empty()) std::cout << " reason=\"" << last_reason << "\"";
      std::cout << std::endl;
      last_report = now;
    }
  }
  std::cout << "Real-control preflight completed: permitted_cycles=" << permitted_cycles
            << " rejected_cycles=" << rejected_cycles;
  if (!last_reason.empty()) std::cout << " last_rejection=\"" << last_reason << "\"";
  std::cout << (bootstrap_feedback ? ". Only the two pyAgxArm feedback-bootstrap frames were sent.\n"
                                  : ". No CAN command was sent.\n");
  if (permitted_cycles == 0 || rejected_cycles != 0) {
    throw std::runtime_error("real-control safety preflight did not remain continuously ready");
  }
  return EXIT_SUCCESS;
}

#ifdef XROBOT_CPP_WITH_XR
#include "xrobot_cpp/xr_client.hpp"
#endif

namespace {
using namespace xrobot_cpp;

int run_demo() {
  const auto start = Clock::now();
  Eigen::Matrix3d alignment;
  alignment << 0.0, -1.0, 0.0,
               1.0, 0.0, 0.0,
               0.0, 0.0, 1.0;
  RelativeMapper mapper(0.7, 1.0, 0.012, 0.05, alignment);
  XrFrame reference;
  reference.sequence = 1;
  reference.timestamp_ns = 1;
  reference.device_id = "demo";
  reference.deadman = true;
  Pose tcp;
  mapper.rebase(reference, tcp, start);

  Vec7 command = Vec7::Zero();
  command[0] = 0.70;

  for (int index = 1; index <= 120; ++index) {
    const auto now = start + std::chrono::milliseconds(index * 1000 / 60);
    XrFrame frame = reference;
    frame.sequence = static_cast<std::uint64_t>(index + 1);
    frame.timestamp_ns = static_cast<std::uint64_t>(index + 1) * 16666667ULL;
    frame.controller.position = Vec3(0.0, 0.10 * static_cast<double>(index) / 120.0, 0.0);
    const TcpTarget target = mapper.target(frame, now);
    if (index % 30 == 0) {
      std::cout << "demo t=" << std::fixed << std::setprecision(2)
                << static_cast<double>(index) / 60.0
                << " tcp_xyz=[" << target.pose.position.transpose() << "]"
                << " q0=" << command[0] << "\n";
    }
  }
  std::cout << "C++ control-core demo completed. No CAN or NERO command was opened.\n";
  return EXIT_SUCCESS;
}

#ifdef XROBOT_CPP_WITH_XR
int run_xr_monitor() {
  XrClient client;
  client.start();
  std::cout << "XR monitor started. Ctrl+C to stop. No CAN or NERO command is opened.\n";
  while (true) {
    if (const auto frame = client.read_fresh()) {
      std::cout << "XR seq=" << frame->sequence << " A=" << frame->deadman
                << " p=[" << std::fixed << std::setprecision(4) << frame->controller.position.transpose() << "]"
                << " q=[" << frame->controller.orientation.coeffs().transpose() << "]"
                << " trigger=" << std::setprecision(3) << frame->trigger
                << " grip=" << frame->grip << "\n";
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

xrobot_cpp::GripperFeedback await_gripper_feedback(xrobot_cpp::NeroSocketCan* can,
                                                    std::chrono::milliseconds timeout,
                                                    std::uint64_t newer_than_ns = 0) {
  if (can == nullptr) throw std::invalid_argument("gripper CAN transport is null");
  const auto deadline = xrobot_cpp::Clock::now() + timeout;
  while (xrobot_cpp::Clock::now() < deadline) {
    (void)can->read(std::chrono::milliseconds(20));
    const xrobot_cpp::GripperFeedback feedback = can->gripper_feedback();
    if (feedback.monotonic_ns > newer_than_ns) return feedback;
  }
  throw std::runtime_error("no 0x2A8 gripper feedback received");
}

void print_gripper_status(const xrobot_cpp::GripperFeedback& feedback) {
  std::cout << "Gripper status: width_m=" << feedback.width_m
            << " force_n=" << feedback.force_n
            << " enabled=" << (feedback.enabled ? "true" : "false")
            << " homed=" << (feedback.homed ? "true" : "false")
            << " fault=" << (feedback.fault ? "true" : "false")
            << " mode=" << (feedback.mode == 0x00U ? "width" : feedback.mode == 0x01U ? "angle" : "unknown")
            << " status_code=" << static_cast<unsigned>(feedback.status_code) << "\n";
}

int run_gripper_maintenance(int argc, char** argv) {
  enum class Action { kStatus, kEnable, kReset, kRelease, kConfigureStroke, kSetZero };
  std::optional<Action> action;
  if (argc > 1) {
    const std::string_view command(argv[1]);
    if (command == "--gripper-status") action = Action::kStatus;
    else if (command == "--gripper-enable") action = Action::kEnable;
    else if (command == "--gripper-reset") action = Action::kReset;
    else if (command == "--gripper-release") action = Action::kRelease;
    else if (command == "--gripper-configure-stroke") action = Action::kConfigureStroke;
    else if (command == "--gripper-set-zero") action = Action::kSetZero;
  }
  bool unlocked = false, manually_closed = false;
  int stroke_mm = 0;
  std::string interface_name{"can0"};
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--gripper-status") action = Action::kStatus;
    else if (argument == "--gripper-enable") action = Action::kEnable;
    else if (argument == "--gripper-reset") action = Action::kReset;
    else if (argument == "--gripper-release") action = Action::kRelease;
    else if (argument == "--gripper-configure-stroke") action = Action::kConfigureStroke;
    else if (argument == "--gripper-set-zero") action = Action::kSetZero;
    else if (argument == "--gripper-command-unlock") unlocked = true;
    else if (argument == "--gripper-manually-closed") manually_closed = true;
    else if (argument == "--gripper-stroke-mm" && index + 1 < argc) stroke_mm = std::stoi(argv[++index]);
    else if (argument == "--can" && index + 1 < argc) interface_name = argv[++index];
    else throw std::invalid_argument("usage: --gripper-status|--gripper-enable|--gripper-reset|--gripper-release|--gripper-configure-stroke --gripper-stroke-mm 70|100|--gripper-set-zero [--gripper-command-unlock] [--gripper-manually-closed] [--can can0]");
  }
  if (!action) throw std::invalid_argument("one gripper maintenance action is required");
  if (*action != Action::kStatus && !unlocked) {
    throw std::invalid_argument("refusing gripper command: --gripper-command-unlock is required");
  }
  if (*action == Action::kConfigureStroke && stroke_mm != 70 && stroke_mm != 100) {
    throw std::invalid_argument("gripper stroke configuration requires --gripper-stroke-mm 70 or 100");
  }
  if (*action == Action::kSetZero && !manually_closed) {
    throw std::invalid_argument("refusing gripper zero calibration: manually close it, then pass --gripper-manually-closed");
  }

  xrobot_cpp::NeroSocketCan can(interface_name);
  can.open_read_only();
  const xrobot_cpp::GripperFeedback initial = await_gripper_feedback(&can, std::chrono::seconds(2));
  print_gripper_status(initial);
  if (*action == Action::kStatus) return EXIT_SUCCESS;
  if (initial.fault) throw std::runtime_error("gripper maintenance blocked: current 0x2A8 feedback reports a fault");

  if (*action == Action::kEnable) {
    // This only changes the driver state to width-mode enabled. It retains the
    // measured width as target, so it does not request an opening or closing move.
    can.send_normal_single_arm_config();
    can.send_normal_mode_feedback(50);
    can.send_gripper_control(initial.width_m, 1.0, 0x03U);
    const xrobot_cpp::GripperFeedback ready =
        await_gripper_feedback(&can, std::chrono::seconds(2), initial.monotonic_ns);
    if (ready.fault) throw std::runtime_error("gripper enable reported a fault");
    if (!ready.enabled || ready.mode != 0x00U) {
      throw std::runtime_error("gripper did not acknowledge enabled width mode");
    }
    std::cout << "Gripper enabled in width mode without a position change. ";
    print_gripper_status(ready);
    return EXIT_SUCCESS;
  }

  if (*action == Action::kReset) {
    const std::uint8_t reset_code = initial.mode == 0x01U ? 0x06U : 0x02U;
    can.send_normal_single_arm_config();
    can.send_normal_mode_feedback(50);
    can.send_gripper_control(0.0, 0.0, reset_code);
    const xrobot_cpp::GripperFeedback reset =
        await_gripper_feedback(&can, std::chrono::seconds(2), initial.monotonic_ns);
    if (reset.fault) throw std::runtime_error("gripper reset reported a fault");
    std::cout << "Gripper reset request completed. ";
    print_gripper_status(reset);
    return EXIT_SUCCESS;
  }

  if (*action == Action::kConfigureStroke) {
    can.send_normal_single_arm_config();
    can.send_normal_mode_feedback(50);
    can.send_gripper_teaching_config(static_cast<std::uint8_t>(stroke_mm));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    can.send_gripper_teaching_query();
    const auto deadline = xrobot_cpp::Clock::now() + std::chrono::seconds(2);
    while (xrobot_cpp::Clock::now() < deadline) {
      const std::optional<xrobot_cpp::CanFrame> frame = can.read(std::chrono::milliseconds(20));
      if (frame && frame->id == 0x47EU && frame->size >= 3) {
        if (frame->data[0] == 100U && frame->data[1] == static_cast<std::uint8_t>(stroke_mm) &&
            frame->data[2] >= 1U && frame->data[2] <= 10U) {
          std::cout << "Gripper stroke configured and verified: " << stroke_mm
                    << " mm, teaching_friction=" << static_cast<unsigned>(frame->data[2]) << "\n";
          return EXIT_SUCCESS;
        }
        std::ostringstream detail;
        detail << "gripper stroke configuration feedback 0x47E does not match the requested settings: "
               << "teaching_range_percent=" << static_cast<unsigned>(frame->data[0])
               << " stroke_mm=" << static_cast<unsigned>(frame->data[1])
               << " teaching_friction=" << static_cast<unsigned>(frame->data[2]);
        throw std::runtime_error(detail.str());
      }
    }
    throw std::runtime_error("gripper stroke configuration was not verified by a fresh 0x47E feedback frame");
  }

  if (*action == Action::kRelease) {
    const std::uint8_t disable_code = initial.mode == 0x01U ? 0x04U : 0x00U;
    can.send_gripper_control(0.0, 0.0, disable_code);
    const auto deadline = xrobot_cpp::Clock::now() + std::chrono::seconds(2);
    while (xrobot_cpp::Clock::now() < deadline) {
      const xrobot_cpp::GripperFeedback feedback = await_gripper_feedback(&can, std::chrono::milliseconds(100));
      if (!feedback.enabled) {
        std::cout << "Gripper released. Manually close it to the mechanical zero position.\n";
        return EXIT_SUCCESS;
      }
    }
    throw std::runtime_error("gripper release was not acknowledged by 0x2A8 feedback");
  }

  // Establish normal CAN control before the one-shot 0x159 zero request.
  // NERO's firmware branch emits no 0x476 acknowledgement for this command;
  // readiness must therefore be verified from a fresh 0x2A8 state instead.
  // These are configuration frames only; no joint target or MIT frame is sent.
  can.send_normal_single_arm_config();
  can.send_normal_mode_feedback(50);
  can.send_gripper_control(0.0, 0.0, 0x00U, 0xAEU);
  const xrobot_cpp::GripperFeedback after_zero =
      await_gripper_feedback(&can, std::chrono::seconds(2), initial.monotonic_ns);
  if (after_zero.fault) throw std::runtime_error("gripper zero calibration reported a fault");
  // 0x03 is the official width-mode enable-and-clear code for first activation.
  can.send_gripper_control(0.0, 1.0, 0x03U);
  const xrobot_cpp::GripperFeedback ready =
      await_gripper_feedback(&can, std::chrono::seconds(2), after_zero.monotonic_ns);
  if (ready.fault) throw std::runtime_error("gripper width-mode enable reported a fault");
  if (ready.enabled && ready.mode == 0x00U) {
    std::cout << "Gripper zero request sent and width mode enabled at 0.0 m. "
              << "NERO firmware does not provide a 0x476 zero acknowledgement.\n";
    return EXIT_SUCCESS;
  }
  throw std::runtime_error("gripper did not enter enabled width mode after zero calibration");
}

int run_real_xr_teleop(int argc, char** argv) {
  bool unlocked = false, supported_arm = false, dry_run = false, xr_gripper_enabled = false, freeze_xr_target = false;
  std::string orientation_mode{"locked"};
  std::filesystem::path absolute_orientation_calibration{"configs/nero_xr_calibration.json"};
  double translation_scale = 0.10, session_radius_m = 0.0, seconds = 0.0;
  double joint_v_max_rad_s = 0.20, joint_a_max_rad_s2 = 0.50, joint_j_max_rad_s3 = 5.0;
  double xr_timeout_s = 2.0, can_feedback_timeout_s = 1.0;
  std::string interface_name{"can0"};
  std::filesystem::path telemetry_path;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--arm-command-unlock") unlocked = true;
    else if (argument == "--supported-arm") supported_arm = true;
    else if (argument == "--orientation-mode" && index + 1 < argc) orientation_mode = argv[++index];
    else if (argument == "--absolute-orientation-calibration" && index + 1 < argc) absolute_orientation_calibration = argv[++index];
    else if (argument == "--dry-run") dry_run = true;
    else if (argument == "--enable-xr-gripper") xr_gripper_enabled = true;
    else if (argument == "--freeze-xr-target") freeze_xr_target = true;
    else if (argument == "--translation-scale" && index + 1 < argc) translation_scale = std::stod(argv[++index]);
    else if (argument == "--session-radius-m" && index + 1 < argc) session_radius_m = std::stod(argv[++index]);
    else if (argument == "--joint-v-max" && index + 1 < argc) joint_v_max_rad_s = std::stod(argv[++index]);
    else if (argument == "--joint-a-max" && index + 1 < argc) joint_a_max_rad_s2 = std::stod(argv[++index]);
    else if (argument == "--joint-j-max" && index + 1 < argc) joint_j_max_rad_s3 = std::stod(argv[++index]);
    else if (argument == "--seconds" && index + 1 < argc) seconds = std::stod(argv[++index]);
    else if (argument == "--xr-timeout-s" && index + 1 < argc) xr_timeout_s = std::stod(argv[++index]);
    else if (argument == "--can-feedback-timeout-s" && index + 1 < argc) can_feedback_timeout_s = std::stod(argv[++index]);
    else if (argument == "--can" && index + 1 < argc) interface_name = argv[++index];
    else if (argument == "--telemetry-log" && index + 1 < argc) telemetry_path = argv[++index];
    else throw std::invalid_argument("usage: --real-xr-teleop --arm-command-unlock --supported-arm [--translation-scale S<=0.70] [--orientation-mode locked|absolute] [--enable-xr-gripper] [--freeze-xr-target] [--absolute-orientation-calibration PATH] [--session-radius-m R<=0.50; 0=disabled] [--joint-v-max V<=0.60] [--joint-a-max A<=1.50] [--joint-j-max J<=12] [--seconds 0=unlimited|S<=30] [--xr-timeout-s T<=10] [--can-feedback-timeout-s T<=5] [--can can0] [--telemetry-log PATH] [--dry-run]");
  }
  if (!unlocked) throw std::invalid_argument("refusing real XR MIT control: --arm-command-unlock is required");
  if (orientation_mode != "locked" && orientation_mode != "absolute") throw std::invalid_argument("--orientation-mode must be locked or absolute");
  const Eigen::Matrix3d controller_to_tcp_rotation = orientation_mode == "absolute"
      ? load_absolute_controller_to_tcp_rotation(absolute_orientation_calibration) : Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d base_from_xr = load_calibrated_base_from_xr_rotation(absolute_orientation_calibration);
  if (!dry_run && !supported_arm) throw std::invalid_argument("refusing real XR MIT control: physically support the arm and pass --supported-arm");
  if (!std::isfinite(translation_scale) || translation_scale <= 0.0 || translation_scale > 0.70) throw std::invalid_argument("--translation-scale must be in (0, 0.70]");
  if (!std::isfinite(session_radius_m) || session_radius_m < 0.0 || session_radius_m > 0.50) throw std::invalid_argument("--session-radius-m must be in [0, 0.50]; 0 disables the session cap");
  if (!std::isfinite(joint_v_max_rad_s) || joint_v_max_rad_s <= 0.0 || joint_v_max_rad_s > 0.60) throw std::invalid_argument("--joint-v-max must be in (0, 0.60]");
  if (!std::isfinite(joint_a_max_rad_s2) || joint_a_max_rad_s2 <= 0.0 || joint_a_max_rad_s2 > 1.50) throw std::invalid_argument("--joint-a-max must be in (0, 1.50]");
  if (!std::isfinite(joint_j_max_rad_s3) || joint_j_max_rad_s3 <= 0.0 || joint_j_max_rad_s3 > 12.0) throw std::invalid_argument("--joint-j-max must be in (0, 12]");
  if (!std::isfinite(seconds) || seconds < 0.0 || seconds > 30.0) throw std::invalid_argument("--seconds must be 0 (unlimited) or in (0, 30]");
  if (!std::isfinite(xr_timeout_s) || xr_timeout_s <= 0.0 || xr_timeout_s > 10.0) throw std::invalid_argument("--xr-timeout-s must be in (0, 10]");
  if (!std::isfinite(can_feedback_timeout_s) || can_feedback_timeout_s <= 0.0 || can_feedback_timeout_s > 5.0) throw std::invalid_argument("--can-feedback-timeout-s must be in (0, 5]");
#ifdef XROBOT_CPP_WITH_PLACO
  constexpr const char* kIkBackend = "Placo-QP";
#else
  constexpr const char* kIkBackend = "KDL-DLS";
#endif
  if (dry_run) {
    std::cout << "Real XR teleoperation dry run passed: TCP orientation=" << orientation_mode
              << ", IK backend=" << kIkBackend
              << ", translation_scale=" << translation_scale
              << ", session displacement cap=" << (session_radius_m == 0.0 ? std::string("disabled") : std::to_string(session_radius_m) + "m")
              << ", xr_timeout=" << xr_timeout_s << "s, CAN feedback timeout=" << can_feedback_timeout_s
              << "s, gripper=" << (xr_gripper_enabled ? "enabled" : "disabled")
              << ", target=" << (freeze_xr_target ? "frozen" : "tracking") << ", joint_v_max=" << joint_v_max_rad_s << "rad/s, duration="
              << (seconds == 0.0 ? std::string("unlimited") : std::to_string(seconds) + "s")
              << ". No XR, SocketCAN, mode, MIT, joint, or gripper command was opened.\n";
    return EXIT_SUCCESS;
  }

  const std::string urdf = std::string(XROBOT_CPP_SOURCE_DIR) + "/assets/urdf/nero_control_tcp.urdf";
  xrobot_cpp::NeroDynamicsModel model(urdf, "base_link", "gripper_tcp");
#ifdef XROBOT_CPP_WITH_PLACO
  xrobot_cpp::PlacoNeroIkSolver placo_ik(urdf, "gripper_tcp");
#endif
  xrobot_cpp::NeroTorqueSafetyLimits limits;
  limits.max_velocity_rad_s = xrobot_cpp::Vec7::Constant(1.0);
  limits.max_torque_nm = mit_support_torque_limits();
  limits.max_tracking_error_rad = 0.25;
  xrobot_cpp::Vec7 safety_lower = model.lower_limits();
  xrobot_cpp::Vec7 safety_upper = model.upper_limits();
  // Encoder zero offsets can leave an already stationary physical joint a few
  // hundredths of a radian beyond the URDF model bound. The QP still solves
  // strictly inside the model limits; this tolerance only permits its first
  // trajectory samples to recover inward from the measured state.
  constexpr double kEncoderLimitToleranceRad = 0.10;
  safety_lower.array() -= kEncoderLimitToleranceRad;
  safety_upper.array() += kEncoderLimitToleranceRad;
  xrobot_cpp::NeroTorqueSafetyGate safety(safety_lower, safety_upper, can_feedback_timeout_s, limits);
  xrobot_cpp::NeroSocketCan can(interface_name);
  can.open_read_only();
  xrobot_cpp::NeroFeedback feedback;
  const auto feedback_deadline = xrobot_cpp::Clock::now() + std::chrono::seconds(2);
  while (xrobot_cpp::Clock::now() < feedback_deadline) {
    (void)can.read(std::chrono::milliseconds(20));
    feedback = can.feedback();
    const auto now = xrobot_cpp::Clock::now();
    const auto now_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
    safety.notify_keepalive(now_ns);
    if (safety.observe(feedback, now_ns) == xrobot_cpp::NeroSafetyState::kReady) break;
  }
  const auto ready_now = xrobot_cpp::Clock::now();
  const auto ready_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(ready_now.time_since_epoch()).count());
  safety.notify_keepalive(ready_ns);
  if (safety.observe(feedback, ready_ns) != xrobot_cpp::NeroSafetyState::kReady) throw std::runtime_error("real XR control blocked: fresh, enabled, fault-free feedback is required");

  if (xr_gripper_enabled) {
    // The independent gripper must prove its own 0x2A8 feedback state. Zero
    // calibration is deliberately not implicit in XR teleoperation.
    const auto gripper_deadline = xrobot_cpp::Clock::now() + std::chrono::seconds(2);
    bool gripper_ready = false, initialization_sent = false;
    std::string gripper_reason{"no 0x2A8 gripper feedback"};
    while (xrobot_cpp::Clock::now() < gripper_deadline) {
      (void)can.read(std::chrono::milliseconds(20));
      const xrobot_cpp::GripperFeedback gripper = can.gripper_feedback();
      const auto now = xrobot_cpp::Clock::now();
      const auto now_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
      const double age_s = gripper.monotonic_ns == 0 ? std::numeric_limits<double>::infinity()
          : static_cast<double>(now_ns - gripper.monotonic_ns) * 1e-9;
      if (age_s > can_feedback_timeout_s) {
        gripper_reason = "gripper feedback is stale or absent";
        continue;
      }
      if (gripper.fault) {
        throw std::runtime_error("real XR gripper blocked: 0x2A8 reports a driver fault");
      }
      if (gripper.mode != 0x00U) {
        throw std::runtime_error("real XR gripper blocked: 0x2A8 reports angle mode; switch it to width mode first");
      }
      if (!std::isfinite(gripper.width_m) || gripper.width_m < 0.0 || gripper.width_m > 0.100) {
        throw std::runtime_error("real XR gripper blocked: 0x2A8 reports an invalid width");
      }
      if (!gripper.enabled) {
        if (!initialization_sent) {
          // First activation clears any retained gripper driver state before enabling width mode.
          can.send_gripper_control(gripper.width_m, 1.0, 0x03U);
          initialization_sent = true;
          gripper_reason = "width-mode enable command was not acknowledged";
        }
        continue;
      }
      gripper_ready = true;
      std::cout << "XR gripper ready: width_m=" << gripper.width_m
                << " homed=" << (gripper.homed ? "true" : "unreported") << " mode=width"
                << (initialization_sent ? " (initialized by C++ preflight)" : "") << "\n";
      break;
    }
    if (!gripper_ready) throw std::runtime_error("real XR gripper preflight failed: " + gripper_reason);
  }

  std::ofstream telemetry;
  if (!telemetry_path.empty()) {
    if (!telemetry_path.parent_path().empty()) std::filesystem::create_directories(telemetry_path.parent_path());
    telemetry.open(telemetry_path);
    if (!telemetry) throw std::runtime_error("cannot open real XR telemetry: " + telemetry_path.string());
  }
  xrobot_cpp::JointVelocityEstimator velocity_estimator(1.0);
  const auto feedback_stamp = feedback.joint_state_monotonic_ns ? xrobot_cpp::Clock::time_point(std::chrono::nanoseconds(feedback.joint_state_monotonic_ns)) : ready_now;
  velocity_estimator.reset(feedback.joints, feedback_stamp);
  std::uint64_t last_joint_state_sequence = feedback.joint_state_sequence;
  xrobot_cpp::Vec7 filtered_velocity = xrobot_cpp::Vec7::Zero();
  xrobot_cpp::JerkLimitedJointPlanner trajectory(xrobot_cpp::Vec7::Constant(joint_v_max_rad_s), xrobot_cpp::Vec7::Constant(joint_a_max_rad_s2), xrobot_cpp::Vec7::Constant(joint_j_max_rad_s3));
  trajectory.reset(feedback.joints);
  xrobot_cpp::JointTrajectoryState desired = trajectory.state();
  xrobot_cpp::Vec7 joint_goal = feedback.joints;
  auto last_ik_update = ready_now;
  const xrobot_cpp::Vec7 max_torque_delta_per_cycle = xrobot_cpp::Vec7::Constant(0.03);
  xrobot_cpp::Vec7 applied_torque = xrobot_cpp::Vec7::Zero();
  xrobot_cpp::Vec7 impedance_scale = xrobot_cpp::Vec7::Zero();
  xrobot_cpp::Vec7 gravity_trim_nm = xrobot_cpp::Vec7::Zero();
  std::optional<xrobot_cpp::Clock::time_point> j2_trim_saturation_started;
  // K=12/D=1 is the verified real-arm hold setting with nominal G(q) and the
  // bounded gravity trim. Higher K=16 destabilized real static hold.
  constexpr double kKp = 12.0, kKd = 1.0, kGainRise = 0.025;
  xrobot_cpp::RelativeMapper mapper(translation_scale, 1.0, 0.012, 0.05, base_from_xr);
  xrobot_cpp::SafetyGate xr_gate(xr_timeout_s);
  constexpr double kGripperCloseM = 0.0, kGripperOpenM = 0.100, kGripperStepM = 0.001;
  constexpr double kGripperDeadbandM = 0.0003, kGripperInputDeadband = 0.02, kGripperForceN = 1.0;
  double gripper_target_m = 0.060, gripper_last_command_m = gripper_target_m;
  xrobot_cpp::XrClient client;
  client.start();
  std::optional<xrobot_cpp::XrFrame> initial_frame;
  while (!initial_frame) {
    initial_frame = client.read_fresh();
    if (!initial_frame) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  // Keep SocketCAN receive latency out of the fixed-rate MIT sender.  The
  // socket owns a mutex-protected latest-feedback snapshot, so the control
  // loop below only reads that snapshot and never waits for a CAN frame.
  std::atomic_bool feedback_receiver_failed{false};
  std::jthread feedback_receiver([&can, &feedback_receiver_failed](std::stop_token stop) {
    while (!stop.stop_requested()) {
      try {
        (void)can.read(std::chrono::milliseconds(10));
        for (int drained = 0; drained < 64; ++drained) {
          if (!can.read(std::chrono::milliseconds(0))) break;
        }
      } catch (...) {
        feedback_receiver_failed.store(true, std::memory_order_release);
        return;
      }
    }
  });

  bool mit_mode_entered = false;
  auto restore_joint_mode = [&] {
    if (!mit_mode_entered) return;
    try {
      can.send_mode_joint(10, true);
      const xrobot_cpp::Vec7 hold_target = feedback.joints;
      for (int repeat = 0; repeat < 3; ++repeat) can.send_joint_target(hold_target);
      std::cout << "XR HOLD: restored MOVE J and sent three measured-position hold targets; no zero-torque MIT frame was sent.\n";
    } catch (const std::exception& error) { std::cerr << "XR HOLD WARNING: " << error.what() << "\n"; }
    mit_mode_entered = false;
  };
  auto enter_mit = [&] {
    trajectory.reset(feedback.joints);
    desired = trajectory.state();
    joint_goal = feedback.joints;
    applied_torque = model.dynamics(feedback.joints, xrobot_cpp::Vec7::Zero()).gravity.cwiseMax(-limits.max_torque_nm).cwiseMin(limits.max_torque_nm);
    impedance_scale.setZero();
    gravity_trim_nm.setZero();
    j2_trim_saturation_started.reset();
    can.send_mode_mit(10, true);
    mit_mode_entered = true;
    for (std::uint8_t joint = 1; joint <= 7; ++joint) {
      const Eigen::Index i = static_cast<Eigen::Index>(joint - 1);
      can.send_joint_mit(joint, {.position_rad = desired.position[i], .velocity_rad_s = 0.0, .kp_nm_rad = kKp, .kd_nm_s_rad = kKd, .feedforward_torque_nm = applied_torque[i]});
    }
  };

  std::cout << "Real XR teleoperation ready on " << interface_name << ". TCP orientation=" << orientation_mode << "; IK backend=" << kIkBackend << "; nominal URDF G(q) feedforward plus MIT Kp=" << kKp << ", Kd=" << kKd << "; gripper=" << (xr_gripper_enabled ? "enabled (Trigger=close, Grip=open)" : "disabled") << ".  CAN feedback runs in a dedicated receiver thread; MIT transmit is scheduled at 100Hz. Hold Pico A to enter MIT tracking. Release A, XR timeout, safety rejection, or Ctrl+C restores MOVE J measured-position hold.\n";
  g_mit_stop_requested = 0;
  std::signal(SIGINT, request_mit_stop);
  const auto start = xrobot_cpp::Clock::now();
  auto next_cycle = start;
  auto last_ik_warning = start - std::chrono::seconds(1);
  auto last_ik_telemetry = start - std::chrono::milliseconds(100);
  auto last_mit_telemetry = start - std::chrono::milliseconds(100);
  xrobot_cpp::Pose session_tcp_reference = model.forward_kinematics(feedback.joints);
  std::size_t sent_cycles = 0;
  try {
    while (!g_mit_stop_requested &&
           (seconds == 0.0 || std::chrono::duration<double>(xrobot_cpp::Clock::now() - start).count() < seconds)) {
      if (feedback_receiver_failed.load(std::memory_order_acquire)) {
        throw std::runtime_error("real XR safety stop: SocketCAN feedback receiver failed");
      }
      const auto now = xrobot_cpp::Clock::now();
      const auto now_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
      feedback = can.feedback();
      // The receiver can publish a frame between the earlier loop timestamp
      // and this snapshot.  Compute feedback age after taking the snapshot.
      const auto safety_now = xrobot_cpp::Clock::now();
      const auto safety_now_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(safety_now.time_since_epoch()).count());
      if (feedback.joint_state_sequence != last_joint_state_sequence) {
        const auto stamp = feedback.joint_state_monotonic_ns ? xrobot_cpp::Clock::time_point(std::chrono::nanoseconds(feedback.joint_state_monotonic_ns)) : now;
        filtered_velocity = velocity_estimator.update(feedback.joints, stamp);
        last_joint_state_sequence = feedback.joint_state_sequence;
      }
      std::optional<xrobot_cpp::XrFrame> frame = initial_frame ? initial_frame : client.read_fresh();
      initial_frame.reset();
      if (frame) {
        const xrobot_cpp::GateState state = xr_gate.observe(*frame, now);
        if (state == xrobot_cpp::GateState::kActivate) {
          session_tcp_reference = model.forward_kinematics(feedback.joints);
          mapper.rebase(*frame, session_tcp_reference, now);
          // The loop may have spent seconds waiting for a new A press.  Start
          // the 100 Hz deadline at this activation, rather than repeatedly
          // sleeping until an already-expired deadline and flooding SocketCAN.
          next_cycle = now;
          last_ik_update = now;
          if (xr_gripper_enabled) {
            const xrobot_cpp::GripperFeedback gripper = can.gripper_feedback();
            if (gripper.monotonic_ns != 0 && std::isfinite(gripper.width_m) &&
                gripper.width_m >= kGripperCloseM && gripper.width_m <= kGripperOpenM) {
              gripper_target_m = gripper.width_m;
            }
            gripper_last_command_m = gripper_target_m;
          }
          enter_mit();
          std::cout << "XR ACTIVE: rebased at measured TCP and entered MIT tracking.\n";
        } else if (state == xrobot_cpp::GateState::kTrack && mit_mode_entered) {
          if (!freeze_xr_target) {
            try {
            xrobot_cpp::Pose candidate = freeze_xr_target ? session_tcp_reference : mapper.target(*frame, now).pose;
            xrobot_cpp::Vec3 offset = candidate.position - session_tcp_reference.position;
            if (session_radius_m > 0.0 && offset.norm() > session_radius_m) {
              offset *= session_radius_m / offset.norm();
            }
            const xrobot_cpp::Pose reference = model.forward_kinematics(joint_goal);
            // Let the QP see the full hand displacement.  Its hard joint-speed
            // constraints produce the continuous path; an XR-frame waypoint cap
            // would incorrectly limit TCP motion by the callback rate.
            candidate.position = session_tcp_reference.position + offset;
            if (orientation_mode == "absolute") {
              const Eigen::Quaterniond requested = (Eigen::Quaterniond(base_from_xr) *
                  frame->controller.orientation.normalized() *
                  Eigen::Quaterniond(controller_to_tcp_rotation)).normalized();
              // The QP already constrains joint velocity.  Give it the full absolute
              // attitude target so a large hand orientation change is not rate-limited
              // a second time by an artificial TCP waypoint cap.
              candidate.orientation = requested;
            } else {
              candidate.orientation = reference.orientation;
            }
            xrobot_cpp::ContinuousIkOptions ik_options;
            ik_options.position_weight = 1.0;
            ik_options.orientation_weight = orientation_mode == "absolute" ? 1.2 : 0.02;
            ik_options.position_tolerance_m = 0.005;
            ik_options.orientation_tolerance_rad = orientation_mode == "absolute" ? 0.030 : 0.15;
            ik_options.continuity_weight = 0.005;
            ik_options.joint_limit_weight = 0.000005;
            ik_options.max_iterations = 160;
            // Match the Python Placo control law: position and absolute TCP
            // orientation are one 6-D task, rather than making wrist motion
            // depend on translation null-space freedom.
            #ifdef XROBOT_CPP_WITH_PLACO
            // The XR callback rate is lower than the 100 Hz MIT transmit loop.
            // Use its actual interval so the QP velocity bound remains in rad/s.
            const double ik_dt_s = std::clamp(
                std::chrono::duration<double>(now - last_ik_update).count(), 0.010, 0.100);
            last_ik_update = now;
            const auto ik_solve_started = xrobot_cpp::Clock::now();
            const xrobot_cpp::ContinuousIkResult ik_result = placo_ik.solve(
                candidate, joint_goal, ik_options, ik_dt_s, joint_v_max_rad_s);
            const double ik_solve_ms = std::chrono::duration<double, std::milli>(xrobot_cpp::Clock::now() - ik_solve_started).count();
#else
            const xrobot_cpp::ContinuousIkResult ik_result = model.solve_redundant_continuous_ik(
                candidate, joint_goal, ik_options, 0.02);
#endif
            const xrobot_cpp::Vec7 proposed = ik_result.joints;
            if (now - last_ik_telemetry >= std::chrono::milliseconds(100)) {
              write_xr_ik_target_event(&telemetry, now_ns, *frame, candidate, ik_result, ik_solve_ms);
              last_ik_telemetry = now;
            }
            const xrobot_cpp::Vec7 lower_distance = proposed - model.lower_limits();
            const xrobot_cpp::Vec7 upper_distance = model.upper_limits() - proposed;
            if (lower_distance.minCoeff() < -1e-8 || upper_distance.minCoeff() < -1e-8) {
              throw std::runtime_error("unified IK returned a target outside URDF joint limits");
            }
            joint_goal = proposed;
            trajectory.set_goal(joint_goal);
            if (xr_gripper_enabled) {
              const double trigger = std::clamp(frame->trigger, 0.0, 1.0);
              const double grip = std::clamp(frame->grip, 0.0, 1.0);
              if (grip >= kGripperInputDeadband && trigger < kGripperInputDeadband) {
                gripper_target_m = std::min(kGripperOpenM, gripper_target_m + grip * kGripperStepM);
              } else if (trigger >= kGripperInputDeadband && grip < kGripperInputDeadband) {
                gripper_target_m = std::max(kGripperCloseM, gripper_target_m - trigger * kGripperStepM);
              }
              if (std::abs(gripper_target_m - gripper_last_command_m) >= kGripperDeadbandM) {
                can.send_gripper_width(gripper_target_m, kGripperForceN, true);
                write_xr_gripper_command_event(&telemetry, now_ns, *frame, gripper_target_m, kGripperForceN);
                gripper_last_command_m = gripper_target_m;
              }
            }
          } catch (const std::exception& error) {
            if (now - last_ik_warning >= std::chrono::seconds(1)) { std::cerr << "XR HOLD: IK target rejected; retaining last reachable target: " << error.what() << "\n"; last_ik_warning = now; }
          }
          }
        } else if (state == xrobot_cpp::GateState::kRelease) {
          std::cout << "XR HOLD: Pico A released.\n";
          restore_joint_mode();
        }
      }
      if (xr_gate.poll(now) == xrobot_cpp::GateState::kTimeout) { std::cout << "XR HOLD: input timed out.\n"; restore_joint_mode(); }
      if (!mit_mode_entered) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
      desired = trajectory.step(0.010);
      const xrobot_cpp::DynamicsTerms dynamics = model.dynamics(feedback.joints, filtered_velocity);
      const xrobot_cpp::Vec7 position_error = desired.position - feedback.joints;
      const xrobot_cpp::Vec7 velocity_error = desired.velocity - filtered_velocity;
      update_gravity_trim(&gravity_trim_nm, position_error, filtered_velocity, 0.010);
      // J2 is the only axis that reached the trim bound during the bounded
      // 30-second XR validation. A sustained bound means nominal G(q) is no
      // longer sufficient at this pose, so hand control back to MOVE J.
      constexpr Eigen::Index kJ2 = 1;
      constexpr double kJ2TrimWatchdogThresholdNm = 0.249;
      constexpr double kJ2TrimWatchdogDurationS = 0.50;
      if (std::abs(gravity_trim_nm[kJ2]) >= kJ2TrimWatchdogThresholdNm) {
        if (!j2_trim_saturation_started) j2_trim_saturation_started = now;
        const double duration_s = std::chrono::duration<double>(now - *j2_trim_saturation_started).count();
        if (duration_s >= kJ2TrimWatchdogDurationS) {
          write_gravity_trim_watchdog_event(&telemetry, now_ns, feedback, dynamics, gravity_trim_nm, duration_s);
          std::cerr << "XR HOLD: J2 gravity-trim watchdog saturated for " << duration_s
                    << "s; restoring MOVE J at q=[" << feedback.joints.transpose() << "]\n";
          restore_joint_mode();
          j2_trim_saturation_started.reset();
          continue;
        }
      } else {
        j2_trim_saturation_started.reset();
      }
      const xrobot_cpp::Vec7 gravity = (nominal_gravity_feedforward(dynamics, limits.max_torque_nm) + gravity_trim_nm)
          .cwiseMax(-limits.max_torque_nm).cwiseMin(limits.max_torque_nm);
      applied_torque += (gravity - applied_torque).cwiseMax(-max_torque_delta_per_cycle).cwiseMin(max_torque_delta_per_cycle);
      const xrobot_cpp::Vec7 impedance_torque = xrobot_cpp::Vec7::Constant(kKp).cwiseProduct(position_error) + xrobot_cpp::Vec7::Constant(kKd).cwiseProduct(velocity_error);
      xrobot_cpp::Vec7 headroom = xrobot_cpp::Vec7::Ones();
      for (Eigen::Index i = 0; i < 7; ++i) {
        const double available = std::max(0.0, limits.max_torque_nm[i] - std::abs(applied_torque[i]));
        if (std::abs(impedance_torque[i]) > available && std::abs(impedance_torque[i]) > 1e-9) headroom[i] = available / std::abs(impedance_torque[i]);
        impedance_scale[i] = headroom[i] < impedance_scale[i] ? headroom[i] : std::min(headroom[i], impedance_scale[i] + kGainRise);
      }
      const xrobot_cpp::Vec7 kp = xrobot_cpp::Vec7::Constant(kKp).cwiseProduct(impedance_scale);
      const xrobot_cpp::Vec7 kd = xrobot_cpp::Vec7::Constant(kKd).cwiseProduct(impedance_scale);
      const xrobot_cpp::Vec7 total_torque = applied_torque + impedance_scale.cwiseProduct(impedance_torque);
      safety.notify_keepalive(safety_now_ns);
      std::string reason;
      if (!safety.permits(desired, filtered_velocity, total_torque, feedback, safety_now_ns, &reason)) {
        // A transient CAN feedback gap must not terminate the teleop process.
        // Stop MIT output, hand back to MOVE J hold, and wait for fresh feedback
        // plus the next A press before another MIT entry.
        std::cerr << "XR HOLD: safety gate paused MIT: " << reason;
        if (reason == "feedback is stale or incomplete") {
          const xrobot_cpp::CanReceiveDiagnostics rx = can.receive_diagnostics();
          const xrobot_cpp::NeroFeedback current_feedback = can.feedback();
          const auto diagnostic_now = xrobot_cpp::Clock::now();
          const auto diagnostic_now_ns = static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(diagnostic_now.time_since_epoch()).count());
          const auto age_ms = [diagnostic_now_ns](std::uint64_t stamp) {
            return stamp == 0 || diagnostic_now_ns < stamp ? -1.0
                : static_cast<double>(diagnostic_now_ns - stamp) / 1e6;
          };
          const double safety_snapshot_age_ms = feedback.monotonic_ns == 0 || safety_now_ns < feedback.monotonic_ns ? -1.0
              : static_cast<double>(safety_now_ns - feedback.monotonic_ns) / 1e6;
          std::cerr << " safety_snapshot_age_ms=" << safety_snapshot_age_ms
                    << " rx_frames=" << rx.received_frames
                    << " joint_rx_frames=" << rx.joint_feedback_frames
                    << " last_rx_id=0x" << std::hex << rx.last_received_id << std::dec
                    << " last_rx_age_ms=" << age_ms(rx.last_received_monotonic_ns)
                    << " last_joint_id=0x" << std::hex << rx.last_joint_feedback_id << std::dec
                    << " last_joint_age_ms=" << age_ms(rx.last_joint_feedback_monotonic_ns)
                    << " safety_cycle_sequence=" << feedback.joint_state_sequence
                    << " current_cycle_sequence=" << current_feedback.joint_state_sequence;
        }
        std::cerr << "\n";
        restore_joint_mode();
        continue;
      }
      if (now - last_mit_telemetry >= std::chrono::milliseconds(100)) {
        const xrobot_cpp::GripperFeedback gripper_feedback = can.gripper_feedback();
        const double gripper_width_m = gripper_feedback.monotonic_ns != 0 && std::isfinite(gripper_feedback.width_m)
            ? gripper_feedback.width_m : std::numeric_limits<double>::quiet_NaN();
        write_mit_telemetry(&telemetry, now_ns, feedback, filtered_velocity, desired, dynamics, applied_torque, kp, kd,
                            "real_xr_teleop", gripper_width_m);
        telemetry.flush();
        last_mit_telemetry = now;
      }
      for (std::uint8_t joint = 1; joint <= 7; ++joint) {
        const Eigen::Index i = static_cast<Eigen::Index>(joint - 1);
        can.send_joint_mit(joint, {.position_rad = desired.position[i], .velocity_rad_s = desired.velocity[i], .kp_nm_rad = kp[i], .kd_nm_s_rad = kd[i], .feedforward_torque_nm = applied_torque[i]});
      }
      ++sent_cycles;
      next_cycle += std::chrono::milliseconds(10);
      std::this_thread::sleep_until(next_cycle);
    }
    restore_joint_mode();
    std::cout << "Real XR teleoperation ended: mit_cycles=" << sent_cycles << ".\n";
  } catch (...) { restore_joint_mode(); throw; }
  return EXIT_SUCCESS;
}
#if defined(XROBOT_CPP_WITH_MUJOCO)
int run_mujoco_xr_teleop(int argc, char** argv) {
  double translation_scale = 0.70, rotation_scale = 1.00, gripper_width_m = 0.060, payload_mass_kg = 0.0, seconds = 0.0, xr_connect_timeout_s = 15.0;
  std::string orientation_mode{"locked"};
  std::filesystem::path absolute_orientation_calibration{"configs/nero_xr_calibration.json"};
  double orientation_step_rad = 0.03;
  bool viewer = false;
  std::filesystem::path telemetry_path{"results/mujoco_xr_teleop.csv"};
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--translation-scale" && index + 1 < argc) translation_scale = std::stod(argv[++index]);
    else if (argument == "--rotation-scale" && index + 1 < argc) rotation_scale = std::stod(argv[++index]);
    else if (argument == "--orientation-mode" && index + 1 < argc) orientation_mode = argv[++index];
    else if (argument == "--absolute-orientation-calibration" && index + 1 < argc) absolute_orientation_calibration = argv[++index];
    else if (argument == "--orientation-step-rad" && index + 1 < argc) orientation_step_rad = std::stod(argv[++index]);
    else if (argument == "--viewer") viewer = true;
    else if (argument == "--gripper-width-m" && index + 1 < argc) gripper_width_m = std::stod(argv[++index]);
    else if (argument == "--payload-mass-kg" && index + 1 < argc) payload_mass_kg = std::stod(argv[++index]);
    else if (argument == "--seconds" && index + 1 < argc) seconds = std::stod(argv[++index]);
    else if (argument == "--xr-connect-timeout-s" && index + 1 < argc) xr_connect_timeout_s = std::stod(argv[++index]);
    else if (argument == "--telemetry-log" && index + 1 < argc) telemetry_path = argv[++index];
    else throw std::invalid_argument("usage: --mujoco-xr-teleop [--translation-scale S] [--rotation-scale S] [--orientation-mode locked|full|absolute] [--absolute-orientation-calibration PATH] [--orientation-step-rad R<=0.08] [--viewer] [--gripper-width-m M] [--payload-mass-kg KG] [--seconds S] [--xr-connect-timeout-s 0=unlimited|S<=60] [--telemetry-log PATH]");
  }
  if (!std::isfinite(translation_scale) || translation_scale <= 0.0 || translation_scale > 2.0) throw std::invalid_argument("--translation-scale must be in (0, 2]");
  if (!std::isfinite(rotation_scale) || rotation_scale <= 0.0 || rotation_scale > 2.0) throw std::invalid_argument("--rotation-scale must be in (0, 2]");
  if (orientation_mode != "locked" && orientation_mode != "full" && orientation_mode != "absolute") throw std::invalid_argument("--orientation-mode must be locked, full, or absolute");
  const Eigen::Matrix3d controller_to_tcp_rotation = orientation_mode == "absolute"
      ? load_absolute_controller_to_tcp_rotation(absolute_orientation_calibration) : Eigen::Matrix3d::Identity();
  if (!std::isfinite(orientation_step_rad) || orientation_step_rad <= 0.0 || orientation_step_rad > 0.08) throw std::invalid_argument("--orientation-step-rad must be in (0, 0.08]");
  if (!std::isfinite(gripper_width_m) || gripper_width_m < 0.0 || gripper_width_m > 0.100) throw std::invalid_argument("--gripper-width-m must be in [0, 0.100]");
  if (!std::isfinite(payload_mass_kg) || payload_mass_kg < 0.0 || payload_mass_kg > 3.0) throw std::invalid_argument("--payload-mass-kg must be in [0, 3]");
  if (!std::isfinite(seconds) || seconds < 0.0 || seconds > 3600.0) throw std::invalid_argument("--seconds must be in [0, 3600]");
  if (!std::isfinite(xr_connect_timeout_s) || xr_connect_timeout_s < 0.0 || xr_connect_timeout_s > 60.0) throw std::invalid_argument("--xr-connect-timeout-s must be 0 (unlimited) or in (0, 60]");
#ifndef XROBOT_CPP_WITH_MUJOCO_VIEWER
  if (viewer) throw std::runtime_error("--viewer requires a build configured with -DXROBOT_CPP_WITH_MUJOCO_VIEWER=ON");
#endif

  if (!telemetry_path.parent_path().empty()) std::filesystem::create_directories(telemetry_path.parent_path());
  std::ofstream telemetry(telemetry_path);
  if (!telemetry) throw std::runtime_error("cannot open teleoperation telemetry: " + telemetry_path.string());
  telemetry << "sim_time_s,active,gripper_width_m,q0,q1,q2,q3,q4,q5,q6,qd0,qd1,qd2,qd3,qd4,qd5,qd6,q_des0,q_des1,q_des2,q_des3,q_des4,q_des5,q_des6,tau0,tau1,tau2,tau3,tau4,tau5,tau6\n";
  const std::string root(XROBOT_CPP_SOURCE_DIR);
  xrobot_cpp::NeroDynamicsModel dynamics(root + "/assets/urdf/nero_control_tcp.urdf", "base_link", "gripper_tcp");
#ifdef XROBOT_CPP_WITH_PLACO
  xrobot_cpp::PlacoNeroIkSolver placo_ik(root + "/assets/urdf/nero_control_tcp.urdf", "gripper_tcp");
#endif
  xrobot_cpp::NeroMujocoSimulation simulation(root + "/assets/mujoco/nero_torque.xml");
  const xrobot_cpp::Vec7 initial = (xrobot_cpp::Vec7() << 0.0, -0.20, 0.30, 0.20, -0.10, 0.10, 0.0).finished();
  const xrobot_cpp::Vec7 torque_limit = mit_support_torque_limits();
  const xrobot_cpp::ComputedTorqueController controller(xrobot_cpp::Vec7::Constant(kNeroLowSpeedSpringStiffnessNmRad), xrobot_cpp::Vec7::Constant(kNeroLowSpeedDampingNmSRad), torque_limit, xrobot_cpp::JointDriveModel::mujoco_default());
#ifndef XROBOT_CPP_WITH_PLACO
  const xrobot_cpp::Vec7 max_joint_target_step = xrobot_cpp::Vec7::Constant(0.006);
#endif
  xrobot_cpp::JerkLimitedJointPlanner trajectory(
      xrobot_cpp::Vec7::Constant(0.20), xrobot_cpp::Vec7::Constant(0.50), xrobot_cpp::Vec7::Constant(5.0));
  Eigen::Matrix3d base_from_xr;
  base_from_xr << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  xrobot_cpp::RelativeMapper mapper(translation_scale, rotation_scale, 0.012, 0.05, base_from_xr);
  xrobot_cpp::SafetyGate gate(0.50);
  xrobot_cpp::JointTrajectoryState desired{.position = initial, .velocity = xrobot_cpp::Vec7::Zero(), .acceleration = xrobot_cpp::Vec7::Zero()};
  xrobot_cpp::Vec7 joint_goal = initial;
  trajectory.reset(initial);
  xrobot_cpp::Pose session_tcp_reference = dynamics.forward_kinematics(initial);
  simulation.reset(initial);
  simulation.set_payload_mass_kg(payload_mass_kg);
  simulation.set_gripper_width(gripper_width_m);
#ifdef XROBOT_CPP_WITH_MUJOCO_VIEWER
  GLFWwindow* viewer_window = nullptr;
  mjvCamera viewer_camera; mjv_defaultCamera(&viewer_camera);
  mjvOption viewer_option; mjv_defaultOption(&viewer_option);
  mjvScene viewer_scene; mjv_defaultScene(&viewer_scene);
  mjrContext viewer_context; mjr_defaultContext(&viewer_context);
  if (viewer) {
    glfwSetErrorCallback([](int, const char* message) { std::cerr << "GLFW: " << message << "\n"; });
    if (!glfwInit()) throw std::runtime_error("GLFW initialization failed; run this command in a desktop session");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    viewer_window = glfwCreateWindow(1280, 900, "NERO MuJoCo XR - absolute TCP orientation", nullptr, nullptr);
    if (viewer_window == nullptr) { glfwTerminate(); throw std::runtime_error("cannot create MuJoCo XR viewer window; see GLFW error above"); }
    glfwMakeContextCurrent(viewer_window); glfwSwapInterval(1);
    glfwSetKeyCallback(viewer_window, [](GLFWwindow* target, int key, int, int action, int) { if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(target, GLFW_TRUE); });
    viewer_camera.type = mjCAMERA_FREE; viewer_camera.azimuth = 135.0; viewer_camera.elevation = -25.0; viewer_camera.distance = 1.5;
    viewer_camera.lookat[0] = 0.0; viewer_camera.lookat[1] = 0.0; viewer_camera.lookat[2] = 0.35;
    mjv_makeScene(simulation.model(), &viewer_scene, 2000);
    mjr_makeContext(simulation.model(), &viewer_context, mjFONTSCALE_150);
  }
#endif
  xrobot_cpp::XrClient client;
  client.start();
  std::optional<xrobot_cpp::XrFrame> initial_frame;
  const auto connect_started = xrobot_cpp::Clock::now();
  std::cout << "Waiting for a fresh Pico XR frame (timeout=" << xr_connect_timeout_s << "s)...\n";
  while (!initial_frame && (xr_connect_timeout_s == 0.0 ||
         std::chrono::duration<double>(xrobot_cpp::Clock::now() - connect_started).count() < xr_connect_timeout_s)) {
    initial_frame = client.read_fresh();
    if (!initial_frame) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (!initial_frame) {
    throw std::runtime_error("no fresh Pico XR frame received before the configured timeout; on Pico, open XRoboToolkit, set the PC address, and enable Tracking/Controller plus Data & Control/Send");
  }
  const auto start = xrobot_cpp::Clock::now();
  auto next_step = start;
  auto last_report = start;
  double gripper_target_m = gripper_width_m;
  std::size_t physics_steps = 0;
  auto last_ik_warning = start - std::chrono::seconds(1);
  auto last_tracking_adapt = start - std::chrono::seconds(1);
  std::cout << "MuJoCo XR teleoperation started. TCP orientation=" << orientation_mode
#ifdef XROBOT_CPP_WITH_PLACO
            << "; IK backend=Placo-QP"
#else
            << "; IK backend=KDL-DLS"
#endif
            << "; trajectory limits=[0.20 rad/s, 0.50 rad/s^2, 5.0 rad/s^3]"
            << "; controller torque limits=[" << torque_limit.transpose() << "]Nm. Simulation only: no SocketCAN interface is opened.\n";
  std::cout << "Hold Pico A to rebase and track. Release A to hold. Trigger closes; Grip opens the simulated gripper. Ctrl+C stops.\n";
  while (true) {
#ifdef XROBOT_CPP_WITH_MUJOCO_VIEWER
    if (viewer && glfwWindowShouldClose(viewer_window)) break;
#endif
    const auto now = xrobot_cpp::Clock::now();
    if (seconds > 0.0 && std::chrono::duration<double>(now - start).count() >= seconds) break;
    std::optional<xrobot_cpp::XrFrame> frame;
    if (initial_frame) {
      frame = initial_frame;
      initial_frame.reset();
    } else {
      frame = client.read_fresh();
    }
    if (frame) {
      const xrobot_cpp::GateState state = gate.observe(*frame, now);
      if (state == xrobot_cpp::GateState::kActivate) {
        joint_goal = simulation.joint_position();
        trajectory.reset(joint_goal);
        desired = trajectory.state();
        session_tcp_reference = dynamics.forward_kinematics(joint_goal);
        mapper.rebase(*frame, session_tcp_reference, now);
        std::cout << "SIM ACTIVE: target rebased at current simulation state.\n";
      } else if (state == xrobot_cpp::GateState::kTrack) {
        try {
          xrobot_cpp::Pose candidate = mapper.target(*frame, now).pose;
          const xrobot_cpp::Pose reference = dynamics.forward_kinematics(joint_goal);
          constexpr double kSessionWorkspaceRadiusM = 0.08;
          xrobot_cpp::Vec3 session_offset = candidate.position - session_tcp_reference.position;
          if (session_offset.norm() > kSessionWorkspaceRadiusM) {
            session_offset *= kSessionWorkspaceRadiusM / session_offset.norm();
          }
          candidate.position = session_tcp_reference.position + session_offset;
          // The Placo joint-velocity constraints form the continuous path.
          // Do not turn an XR callback-rate-dependent 1 mm cap into a hidden
          // Cartesian speed limiter.
          if (orientation_mode == "locked") {
            candidate.orientation = session_tcp_reference.orientation;
          } else {
            const Eigen::Quaterniond requested = orientation_mode == "absolute"
                ? (Eigen::Quaterniond(base_from_xr) * frame->controller.orientation.normalized() *
                   Eigen::Quaterniond(controller_to_tcp_rotation)).normalized()
                : candidate.orientation.normalized();
            Eigen::Quaterniond rotation_delta = (requested * reference.orientation.conjugate()).normalized();
            if (rotation_delta.w() < 0.0) rotation_delta.coeffs() *= -1.0;
            const Eigen::AngleAxisd angle_axis(rotation_delta);
            candidate.orientation = angle_axis.angle() > orientation_step_rad
                ? (Eigen::Quaterniond(Eigen::AngleAxisd(orientation_step_rad, angle_axis.axis())) * reference.orientation).normalized()
                : requested;
          }
          candidate.orientation.normalize();
          xrobot_cpp::Vec7 proposed_goal;
#ifdef XROBOT_CPP_WITH_PLACO
          xrobot_cpp::ContinuousIkOptions ik_options;
          ik_options.position_weight = 1.0;
          ik_options.orientation_weight = orientation_mode == "locked" ? 0.02 : 1.2;
          const xrobot_cpp::ContinuousIkResult ik_result = placo_ik.solve(candidate, joint_goal, ik_options, 0.010, 0.60);
          proposed_goal = ik_result.joints;
#else
          proposed_goal = dynamics.solve_ik_step(candidate, joint_goal, max_joint_target_step[0]);
#endif
#ifndef XROBOT_CPP_WITH_PLACO
          constexpr double kJointLimitMarginRad = 0.10;
          if ((proposed_goal - dynamics.lower_limits()).minCoeff() < kJointLimitMarginRad ||
              (dynamics.upper_limits() - proposed_goal).minCoeff() < kJointLimitMarginRad) {
            throw std::runtime_error("IK target enters the joint-limit margin");
          }
#endif
          joint_goal = proposed_goal;
          trajectory.set_goal(joint_goal);
          gripper_target_m = std::clamp(gripper_target_m + (frame->grip - frame->trigger) * 0.00035, 0.0, 0.100);
        } catch (const std::exception& error) {
          if (now - last_ik_warning >= std::chrono::seconds(1)) {
            std::cerr << "SIM HOLD: IK target rejected; retaining the last reachable target: " << error.what() << "\n";
            last_ik_warning = now;
          }
        }
      } else if (state == xrobot_cpp::GateState::kRelease) {
        std::cout << "SIM HOLD: Pico A released; retaining last target.\n";
      }
    }
    if (gate.poll(now) == xrobot_cpp::GateState::kTimeout) std::cout << "SIM HOLD: XR input timed out; retaining last target.\n";
    int catchup_steps = 0;
    while (now >= next_step && catchup_steps < 4) {
      const xrobot_cpp::Vec7 measured_position = simulation.joint_position();
      if ((desired.position - measured_position).cwiseAbs().maxCoeff() > 0.20) {
        joint_goal = measured_position;
        trajectory.reset(joint_goal);
        desired = trajectory.state();
        session_tcp_reference = dynamics.forward_kinematics(joint_goal);
        if (frame && gate.active()) mapper.rebase(*frame, session_tcp_reference, now);
        if (now - last_tracking_adapt >= std::chrono::seconds(1)) {
          std::cerr << "SIM TRACKING ADAPT: re-anchored after joint tracking lag.\n";
          last_tracking_adapt = now;
        }
      }
      desired = trajectory.step(0.001);
      const xrobot_cpp::ComputedTorqueCommand command = controller.compute(simulation.dynamics_terms(), simulation.joint_position(), simulation.joint_velocity(), desired);
      simulation.set_joint_torque(command.feedforward_torque_nm);
      simulation.set_gripper_width(gripper_target_m);
      simulation.step();
      ++physics_steps;
      if (physics_steps % 10 == 0) {
        const xrobot_cpp::Vec7 q = simulation.joint_position();
        const xrobot_cpp::Vec7 qd = simulation.joint_velocity();
        telemetry << simulation.time_s() << "," << gate.active() << "," << simulation.gripper_width_m();
        for (int joint = 0; joint < 7; ++joint) telemetry << "," << q[joint];
        for (int joint = 0; joint < 7; ++joint) telemetry << "," << qd[joint];
        for (int joint = 0; joint < 7; ++joint) telemetry << "," << desired.position[joint];
        for (int joint = 0; joint < 7; ++joint) telemetry << "," << command.feedforward_torque_nm[joint];
        telemetry << "\n";
      }
      next_step += std::chrono::milliseconds(1);
      ++catchup_steps;
    }
#ifdef XROBOT_CPP_WITH_MUJOCO_VIEWER
    if (viewer) {
      mjrRect viewport{0, 0, 0, 0}; glfwGetFramebufferSize(viewer_window, &viewport.width, &viewport.height);
      mjv_updateScene(simulation.model(), simulation.data(), &viewer_option, nullptr, &viewer_camera, mjCAT_ALL, &viewer_scene);
      mjr_render(viewport, &viewer_scene, &viewer_context);
      glfwSwapBuffers(viewer_window); glfwPollEvents();
    }
#endif
    if (now - last_report >= std::chrono::milliseconds(500)) {
      std::cout << "SIM t=" << std::fixed << std::setprecision(2) << simulation.time_s() << " q_error=" << (desired.position - simulation.joint_position()).norm() << " gripper=" << simulation.gripper_width_m() << "\n";
      last_report = now;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
#ifdef XROBOT_CPP_WITH_MUJOCO_VIEWER
  if (viewer) { mjr_freeContext(&viewer_context); mjv_freeScene(&viewer_scene); glfwDestroyWindow(viewer_window); glfwTerminate(); }
#endif
  std::cout << "MuJoCo XR teleoperation stopped at sim_time_s=" << simulation.time_s() << ". No SocketCAN interface was opened.\n";
  return EXIT_SUCCESS;
}
#endif
#endif
}  // namespace

int main(int argc, char** argv) {
  try {
  if (argc > 1 && (std::string_view(argv[1]) == "--gripper-status" ||
                   std::string_view(argv[1]) == "--gripper-enable" ||
                   std::string_view(argv[1]) == "--gripper-reset" ||
                   std::string_view(argv[1]) == "--gripper-release" ||
                   std::string_view(argv[1]) == "--gripper-configure-stroke" ||
                   std::string_view(argv[1]) == "--gripper-set-zero")) return run_gripper_maintenance(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--mit-telemetry-summary") return run_mit_telemetry_summary(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--effective-torque-identify") return run_effective_torque_identify(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--mit-effective-torque-collect") return run_mit_effective_torque_collect(argc, argv);
#ifdef XROBOT_CPP_WITH_MUJOCO
  if (argc > 1 && std::string_view(argv[1]) == "--mujoco-dry-run") return run_mujoco_dry_run(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--mujoco-benchmark") return run_mujoco_benchmark(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--mujoco-robustness") return run_mujoco_robustness(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--mujoco-full-chain-test") return run_mujoco_full_chain_test(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--mujoco-smooth-torque-test") return run_mujoco_smooth_torque_test(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--mujoco-gravity-impedance-hold") return run_mujoco_gravity_impedance_hold(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--mujoco-model-parity") return run_mujoco_model_parity();
  if (argc > 1 && std::string_view(argv[1]) == "--mujoco-impedance-convergence") return run_mujoco_impedance_convergence_benchmark(argc, argv);
#endif
#ifdef XROBOT_CPP_WITH_MUJOCO_VIEWER
  if (argc > 1 && std::string_view(argv[1]) == "--mujoco-viewer") return run_mujoco_viewer(argc, argv);
#endif
  if (argc > 1 && std::string_view(argv[1]) == "--dynamics-dry-run") return run_dynamics_dry_run(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--request-feedback") return run_feedback_request(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--enable-arm") return run_enable_arm(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--real-control-preflight") return run_real_control_preflight(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--mit-single-joint-hold") return run_mit_single_joint_hold(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--gravity-envelope-scan") return run_gravity_envelope_scan(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--real-gravity-preview") return run_real_gravity_preview(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--cartesian-ik-preview") return run_cartesian_ik_preview(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--mit-seven-axis-gravity-hold") return run_mit_seven_axis_gravity_hold(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--mit-j7-scurve-step") return run_mit_j7_scurve_step(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--mit-cartesian-scurve-step") return run_mit_cartesian_scurve_step(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--single-joint-test") return run_single_joint_test(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--real") {
    std::cerr << "ERROR: no AgileX NERO C++ driver is available in this environment. "
              << "This C++ project intentionally cannot send real-arm commands.\n";
    return EXIT_FAILURE;
  }
  if (argc > 1 && std::string_view(argv[1]) == "--can-protocol-test") return run_can_protocol_test();
  if (argc > 1 && std::string_view(argv[1]) == "--can-monitor") return run_can_monitor(argc > 2 ? std::string_view(argv[2]) : "can0");
#if defined(XROBOT_CPP_WITH_XR) && defined(XROBOT_CPP_WITH_MUJOCO)
  if (argc > 1 && std::string_view(argv[1]) == "--mujoco-xr-teleop") return run_mujoco_xr_teleop(argc, argv);
#endif
#ifdef XROBOT_CPP_WITH_XR
  if (argc > 1 && std::string_view(argv[1]) == "--real-xr-teleop") return run_real_xr_teleop(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "--xr-monitor") {
    return run_xr_monitor();
  }
#else
  if (argc > 1 && std::string_view(argv[1]) == "--xr-monitor") {
    std::cerr << "ERROR: rebuild with -DXROBOT_CPP_WITH_XR=ON for the native XR monitor.\n";
    return EXIT_FAILURE;
  }
#endif
  if (argc > 1 && std::string_view(argv[1]) != "--demo") {
    std::cerr << "Usage: xrobot_cpp [--demo|--dynamics-dry-run [--urdf PATH]|--mujoco-dry-run [--seconds S] [--gripper-width-m M]|--xr-monitor|--can-protocol-test|--can-monitor [can0]|--real]\n";
    return EXIT_FAILURE;
  }
  return run_demo();
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
