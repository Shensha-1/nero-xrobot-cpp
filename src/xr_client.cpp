#include "xrobot_cpp/xr_client.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include "PXREARobotSDK.h"

namespace xrobot_cpp {
namespace {
constexpr int kDeviceStateJson = PXREADeviceStateJson;

std::array<double, 7> parse_pose(const nlohmann::json& value) {
  const std::string encoded = value.get<std::string>();
  std::array<double, 7> pose{};
  std::stringstream stream(encoded);
  std::string field;
  for (std::size_t index = 0; index < pose.size(); ++index) {
    if (!std::getline(stream, field, ',')) {
      throw std::runtime_error("XR pose does not contain seven values");
    }
    pose[index] = std::stod(field);
    if (!std::isfinite(pose[index])) {
      throw std::runtime_error("XR pose contains a non-finite value");
    }
  }
  if (std::getline(stream, field, ',')) {
    throw std::runtime_error("XR pose contains more than seven values");
  }
  return pose;
}

Pose to_pose(const nlohmann::json& value) {
  const auto raw = parse_pose(value);
  Pose pose;
  pose.position = Vec3(raw[0], raw[1], raw[2]);
  pose.orientation = Eigen::Quaterniond(raw[6], raw[3], raw[4], raw[5]);
  if (pose.orientation.norm() <= 1e-8) {
    throw std::runtime_error("XR quaternion is invalid");
  }
  pose.orientation.normalize();
  return pose;
}

std::string bounded(const char* value, std::size_t size) {
  return std::string(value, std::find(value, value + size, 0));
}
}  // namespace

XrClient::~XrClient() { stop(); }

void XrClient::start() {
  if (started_) {
    return;
  }
  if (PXREAInit(this, reinterpret_cast<pfPXREAClientCallback>(&XrClient::callback), kDeviceStateJson) != 0) {
    throw std::runtime_error("PXREAInit failed");
  }
  started_ = true;
}

void XrClient::stop() {
  if (started_) {
    PXREADeinit();
    started_ = false;
  }
}

void XrClient::callback(void* context, int type, int status, void* user_data) {
  static_cast<XrClient*>(context)->on_callback(type, status, user_data);
}

void XrClient::on_callback(int type, int status, void* user_data) {
  if (type != kDeviceStateJson || status != 0 || user_data == nullptr) {
    return;
  }
  try {
    const auto& data = *static_cast<PXREADevStateJson*>(user_data);
    const nlohmann::json outer = nlohmann::json::parse(bounded(data.stateJson, sizeof(data.stateJson)));
    const nlohmann::json value = nlohmann::json::parse(outer.at("value").get<std::string>());
    const auto& right = value.at("Controller").at("right");
    XrFrame frame;
    frame.timestamp_ns = value.at("timeStampNs").get<std::uint64_t>();
    frame.device_id = bounded(data.devID, sizeof(data.devID));
    frame.controller = to_pose(right.at("pose"));
    frame.headset = to_pose(value.at("Head").at("pose"));
    frame.trigger = right.at("trigger").get<double>();
    frame.grip = right.at("grip").get<double>();
    frame.deadman = right.at("primaryButton").get<bool>();
    if (frame.device_id.empty() || frame.timestamp_ns == 0 || !std::isfinite(frame.trigger) ||
        !std::isfinite(frame.grip) || frame.trigger < 0.0 || frame.trigger > 1.0 ||
        frame.grip < 0.0 || frame.grip > 1.0) {
      return;
    }
    std::lock_guard lock(mutex_);
    frame.sequence = latest_ ? latest_->sequence + 1 : 1;
    latest_ = std::move(frame);
  } catch (const std::exception&) {
    // A malformed callback is dropped atomically; no partial state is exposed.
  }
}

std::optional<XrFrame> XrClient::read_fresh() {
  std::lock_guard lock(mutex_);
  if (!latest_ || latest_->sequence <= consumed_sequence_) {
    return std::nullopt;
  }
  consumed_sequence_ = latest_->sequence;
  return latest_;
}

}  // namespace xrobot_cpp
