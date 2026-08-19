#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

#include "xrobot_cpp/control_core.hpp"

namespace xrobot_cpp {

class XrClient {
 public:
  XrClient() = default;
  ~XrClient();
  XrClient(const XrClient&) = delete;
  XrClient& operator=(const XrClient&) = delete;

  void start();
  void stop();
  [[nodiscard]] std::optional<XrFrame> read_fresh();

 private:
  static void callback(void* context, int type, int status, void* user_data);
  void on_callback(int type, int status, void* user_data);

  std::mutex mutex_;
  std::optional<XrFrame> latest_;
  std::uint64_t consumed_sequence_{};
  bool started_{false};
};

}  // namespace xrobot_cpp
