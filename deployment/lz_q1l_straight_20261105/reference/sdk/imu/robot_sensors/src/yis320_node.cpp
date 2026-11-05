/*
 * @Description: imu yis320 node main function
 * @Version: V1.0
 * @Author: zw_1520@163.com
 * @Date: 2025-12-02 15:51:14
 * @LastEditors: zw_1520@163.com
 * @LastEditTime: 2025-12-03 09:22:33
 * Copyright (C) 2024-2050 Lens All rights reserved.
 */

#include "yis320_controller.h"

int main(int argc, char *argv[]) {
  // Initializes node
  rclcpp::init(argc, argv);
  auto g_node_ptr = std::make_shared<ImuYis320Node>();

  // Creates executor
  auto executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();

  // Adds node to executor
  executor->add_node(g_node_ptr);

  // Spins node
  executor->spin();

  // Shutdowns node
  rclcpp::shutdown();
  return 0;
}
