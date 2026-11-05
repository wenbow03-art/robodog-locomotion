#include <fstream>
#include <iostream>
// #include <nlohmann/json.hpp>
#include "joint_actuator/json.hpp"
#include "joint_actuator/joint_actuator_node.hpp"
#include "joint_actuator/logger.h"

void JointActuatorNode::init()
{
  LOG_INFO("Enter JointActuatorNode::init.");

  motor_thread_ = std::thread(&JointActuatorNode::motor_thread,  this);

  LOG_INFO("Leave JointActuatorNode::init.");
}

JointActuatorNode::JointActuatorNode() : Node("joint_actuator_node")
{
	std::string package_path = ament_index_cpp::get_package_share_directory("joint_actuator");
  auto yaml_path = package_path + "/actuator_config/robot_default_config.yaml";
  YAML::Node config;
  config = YAML::LoadFile(yaml_path);

  robot_type_str_ = config["robot_default_config"]["robot_type"].as<std::string>();
  LOG_INFO("robot_type: {}", robot_type_str_);

  robot_model_str_ = config["robot_default_config"]["robot_model"].as<std::string>();
  LOG_INFO("robot_model: {}", robot_model_str_);

  ethercat_device_str_ = config["robot_default_config"]["ethercat_device"].as<std::string>();
  LOG_INFO("ethercat_device: {}", ethercat_device_str_);
	
  enable_KpKd_ = config["robot_default_config"]["enable_test"]["enable_KpKd"].as<int>();
	LOG_INFO("enable_KpKd_: {}", enable_KpKd_);

	// 创建Publisher
	joint_state_publisher_ = this->create_publisher<JointCommand>(
	"/joint_actuator_states", rclcpp::QoS(10));
	// 创建Subscriber    
	joint_command_subscriber_ = this->create_subscription<JointCommand>(
											"/joint_actuator_commands", rclcpp::QoS(10),
							std::bind(&JointActuatorNode::joint_command_callback, this, std::placeholders::_1));

	soem_state_publisher_ = create_publisher<std_msgs::msg::String>("/tts_play", 10);
  joint_actuator_data_publisher_ = create_publisher<JointActuatorData>("/joint_actuator_other_data", 10);

  error_count_ = 0;
}

JointActuatorNode::~JointActuatorNode()
{
	if (motor_thread_.joinable()) {
    motor_thread_.join();
    LOG_INFO("motor_thread joined.");
  }

	// 第1步：立即停止回调
  joint_command_subscriber_.reset();

  // 第2步：释放发布器
  joint_state_publisher_.reset();
  soem_state_publisher_.reset();
  joint_actuator_data_publisher_.reset();
  LOG_INFO("Publishers and Subscriber have been reset.");
}

bool JointActuatorNode::SetRealTimeThread(pthread_t pid, std::string name, int rt_priority, int bind_cpu) 
{
  bool ret = true;
  if(name.size()) 
  {
    pthread_setname_np(pid, name.c_str());
  }
  // set rt-thread.
  if(rt_priority >= 0) 
  {
      int max = sched_get_priority_max(SCHED_FIFO);
       if(rt_priority > max) rt_priority = max;
      struct sched_param s_parm;
      s_parm.sched_priority = rt_priority;
      if(pthread_setschedparam(pid, SCHED_FIFO, &s_parm) < 0) 
      {
        LOG_ERROR("setschedu error {}", strerror(errno));
        ret = false;
      } 
      else 
      {
        LOG_DEBUG("Thread {} set priority: {}", name.c_str(), rt_priority);
      }
  }

  // bind cpu core
  if(bind_cpu >= 0) 
  {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(bind_cpu, &cpuset);
    if(pthread_setaffinity_np(pid, sizeof(cpuset), &cpuset) != 0) 
    {
      LOG_ERROR("setaffinity error {}", strerror(errno));
      ret = false;
    } 
    else 
    {
      LOG_DEBUG("Thread {} bind cpu core: {}", name.c_str(), bind_cpu);
    }
  }
  return ret;
}

void JointActuatorNode::joint_command_callback(const JointCommand& msg) 
{
	std::lock_guard<std::mutex> lock(joint_command_mutex_);
  for (int i = 0; i < whole_body_joint_names_.size(); i++) 
  {
    auto it = std::find(msg.name.begin(), msg.name.end(), whole_body_joint_names_[i]);
    if (it != msg.name.end()) 
    {
      size_t idx = std::distance(msg.name.begin(), it);// 计算对应的索引位置
      joint_command_.position[i] = (msg.position[idx])*joint_direction_[i];
      joint_command_.kp[i] = (msg.kp[idx])/std::pow(joint_direction_[i],2);
      joint_command_.kd[i] = (msg.kd[idx])/std::pow(joint_direction_[i],2);
    } 
    else 
    {
      LOG_ERROR("未找到关节: {}", whole_body_joint_names_[i].c_str());
      is_motor_error_ = true;
      return;
    }
    
  }
}

void JointActuatorNode::setActuatorController()
{
	LOG_INFO("Enter JointActuatorNode::setActuatorController.");
  //motor config file path 
  std::string package_path = ament_index_cpp::get_package_share_directory("joint_actuator");
  std::string motor_config_file = package_path + "/actuator_config/" + robot_type_str_ + "_" + robot_model_str_ + "_actuator/actuator_config.json";
  std::string actuator_params_file = package_path + "/actuator_config/" + robot_type_str_ + "_" + robot_model_str_ + "_actuator/actuator_params_config.json";

  //从配置文件中读取joint_names
	std::ifstream config_stream(motor_config_file);
	nlohmann::json config_json;
	config_stream >> config_json;

	// 清空现有的joint_names
	whole_body_joint_names_.clear();

	// 从配置文件中提取所有joint_name
	if (config_json.contains("actuator_list") && config_json["actuator_list"].is_array()) {
			for (const auto& actuator_obj : config_json["actuator_list"]) {
					if (actuator_obj.contains("can_id_list") && actuator_obj["can_id_list"].is_array()) {
							for (const auto& can_id_obj : actuator_obj["can_id_list"]) {
									if (can_id_obj.contains("joint_name")) {
											std::string joint_name = can_id_obj["joint_name"];
											whole_body_joint_names_.push_back(joint_name);
									}
							}
					}
			}
	}
	joint_num_all_ = whole_body_joint_names_.size();

  // 从配置文件中读取joint_direction
  auto yaml_path1 = package_path + "/actuator_config/" + robot_type_str_ + "_" + robot_model_str_ + "_actuator/joint_config.yaml";
  YAML::Node config1;
  config1 = YAML::LoadFile(yaml_path1);

  auto joint_drt_map = config1["joint_config"]["joint_direction"].as<std::map<std::string, double>>();
  auto joint_lower_map = config1["joint_config"]["joint_lower"].as<std::map<std::string, double>>();
  auto joint_upper_map = config1["joint_config"]["joint_upper"].as<std::map<std::string, double>>();

  joint_direction_.clear();
  
  for (size_t i = 0; i <whole_body_joint_names_.size(); ++i) 
  {
    joint_direction_.push_back(joint_drt_map[whole_body_joint_names_[i]]);// 按照 whole_body_joint_names_ 的顺序赋值
    joint_lower_.push_back(joint_lower_map[whole_body_joint_names_[i]] * deg2rad_);  // 按照 whole_body_joint_names_ 的顺序赋值
    joint_upper_.push_back(joint_upper_map[whole_body_joint_names_[i]] * deg2rad_);  // 按照 whole_body_joint_names_ 的顺序赋值
    LOG_INFO("joint_name: {}, direction: {}, lower: {}, upper: {}",whole_body_joint_names_[i], 
      joint_direction_[i], joint_lower_map[whole_body_joint_names_[i]], joint_upper_map[whole_body_joint_names_[i]]);
  }

  // 初始化ActuatorController
  ActuatorController::loadActuatorParamsConfig(actuator_params_file);
  actuator_controller_ = std::make_shared<ActuatorController>();
	
  std_msgs::msg::String tts_msg;
  tts_msg.data = "正在初始化电机，请稍候！";
  soem_state_publisher_->publish(tts_msg);

  while(actuator_controller_->initCanIdAndEthercatRelation(ethercat_device_str_,motor_config_file,500) == false && rclcpp::ok())
  {
    tts_msg.data = "初始化电机失败，请检查电机连接！";
    soem_state_publisher_->publish(tts_msg);
    
    LOG_WARN("JointActuatorNode::setActuatorController: Retry to initialize ActuatorController");
    usleep(3*1000*1000);
  }
  tts_msg.data = "电机初始化成功！";
  soem_state_publisher_->publish(tts_msg);

  //设置电机模式
  std::vector<MotorModeEnum> modes;
  for (int i = 0; i < joint_num_all_; i++)
  {
    modes.push_back(MotorModeEnum::MODE_MIT);
  }

  for(int retry_cnt = 0; retry_cnt < 5; retry_cnt++)
  {
    if(actuator_controller_->setMotorMode(whole_body_joint_names_, modes))
    {
      break;
    }
    LOG_WARN("JointActuatorNode::setActuatorController: Retry to set motor mode.");
    usleep(500*1000);
  }

  actuator_controller_->enableMotor(whole_body_joint_names_);
  actuator_controller_->setLogLevel(3);

	//初始化command值
	joint_command_.name = whole_body_joint_names_;
	std::vector<double> position;
  std::vector<double> velocity;
  std::vector<double> effort;
	actuator_controller_->getCurrentPosition(whole_body_joint_names_, position);
  actuator_controller_->getCurrentVelocity(whole_body_joint_names_, velocity);
  actuator_controller_->getCurrentTorque(whole_body_joint_names_, effort);
  position_last_ = position;
  effort_last_ = effort;
	LOG_INFO("*****************************************");
  for (int i = 0; i < whole_body_joint_names_.size(); i++)
  {
    joint_command_.position.push_back(position[i]);
		LOG_INFO("joint_name: {}, position: {}", whole_body_joint_names_[i].c_str(), position[i]);
    joint_command_.velocity.push_back(velocity[i]);
    joint_command_.effort.push_back(effort[i]);
    joint_command_.kp.push_back(100);
    joint_command_.kd.push_back(1);
  }

	LOG_INFO("Leave JointActuatorNode::setActuatorController.");
}

void JointActuatorNode::sendTargetMit()
{
	std::vector<double> command_position(joint_num_all_,0.0);
  std::vector<double> command_velocity(joint_num_all_,0.0);
	std::vector<double> command_effort(joint_num_all_,0.0);
  std::vector<double> kps(joint_num_all_,0.0);
  std::vector<double> kds(joint_num_all_,0.0);

  if (enable_KpKd_ == true)
  {
    std::lock_guard<std::mutex> lock(joint_command_mutex_);
    for (size_t i = 0; i <whole_body_joint_names_.size(); ++i) 
    {
      auto it = std::find(joint_command_.name.begin(), joint_command_.name.end(), whole_body_joint_names_[i]);

      if (it != joint_command_.name.end()) 
      {
        size_t idx = std::distance(joint_command_.name.begin(), it);// 计算对应的索引位置
        command_position[i] = joint_command_.position[idx];// 按照 joint_name_ 的顺序赋值
        // command_velocity[i] = joint_command_.velocity[idx];
        // command_effort[i] = joint_command_.effort[idx];
        kps[i] = joint_command_.kp[idx];
        kds[i] = joint_command_.kd[idx];
      } 
      else 
      {
        LOG_ERROR("未找到关节: {}", whole_body_joint_names_[i].c_str());
        is_motor_error_ = true;
        return;
      }
    }
  }

  auto state_message = JointCommand();
  state_message.header.stamp = this->now();
  state_message.name = whole_body_joint_names_;
  std::vector<double> position;
  std::vector<double> velocity;
  std::vector<double> effort;
  actuator_controller_->getCurrentPosition(whole_body_joint_names_, position);
  actuator_controller_->getCurrentVelocity(whole_body_joint_names_, velocity);
  actuator_controller_->getCurrentTorque(whole_body_joint_names_, effort);

  //判断position是否越界
  bool is_position_over_limit = false;
  for (size_t i = 0; i < whole_body_joint_names_.size(); ++i)
  {
    if (position[i] < joint_lower_[i])
    {  
      if (log_count_ % log_interval_ == 0)
      {
        LOG_ERROR("*** joint {} position is over limit, joint_lower:{}, current_position:{}", 
                  whole_body_joint_names_[i].c_str(), joint_lower_[i] / deg2rad_, position[i] / deg2rad_);
      }
      is_position_over_limit = true;
    }
    else if (position[i] > joint_upper_[i])
    {
      if (log_count_ % log_interval_ == 0)
      {
        LOG_ERROR("*** joint {} position is over limit, joint_upper_:{}, current_position:{}", 
                  whole_body_joint_names_[i].c_str(), joint_upper_[i] / deg2rad_, position[i] / deg2rad_);
      }
      is_position_over_limit = true;
    }
  }

  if (is_position_over_limit == false)
  {
    actuator_controller_->setTargetMit(whole_body_joint_names_, command_position, command_velocity, command_effort, kps, kds);
  }
  
  bool is_connection_error = true;
  for (int i = 0; i < whole_body_joint_names_.size(); i++)
  {
    state_message.position.push_back(position[i] / joint_direction_[i]);
    state_message.velocity.push_back(velocity[i] / joint_direction_[i]);
    state_message.effort.push_back(effort[i] * joint_direction_[i]);
    state_message.kp.push_back(0);
    state_message.kd.push_back(0);
    if (fabs(position[i] - position_last_[i]) > 0.000000001 || fabs(effort[i] - effort_last_[i]) > 0.000000001)
    {
      is_connection_error = false;
    }
  }
  position_last_ = position;
  effort_last_ = effort;
	//pub state message
  joint_state_publisher_->publish(state_message);

  if (is_connection_error == true)
  {
    error_count_++;
    if (error_count_ >= 500)
    {
      is_motor_error_ = true;
      LOG_ERROR("*** check motor connection error, joint disconnect! ***");
    }
  }
  else
  {
    error_count_ = 0;
  }
}

//获取电机温度和在线状态，监测电机异常
void JointActuatorNode::getJointSelfData()
{
  auto joint_self_message = JointActuatorData();
  joint_self_message.motor_temp.resize(joint_num_all_, 0.0);
  joint_self_message.mos_temp.resize(joint_num_all_, 0.0);
  joint_self_message.online_statuses.resize(joint_num_all_, false);
  joint_self_message.error_codes.resize(joint_num_all_, 0);


  std::vector<double> motor_temp;
  std::vector<double> mos_temp;
  actuator_controller_->getMotorTemp(whole_body_joint_names_, motor_temp);
  actuator_controller_->getMosTemp(whole_body_joint_names_, mos_temp);

  std::vector<bool> online_statuses;
  actuator_controller_->getOnlineStatus(whole_body_joint_names_, online_statuses);

  std::vector<uint8_t> error_codes;
  actuator_controller_->getErrorCode(whole_body_joint_names_, error_codes);
  
  
  for (size_t i = 0; i < whole_body_joint_names_.size(); i++)
  {
    if (online_statuses[i] == false || error_codes[i] != 0)
    {
      if (i < 12)
      {
        is_motor_error_ = true;
        LOG_ERROR("*** current joint{} is offline:{}, error_code:{}, motor_temp:{}, mos_temp:{}", 
                      i, whole_body_joint_names_[i], error_codes[i], motor_temp[i], mos_temp[i]);
      }

      if (log_count_ % log_interval_ == 0)
      {
        log_count_ = 0;
        LOG_ERROR("*** current joint{} is offline:{}, error_code:{}, motor_temp:{}, mos_temp:{}", 
                      i, whole_body_joint_names_[i], error_codes[i], motor_temp[i], mos_temp[i]);
      }
      
      joint_self_message.exception_data.exception_id = (int)error_codes[i];
      joint_self_message.exception_data.exception_msg = "joint error detected, please check the joint:" 
                        + whole_body_joint_names_[i] + ",error code: " + std::to_string(error_codes[i]);
    }

    joint_self_message.motor_temp[i] = motor_temp[i];
    joint_self_message.mos_temp[i] = mos_temp[i];
    joint_self_message.online_statuses[i] = online_statuses[i];
    joint_self_message.error_codes[i] = (int)error_codes[i];
  }
  
  log_count_++;

  joint_actuator_data_publisher_->publish(joint_self_message);
}


// 电机控制线程，周期性发送命令并获取电机状态
void JointActuatorNode::motor_thread()
{
  LOG_INFO("Enter JointActuatorNode::motor_thread.");
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC, &start);
  SetRealTimeThread(pthread_self(), "motor_thread" , 5, 5);

	is_motor_error_ = true;
	setActuatorController();
	is_motor_error_ = false;
  
  int motor_hz = 1000 / motor_period_ms_;
  rclcpp::Rate rate(motor_hz);
  while (rclcpp::ok()) 
  {
		sendTargetMit();
    getJointSelfData();
    if(is_motor_error_ == true)
    {
      break;
    }
    
    rate.sleep(); // ros2 动态延时更准确
  }

  actuator_controller_->disableMotor(whole_body_joint_names_);
  
  usleep(100 * 1000);

  LOG_INFO("Leave JointActuatorNode::motor_thread.");
}



int main(int argc, char* argv[])
{
	LoggerActuator::getInstance().setLogLevel(LogLevel::INFO);
  LOG_INFO("*********************  Enter joint_actuator package  *********************");
	rclcpp::init(argc, argv);
	auto node = std::make_shared<JointActuatorNode>();

	node->init();

	rclcpp::spin(node);
	rclcpp::shutdown();
	LOG_INFO("*********************  Leave joint_actuator package  *********************");
	return 0;
}
