# 验证范围（2026-11-05）

- `cpu-python310.json`：新建Python3.10虚拟环境，安装CPU PyTorch2.5.1和NumPy1.26.4后，用Python隔离模式 `-I` 执行 `deployment/check.py --model`。不读取服务器PYTHONPATH或用户site-packages；两轮各290步合成观测，形状/数值/状态及重置一致性通过。
- `cpu-python310-freeze.txt`：该独立环境的完整依赖版本，由requirements-cpu.txt引用。
- `ros2-localhost.json`：厂商消息包在ROS2 Humble / 系统Python3.10编译成功；仅localhost的独立ROS域193中发送合成关节、IMU、诊断消息，采集器收到全部三种消息且样本合法。没有消息时按约定返回退出码2。未连接实机、未发布电机指令。
- Git暂存区的35个部署包文件与SHA256SUMS逐字节一致；13个原始ROS消息包文件与SOURCE.json一致。原始SDK/实现快照中的CRLF和尾部空格按源文件保留，新增工具和入口文档单独检查格式。

这些结果证明文件与软件接口可用，不证明板端实时性、坐标标定或实机运动效果。模型原环境的检查结果另存于部署包的validation/local_checks.json。
