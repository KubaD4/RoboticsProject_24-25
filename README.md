# UR5 Simulation Repository

This repository provides the necessary materials to run the project work for the course *Fundamentals of Robotics* at the University of Trento for the academic year 2024-2025.
The project was developed by Alesandro Nardin, Kuba Di Quattro and Francesco Martella.

---

![demo.gif](demo.gif)

## Prerequisites
- **Docker**: Ensure Docker is installed and running on your system.

## How to Use
### 1. Start the UR5 Gazebo Simulation
Run the ROS 2 container using the provided bash script:
```bash
bash scripts/ros2.sh
```
This starts the [pla10/ros2_ur5_interface](https://hub.docker.com/r/pla10/ros2_ur5_interface) container. Access the environment via noVNC at [http://localhost:6081](http://localhost:6081).

- Open a terminal inside the ROS 2 container (accessible via noVNC).
- Navigate to the ROS 2 workspace:
  ```bash
  cd /home/ubuntu/ros2_ws
  ```
- Source the ROS 2 setup:
  ```bash
  source install/setup.bash
  ```
- Launch the simulation nodes using the provided launch file:
  ```bash
  ros2 launch ros2_ur5_interface sim.launch.py
  ```
- Add the block models to the simulation. It is necessary to enable visualization for the block models. You can do so by adding a *RobotModel* from the *Add* menu, selecting *Topic* as the description source, and then choosing the correct topic. Currently, the simulation spawns two blocks.

### 2. Build the Source Code
Before running the necessary nodes, build the source code using:
```bash
colcon build
```

### 3. Run Necessary Nodes
Now, you need to run the three nodes that control the robot. Currently, there is no launch file to automate this process, so each node must be started manually in its own terminal. Remember to source the ROS 2 setup in each terminal:
```bash
source install/setup.bash
```

The three nodes to run are:
```bash
ros2 run ur5_motion_cpp grabmove_server
```
```bash
ros2 run localization detector
```
```bash
ros2 run ur5_planning planning1
```

