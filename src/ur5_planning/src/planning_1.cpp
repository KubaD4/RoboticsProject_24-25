/**
 * @file robot_controller.cpp
 * @defgroup ur5_planning Planning
 * @brief Implements a controller for UR5 robot pick-and-place operations.
 * @{
 */

#include "rclcpp/rclcpp.hpp"
#include "localization_interfaces/srv/block_info_all.hpp"
#include <string>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/point_stamped.hpp>
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "ur5_planning/action_client_wrapper.hpp"
#include <std_srvs/srv/trigger.hpp>
#include "ur5_motion_interfaces/action/grab_move.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <memory>
#include <vector>

constexpr double GRAB_RELEASE_HEIGHT = 0.87;  ///< Height for gripper operations
constexpr double FIRST_CENTER_X = 0.04;      ///< Initial deposit position X coordinate
constexpr double FIRST_CENTER_Y = 0.72;      ///< Initial deposit position Y coordinate
constexpr double CENTER_DISTANCE_X = 0.05;
constexpr double CENTER_DISTANCE_Y = 0.12;
constexpr int MAX_BLOCKS_Y = 5;

using namespace std::chrono_literals;

/**
 * @class RobotController
 * @brief Controls UR5 robot for block manipulation tasks
 * 
 * This class handles the complete pick-and-place sequence including:
 * - Service communication for block localization
 * - TF2 transformations for coordinate systems
 * - Gripper control through service calls
 * - Robot motion planning via action clients
 */
class RobotController {
public:
    /**
     * @brief Constructs a RobotController instance
     * @param node Shared pointer to the ROS2 node
     * 
     * Initializes service clients, action clients, and TF2 components.
     * Waits for required services to become available.
     */
    RobotController(rclcpp::Node::SharedPtr node) : node_(node) {
        RCLCPP_INFO(node_->get_logger(), "Initializing RobotController...");
        
        // Initialize TF2
        RCLCPP_INFO(node_->get_logger(), "Setting up TF2 buffer and listener...");
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Initialize service clients
        RCLCPP_INFO(node_->get_logger(), "Creating service clients...");
        block_info_client_ = node_->create_client<localization_interfaces::srv::BlockInfoAll>("/localization_service");
        open_gripper_client_ = node_->create_client<std_srvs::srv::Trigger>("open_gripper");
        close_gripper_client_ = node_->create_client<std_srvs::srv::Trigger>("close_gripper");
        
        // Initialize action client
        RCLCPP_INFO(node_->get_logger(), "Creating action client for robot movement...");
        action_client_ = std::make_unique<ActionClientWrapper<ur5_motion_interfaces::action::GrabMove>>(
            node_, "grabmove"
        );

        RCLCPP_INFO(node_->get_logger(), "Waiting for services to become available...");
        waitForServices();
        RCLCPP_INFO(node_->get_logger(), "RobotController initialization completed");

        stored_blocks = 0;
        detected_blocks = 0;
    }

    /**
     * @brief Executes the complete pick-and-place sequence
     * @return bool True if entire sequence completes successfully, false otherwise
     * 
     * Sequence steps:
     * 1. Gets block positions from localization service
     * 2. Performs pick-move-place operations for each detected block
     * 3. Handles gripper operations and neutral positioning
     */
    bool run() {
        RCLCPP_INFO(node_->get_logger(), "Starting robot control sequence...");

        RCLCPP_INFO(node_->get_logger(), "Opening gripper...");
        if (!openGripper()) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to open gripper, aborting sequence");
            return false;
        }

        //MOVE TO NEUTRAL POSITION
        RCLCPP_INFO(node_->get_logger(), "Moving to deposit position...");
        if (!moveToNeutral()) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to move to target, aborting sequence");
            return false;
        }
        
        RCLCPP_INFO(node_->get_logger(), "Getting block positions...");
        if (!getBlockPositions()) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to get block positions, aborting sequence");
            return false;
        }

        

        for (int i = 0; i < detected_blocks; i++) {
            //REPEAT FOR EACH BLOCK
            RCLCPP_INFO(node_->get_logger(), "Moving to target position...");
            if (!moveToTarget(i)) {
                RCLCPP_ERROR(node_->get_logger(), "Failed to move to target, aborting sequence");
                return false;
            }

            //CLOSE GRIPPER
            RCLCPP_INFO(node_->get_logger(), "Closing gripper...");
            if (!closeGripper()) {
                RCLCPP_ERROR(node_->get_logger(), "Failed to open gripper, aborting sequence");
                return false;
            }

            //PUT THE BLOCK IN THE RIGHT POSITION
            RCLCPP_INFO(node_->get_logger(), "Moving to deposit position...");
            if (!moveToDeposit(i)) {
                RCLCPP_ERROR(node_->get_logger(), "Failed to move to target, aborting sequence");
                return false;
            }

            //OPEN GRIPPER
            RCLCPP_INFO(node_->get_logger(), "Opening gripper...");
            if (!openGripper()) {
                RCLCPP_ERROR(node_->get_logger(), "Failed to open gripper, aborting sequence");
                return false;
            }

            //MOVE TO NEUTRAL POSITION
            RCLCPP_INFO(node_->get_logger(), "Moving to deposit position...");
            if (!moveToNeutral()) {
                RCLCPP_ERROR(node_->get_logger(), "Failed to move to target, aborting sequence");
                return false;
            }
            
            //READ AGAIN POSITIONS
            RCLCPP_INFO(node_->get_logger(), "Getting block positions...");
            if (!getBlockPositions()) {
                RCLCPP_ERROR(node_->get_logger(), "Failed to get block positions, aborting sequence");
                return false;
            }
            stored_blocks++;
        }
        

        //END
        RCLCPP_INFO(node_->get_logger(), "Robot control sequence completed successfully");
        return true;
    }

private:
    /// @brief Storage for detected block types
    std::vector<std::string> blocks_types;  
    /// @brief Storage for block orientations (quaternions)
    std::vector<Eigen::Quaterniond> blocks_rotation;
    /// @brief Storage for block positions in 3D space
    std::vector<Eigen::Vector3d> blocks_position;
    int stored_blocks;      ///< Count of successfully placed blocks
    int detected_blocks;    ///< Number of blocks detected in current cycle

    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Client<localization_interfaces::srv::BlockInfoAll>::SharedPtr block_info_client_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr open_gripper_client_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr close_gripper_client_;
    std::unique_ptr<ActionClientWrapper<ur5_motion_interfaces::action::GrabMove>> action_client_;
    
    geometry_msgs::msg::PoseStamped source_pose_;
    geometry_msgs::msg::PoseStamped target_pose_;

    /**
     * @brief Waits for required ROS services to become available
     * @throws std::runtime_error If services not available after multiple attempts
     */
    void waitForServices() {
        int attempt = 1;
        while (!block_info_client_->wait_for_service(1s)) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(node_->get_logger(), "Interrupted while waiting for services. Exiting.");
                throw std::runtime_error("Service connection failed");
            }
            RCLCPP_INFO(node_->get_logger(), "Service not available, attempt %d...", attempt++);
        }
    }

    /**
     * @brief Retrieves block positions from localization service
     * @return bool True if block positions successfully retrieved and transformed
     * 
     * - Calls localization service for block positions
     * - Transforms coordinates from camera frame to desk frame
     * - Stores positions, orientations and types in member variables
     */
    bool getBlockPositions() {
        RCLCPP_INFO(node_->get_logger(), "Requesting block information from localization service...");
        
        auto request = std::make_shared<localization_interfaces::srv::BlockInfoAll::Request>();
        auto result_future = block_info_client_->async_send_request(request);

        if (rclcpp::spin_until_future_complete(node_, result_future) != rclcpp::FutureReturnCode::SUCCESS) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to get response from block info service");
            return false;
        }

        auto result = result_future.get();
        RCLCPP_INFO(node_->get_logger(), "Received information for %ld blocks", result->positions_x.size());
        blocks_position.resize(result->positions_x.size());
        blocks_rotation.resize(result->positions_x.size());
        blocks_types.resize(result->positions_x.size());
        detected_blocks = result->positions_x.size();

        geometry_msgs::msg::TransformStamped transform;
        RCLCPP_INFO(node_->get_logger(), "Looking up transform from camera_rgb_frame to desk...");

        try {
            transform = tf_buffer_->lookupTransform("desk", "camera_rgb_frame", tf2::TimePointZero);
            RCLCPP_INFO(node_->get_logger(), "Transform lookup successful");
        } catch (const tf2::TransformException &ex) {
            RCLCPP_ERROR(node_->get_logger(), "Transform lookup failed: %s", ex.what());
            return false;
        }

        for (int i = 0; i < result->positions_x.size(); i++) {
            RCLCPP_INFO(node_->get_logger(), "Processing block %d...", i + 1);

            source_pose_.pose.position.x = result->positions_x[i];
            source_pose_.pose.position.y = result->positions_y[i];
            source_pose_.pose.position.z = result->positions_z[i];
            source_pose_.pose.orientation.w = result->orientations_w[i];
            source_pose_.pose.orientation.x = result->orientations_x[i];
            source_pose_.pose.orientation.y = result->orientations_y[i];
            source_pose_.pose.orientation.z = result->orientations_z[i];

            RCLCPP_DEBUG(node_->get_logger(), "Original position: [%.2f, %.2f, %.2f]",
                        source_pose_.pose.position.x,
                        source_pose_.pose.position.y,
                        source_pose_.pose.position.z);

            // Transform pose
            tf2::doTransform(source_pose_, target_pose_, transform);
            RCLCPP_INFO(
                node_->get_logger(), "Block %d transformed position: [%.2f, %.2f, %.2f]",
                i + 1,
                target_pose_.pose.position.x,
                target_pose_.pose.position.y,
                target_pose_.pose.position.z
            );

            // Fill position data
            Eigen::Vector3d position(
                target_pose_.pose.position.x,
                target_pose_.pose.position.y,
                target_pose_.pose.position.z
            );

            RCLCPP_INFO(node_->get_logger(), "test");
            blocks_position[i] = position;

            // Fill rotation data
            Eigen::Quaterniond rotation(
                target_pose_.pose.orientation.w,
                target_pose_.pose.orientation.x,
                target_pose_.pose.orientation.y,
                target_pose_.pose.orientation.z
            );

            // Isolate rotation along z axis (yaw)
            Eigen::Vector3d euler = rotation.toRotationMatrix().eulerAngles(2, 1, 0);  
            double yaw = euler[0];
            RCLCPP_INFO(node_->get_logger(), "Block %d yaw angle: %.2f degrees", i + 1, yaw * 180.0 / M_PI);
            
            Eigen::Quaterniond q_yaw(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
            blocks_rotation[i] = q_yaw;

            // Fill in type
            blocks_types[i] = result->block_names[0];
            RCLCPP_INFO(node_->get_logger(), "Block %d type: %s", i + 1, blocks_types[i].c_str());
        }

        RCLCPP_INFO(node_->get_logger(), "Block position processing completed");
        return true;
    }

    /**
     * @brief Commands gripper to open position
     * @return bool True if gripper opened successfully
     * 
     * - Calls open_gripper service
     * - Includes 2-second delay for operation completion
     */
    bool openGripper() {
        RCLCPP_INFO(node_->get_logger(), "Sending open gripper command...");
        
        auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
        auto result_future = open_gripper_client_->async_send_request(request);
        
        if (rclcpp::spin_until_future_complete(node_, result_future) != rclcpp::FutureReturnCode::SUCCESS) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to receive response from gripper service");
            return false;
        }

        RCLCPP_INFO(node_->get_logger(), "Waiting for gripper to fully open...");
        std::this_thread::sleep_for(2s);
        RCLCPP_INFO(node_->get_logger(), "Gripper opened successfully");
        return true;
    }

    /**
     * @brief Commands gripper to close position
     * @return bool True if gripper closed successfully
     * 
     * - Calls close_gripper service
     * - Includes 2-second delay for operation completion
     */
    bool closeGripper() {
        RCLCPP_INFO(node_->get_logger(), "Sending close gripper command...");
        
        auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
        auto result_future = close_gripper_client_->async_send_request(request);
        
        if (rclcpp::spin_until_future_complete(node_, result_future) != rclcpp::FutureReturnCode::SUCCESS) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to receive response from gripper service");
            return false;
        }

        RCLCPP_INFO(node_->get_logger(), "Waiting for gripper to fully close...");
        std::this_thread::sleep_for(2s);
        RCLCPP_INFO(node_->get_logger(), "Gripper closed successfully");
        return true;
    }

    /**
     * @brief Moves robot to target block position
     * @param i Index of target block in stored positions
     * @return bool True if movement completed successfully
     * 
     * - Calculates target orientation considering block rotation
     * - Uses GrabMove action for motion execution
     * - Positions gripper at predefined GRAB_RELEASE_HEIGHT
     */
    bool moveToTarget(int i) {
        RCLCPP_INFO(node_->get_logger(), "Preparing to move to target %d...", i + 1);
        
        // Calculate final orientation       
        Eigen::Quaterniond rot_x_180(0, 1, 0, 0);
        Eigen::Quaterniond rot_z_90(0.707, 0, 0, 0.707);
        Eigen::Quaterniond final_orientation = blocks_rotation[i] * rot_x_180 * rot_z_90;

        // Set up movement goal
        auto goal = ur5_motion_interfaces::action::GrabMove::Goal();
        goal.x = blocks_position[i].x();
        goal.y = blocks_position[i].y();
        goal.z = GRAB_RELEASE_HEIGHT;
        goal.r_x = final_orientation.x();
        goal.r_y = final_orientation.y();
        goal.r_z = final_orientation.z();
        goal.w = final_orientation.w();

        RCLCPP_INFO(node_->get_logger(), "Moving to position: [%.2f, %.2f, %.2f]", 
                    goal.x, goal.y, goal.z);
        RCLCPP_DEBUG(node_->get_logger(), "Target orientation (quaternion): [w: %.2f, x: %.2f, y: %.2f, z: %.2f]",
                    goal.w, goal.r_x, goal.r_y, goal.r_z);

        // Execute movement
        RCLCPP_INFO(node_->get_logger(), "Executing movement...");
        typename ActionClientWrapper<ur5_motion_interfaces::action::GrabMove>::Result result;
        bool success = action_client_->sendGoal(goal, result);
        
        if (success) {
            RCLCPP_INFO(node_->get_logger(), "Movement to target %d completed successfully", i + 1);
        } else {
            RCLCPP_ERROR(node_->get_logger(), "Movement to target %d failed", i + 1);
        }
        
        return success;
    }

    /**
     * @brief Moves robot to deposit position
     * @param i Index of target block (determines deposit location)
     * @return bool True if movement completed successfully
     * 
     * - Calculates deposit position based on stored blocks count
     * - Uses fixed orientation for block placement
     * - Maintains consistent height using GRAB_RELEASE_HEIGHT
     */
    bool moveToDeposit(int i) {
        RCLCPP_INFO(node_->get_logger(), "Preparing to move to deposit %d...", i + 1);
        
        // Calculate final drop orientation       
        Eigen::Quaterniond orientation(0, 1, 0, 0);
        Eigen::Quaterniond rot_z_90(0.707, 0, 0, 0.707);

        Eigen::Quaterniond final_orientation = orientation * rot_z_90;

        // Set up movement goal
        auto goal = ur5_motion_interfaces::action::GrabMove::Goal();
        goal.x = FIRST_CENTER_X + CENTER_DISTANCE_X*stored_blocks;
        goal.y = FIRST_CENTER_Y;
        goal.z = GRAB_RELEASE_HEIGHT;
        goal.r_x = final_orientation.x();
        goal.r_y = final_orientation.y();
        goal.r_z = final_orientation.z();
        goal.w = final_orientation.w();

        RCLCPP_INFO(node_->get_logger(), "Moving to position: [%.2f, %.2f, %.2f]", 
                    goal.x, goal.y, goal.z);
        RCLCPP_DEBUG(node_->get_logger(), "Target orientation (quaternion): [w: %.2f, x: %.2f, y: %.2f, z: %.2f]",
                    goal.w, goal.r_x, goal.r_y, goal.r_z);

        // Execute movement
        RCLCPP_INFO(node_->get_logger(), "Executing movement...");
        typename ActionClientWrapper<ur5_motion_interfaces::action::GrabMove>::Result result;
        bool success = action_client_->sendGoal(goal, result);
        
        if (success) {
            RCLCPP_INFO(node_->get_logger(), "Movement to target %d completed successfully", i + 1);
        } else {
            RCLCPP_ERROR(node_->get_logger(), "Movement to target %d failed", i + 1);
        }
        
        return success;
    }

    /**
     * @brief Moves robot to neutral safe position
     * @return bool True if movement completed successfully
     * 
     * - Uses predefined fixed position and orientation
     * - Position chosen to avoid collisions during scanning
     */
    bool moveToNeutral() {
        RCLCPP_INFO(node_->get_logger(), "Moving to neutral position");

        Eigen::Quaterniond final_orientation(0, 1, 0, 0);

        // Set up movement goal
        auto goal = ur5_motion_interfaces::action::GrabMove::Goal();
        goal.x = 0.8;
        goal.y = 0.8;
        goal.z = 1.4;
        goal.r_x = final_orientation.x();
        goal.r_y = final_orientation.y();
        goal.r_z = final_orientation.z();
        goal.w = final_orientation.w();

        RCLCPP_INFO(node_->get_logger(), "Moving to position: [%.2f, %.2f, %.2f]", 
                    goal.x, goal.y, goal.z);
        RCLCPP_DEBUG(node_->get_logger(), "Target orientation (quaternion): [w: %.2f, x: %.2f, y: %.2f, z: %.2f]",
                    goal.w, goal.r_x, goal.r_y, goal.r_z);

        // Execute movement
        RCLCPP_INFO(node_->get_logger(), "Executing movement...");
        typename ActionClientWrapper<ur5_motion_interfaces::action::GrabMove>::Result result;
        bool success = action_client_->sendGoal(goal, result);
        
        if (success) {
            RCLCPP_INFO(node_->get_logger(), "Movement to target completed");
        } else {
            RCLCPP_ERROR(node_->get_logger(), "Movement to target finished");
        }
        
        return success;
    }
};

/**
 * @brief Main function for robot control application
 * 
 * - Initializes ROS2 node
 * - Creates RobotController instance
 * - Executes main control sequence
 */
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    RCLCPP_INFO(rclcpp::get_logger("main"), "Starting robot control application...");
    
    auto node = rclcpp::Node::make_shared("robot_controller");
    
    try {
        RCLCPP_INFO(node->get_logger(), "Creating RobotController instance...");
        RobotController controller(node);
        
        RCLCPP_INFO(node->get_logger(), "Starting robot control sequence...");
        if (controller.run()) {
            RCLCPP_INFO(node->get_logger(), "Robot control sequence completed successfully");
        } else {
            RCLCPP_ERROR(node->get_logger(), "Robot control sequence failed");
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node->get_logger(), "Fatal error: %s", e.what());
        return 1;
    }

    RCLCPP_INFO(node->get_logger(), "Shutting down robot control application...");
    rclcpp::shutdown();
    return 0;
}

/** @} */
