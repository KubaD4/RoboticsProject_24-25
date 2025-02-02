/**
 * @file grab_move_action_server.cpp
 * @defgroup ur5_motion_cpp Motion
 * @brief Action server for executing GrabMove actions for the UR5 robot.
 *
 * This file implements a ROS2 action server node that computes a Cartesian trajectory for the UR5 robot.
 * It performs forward and inverse kinematics using the KDL library, transforms poses between base and desk
 * frames, and sends the joint trajectory to a FollowJointTrajectory action server.
 *
 * Trajectory parameters (velocities, acceleration time, etc.) are defined as constants at the beginning of the file.
 *
 * @author Alessandro Nardin
 * @date 02/02/2025
 * 
 * @{
 */

// Standard Library
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>  

// ROS and related libraries
#include "geometry_msgs/msg/point_stamped.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_eigen/tf2_eigen.hpp"
#include "rclcpp/duration.hpp"

// Control and trajectory
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <control_msgs/msg/joint_tolerance.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

// KDL library
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>

// Project-specific includes
#include "ur5_motion_cpp/visibility_control.h"
#include "ur5_motion_interfaces/action/grab_move.hpp"

// Eigen
#include <Eigen/Dense>

// Trajectory constant velocities in m/s, distances in m, times in s
const double MOV_PLANE_Z = 1.3;         ///< Constant Z height for via points in the trajectory
const double HOR_V = 0.125;             ///< Horizontal velocity (m/s)
const double VER_V = 0.08;              ///< Vertical velocity (m/s)
const double ACC_TIME = 3;              ///< Acceleration time (s)
const double INTERPOLATION_POINTS = 15; ///< Number of interpolation points for the trajectory

using namespace std::chrono_literals;
using namespace KDL;
using namespace Eigen;

namespace ur5_motion_cpp
{
/**
 * @class GrabMoveActionServer
 * @brief Action server node to execute a GrabMove action.
 * @ingroup ur5_motion_cpp
 *
 * This class implements a ROS2 action server that listens for GrabMove action goals. When a goal is received,
 * it transforms the goal pose from the desk frame to the robot base frame, computes the inverse kinematics using
 * the KDL solver, and generates a joint trajectory. The trajectory is then sent to a FollowJointTrajectory action
 * client for execution.
 */
class GrabMoveActionServer : public rclcpp::Node
{
public:
  /// Alias for the FollowJointTrajectory action.
  using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
  /// Alias for the FollowJointTrajectory goal handle.
  using GoalHandleFollowJointTrajectory = rclcpp_action::ClientGoalHandle<FollowJointTrajectory>;
  /// Alias for the GrabMove action.
  using GrabMove = ur5_motion_interfaces::action::GrabMove;
  /// Alias for the GrabMove goal handle.
  using GoalHandleGrabMove = rclcpp_action::ServerGoalHandle<GrabMove>;

  /**
   * @brief Constructor for the GrabMoveActionServer node.
   * @param options Optional ROS2 node options.
   *
   * This constructor initializes the kinematics solvers, subscribes to joint state updates, sets up
   * the TF listener for frame transformations, waits for the FollowJointTrajectory action server, and
   * creates the GrabMove action server.
   */
  UR5_MOTION_CPP_PUBLIC
  explicit GrabMoveActionServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("grabmove_action_server", options)
  {
    using namespace std::placeholders;
    RCLCPP_INFO(this->get_logger(), "Preparing server");
    // KINEMATICS SOLVERS
    RCLCPP_INFO(this->get_logger(), "Building kinematics solvers");
    chain_.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(0.0,     1.57079632679,  0.1625, 0.0)));
    chain_.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(-0.425,  0.0,            0.0,    0.0)));
    chain_.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(-0.3922, 0.0,            0.0,    0.0)));
    chain_.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(0.0,     1.57079632679,  0.1333, 0.0)));
    chain_.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(0.0,     -1.57079632679, 0.0997, 0.0)));
    chain_.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(0.0,     0.0,            0.0996, 0.0)));

    fk_solver_ = std::make_shared<KDL::ChainFkSolverPos_recursive>(chain_);
    ik_solver_ = std::make_shared<KDL::ChainIkSolverPos_LMA>(chain_);

    // CURRENT JOINT CONFIGURATION
    joint_names_ = {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
    q_curr_ = JntArray(joint_names_.size());

    // Subscribe to joint states
    joint_state_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "joint_states",
      10,
      std::bind(
        &GrabMoveActionServer::joint_state_callback,
        this,
        std::placeholders::_1
      )
    );

    // FRAME TRANSFORMATIONS
    RCLCPP_INFO(this->get_logger(), "Getting frame transformations");

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    sleep(2);

    // GET BASE -> DESK TRANSFORMATION
    geometry_msgs::msg::TransformStamped base_to_desk_tr = tf_buffer_->lookupTransform(
      "desk", "base", tf2::TimePointZero);

    // Convert the transform to an Eigen Isometry3d
    base_to_desk_ = tf2::transformToEigen(base_to_desk_tr);

    // GET DESK -> BASE TRANSFORMATION
    geometry_msgs::msg::TransformStamped desk_to_base_tr = tf_buffer_->lookupTransform(
      "base", "desk", tf2::TimePointZero);

    // Convert the transform to an Eigen Isometry3d
    desk_to_base_ = tf2::transformToEigen(desk_to_base_tr);

    // FOLLOW TRAJECTORY CLIENT
    RCLCPP_INFO(this->get_logger(), "Preparing action server");
    action_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
      this, "/scaled_joint_trajectory_controller/follow_joint_trajectory");
    
    if (!action_client_->wait_for_action_server(10s))
    {
        RCLCPP_ERROR(this->get_logger(), "Action server not available after waiting");
        rclcpp::shutdown();
        return;
    }

    // MOVEGRIP ACTION SERVER

    // Define the goal callback to handle new GrabMove goals.
    auto handle_goal = [this](
      const rclcpp_action::GoalUUID & uuid,
      std::shared_ptr<const GrabMove::Goal> goal)
    {
      (void)uuid;
      RCLCPP_INFO(
        this->get_logger(),
        "Received goal request with (desk) x:%f y:%f z:%f r_x:%f r_y:%f r_z:%f, w:%f",
        goal->x, goal->y, goal->z, goal->r_x, goal->r_y, goal->r_z, goal->w
      );

      // Ensure the desired end position is reachable
      Vector3d desk_goal_p(goal->x, goal->y, goal->z);
      Quaterniond desk_goal_r(goal->w, goal->r_x, goal->r_y, goal->r_z);
      
      Vector3d base_goal_p = desk_to_base_ * desk_goal_p;
      Quaterniond base_goal_r = Quaterniond(desk_to_base_.rotation()) * desk_goal_r;
      RCLCPP_INFO(
        this->get_logger(),
        "Transformed goal (base) x:%f y:%f z:%f r_x:%f r_y:%f r_z:%f, w:%f",
        base_goal_p.x(), base_goal_p.y(), base_goal_p.z(), 
        base_goal_r.x(), base_goal_r.y(), base_goal_r.z(), base_goal_r.w()
      );

      KDL::JntArray q_out(6);
      int result = inv_kin(
        base_goal_p.x(), base_goal_p.y(), base_goal_p.z(),
        base_goal_r.x(), base_goal_r.y(), base_goal_r.z(), base_goal_r.w(),
        q_out, q_curr_
      );

      if ( result != 0) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Goal position is not reachable, goal rejected. Inverse kinematics error code:%d",
          result
        );
        return rclcpp_action::GoalResponse::REJECT;
      }

      return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    };

    // Define the cancel callback.
    auto handle_cancel = [this](
      const std::shared_ptr<GoalHandleGrabMove> goal_handle)
    {
      (void)goal_handle;
      RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
      return rclcpp_action::CancelResponse::ACCEPT;
    };

    // Define the accepted callback that launches goal execution in a new thread.
    auto handle_accepted = [this](
      const std::shared_ptr<GoalHandleGrabMove> goal_handle)
    {
      // This lambda returns quickly to avoid blocking the executor.
      auto execute_in_thread = [this, goal_handle](){ return this->execute(goal_handle); };
      std::thread{execute_in_thread}.detach();
    };

    // Create the GrabMove action server.
    this->action_server_ = rclcpp_action::create_server<GrabMove>(
      this,
      "grabmove",
      handle_goal,
      handle_cancel,
      handle_accepted);
  
    RCLCPP_INFO(this->get_logger(), "Server Ready");
  }

private:
  rclcpp_action::Server<GrabMove>::SharedPtr action_server_; ///< GrabMove action server.
  rclcpp_action::Client<FollowJointTrajectory>::SharedPtr action_client_; ///< FollowJointTrajectory action client.
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_; ///< Subscriber for joint state messages.
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_; ///< TF listener for frame transformations.
  std::shared_ptr<ChainFkSolverPos_recursive> fk_solver_;  ///< Forward kinematics solver.
  std::shared_ptr<ChainIkSolverPos_LMA> ik_solver_;          ///< Inverse kinematics solver.
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;               ///< TF buffer for transformations.
  std::vector<std::string> joint_names_;                     ///< Names of the robot joints.
  Eigen::Isometry3d base_to_desk_;                           ///< Transformation from the base frame to the desk frame.
  Eigen::Isometry3d desk_to_base_;                           ///< Transformation from the desk frame to the base frame.
  KDL::Chain chain_;                                         ///< KDL chain for the robot.
  JntArray q_curr_;                                          ///< Current joint configuration.

  /**
   * @brief Executes the GrabMove goal.
   * @param goal_handle Pointer to the goal handle.
   *
   * This method calculates the Cartesian trajectory based on the starting and desired end poses,
   * computes the corresponding joint trajectory via inverse kinematics, and sends the trajectory to
   * the FollowJointTrajectory action client.
   */
  void execute(const std::shared_ptr<GoalHandleGrabMove> goal_handle) {
    RCLCPP_INFO(this->get_logger(), "Executing goal");
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<GrabMove::Feedback>();

    // START POSITION AND ROTATION (BASE FRAME)
    KDL::Frame kdl_start_frame;
    dir_kin(q_curr_(0), q_curr_(1), q_curr_(2), q_curr_(3), q_curr_(4), q_curr_(5), kdl_start_frame);

    KDL::Rotation kdl_start_r = kdl_start_frame.M;
    KDL::Vector kdl_start_p = kdl_start_frame.p;

    Eigen::Vector3d base_start_p(kdl_start_p(0), kdl_start_p(1), kdl_start_p(2));
    RCLCPP_INFO(
      this->get_logger(),
      "Starting position (base frame) is x:%f y:%f z:%f",
      kdl_start_p(0), kdl_start_p(1), kdl_start_p(2)
    );
    
    double x;
    double y;
    double z;
    double w;
    kdl_start_r.GetQuaternion( x, y, z, w );

    Eigen::Quaterniond base_start_r(w, x, y, z);
    RCLCPP_INFO(
      this->get_logger(),
      "Starting rotation (base frame) is x:%f y:%f z:%f w:%f",
      x, y, z, w
    );
    
    // START POSITION AND ROTATION (DESK FRAME)
    Eigen::Vector3d desk_start_p = base_to_desk_ * base_start_p;
    RCLCPP_INFO(
      this->get_logger(),
      "Starting position (desk frame) is x:%f y:%f z:%f",
      desk_start_p(0), desk_start_p(1), desk_start_p(2)
    );

    Eigen::Quaterniond desk_start_r = Eigen::Quaterniond(base_to_desk_.rotation()) * base_start_r;
    RCLCPP_INFO(
      this->get_logger(),
      "Starting rotation (desk frame) is x:%f y:%f z:%f w:%f",
      desk_start_r.x(), desk_start_r.y(), desk_start_r.z(), desk_start_r.w()
    );
    
    // FINAL POSITION AND ROTATION (DESK FRAME)
    Eigen::Vector3d desk_final_p(goal->x, goal->y, goal->z);
    RCLCPP_INFO(
      this->get_logger(),
      "Final position (desk frame) is x:%f y:%f z:%f",
      goal->x, goal->y, goal->z
    );

    Eigen::Quaternion desk_final_r(goal->w, goal->r_x, goal->r_y, goal->r_z);
    RCLCPP_INFO(
      this->get_logger(),
      "Final rotation (desk frame) is x:%f y:%f z:%f w:%f",
      goal->r_x, goal->r_y, goal->r_z, goal->w
    );

    // TRAJECTORY CALCULATION
    RCLCPP_INFO(this->get_logger(), "Begin cartesian trajectory calculation");
    trajectory_msgs::msg::JointTrajectory trajectory;
    trajectory.joint_names = joint_names_;

    // COMPUTE TRAJECTORY POINTS
    RCLCPP_INFO(this->get_logger(), "Begin point interpolation");
    
    // VIA-POINTS
    Vector3d via1(desk_start_p(0), desk_start_p(1), MOV_PLANE_Z );
    Vector3d via2(desk_final_p(0), desk_final_p(1), MOV_PLANE_Z);
    
    // DISTANCES
    double dist_s_v1 = fabs(desk_start_p(2)-MOV_PLANE_Z);
    double dist_v1_v2 = (via2-via1).norm();
    double dist_v2_e =  fabs(desk_final_p(2)- MOV_PLANE_Z);
    
    // TIMES (if only linear movement)
    double t1 = 0.0;
    double t2 = t1+ dist_s_v1/VER_V;
    double t3 = t2 + dist_v1_v2/HOR_V;
    double t4 = t3 + dist_v2_e/VER_V;
    
    // TIMES (considering accelerations)
    double t1p = t1 + ACC_TIME; 
    double t1pp = t2; 
    double t2p = t2 + ACC_TIME; 
    double t2pp = t3;  
    double t3p = t3 + ACC_TIME; 
    double t3pp = t4; 
    
    // t4 needs to be recalculated to compensate for the time lost during first and last accelerations.
    t4 = t3pp + ACC_TIME;

    // VELOCITY VECTORS
    Vector3d v12 = (via1 - desk_start_p)/(t2-t1);
    Vector3d v23 = (via2-via1) / (t3-t2);
    Vector3d v34 = (desk_final_p - via2) / (t4-t3);

    // ACCELERATION VECTORS
    Vector3d a1 = v12/ACC_TIME;
    Vector3d a2 = (v23 - v12) / ACC_TIME;
    Vector3d a3 = (v34 - v23) / ACC_TIME;
    Vector3d a4 = -v34 / ACC_TIME;

    // RELEVANT POSITIONS
    Vector3d pt1p = desk_start_p + a1 * std::pow(t1p, 2) / 2;
    Vector3d pt1pp = pt1p + v12 * (t1pp - t1p);
    Vector3d pt2p = pt1pp + v12 * (t2p - t1pp) + a2 * std::pow(t2p - t1pp, 2) / 2;
    Vector3d pt2pp = pt2p + v23 * (t2pp - t2p);
    Vector3d pt3p = pt2pp + v23 * (t3p - t2pp) + a3 * std::pow(t3p - t2pp, 2) / 2;
    Vector3d pt3pp = pt3p + v34 * (t3pp - t3p);

    double t_inc = t4 / INTERPOLATION_POINTS;
    KDL::JntArray q_prev = q_curr_;

    for (int i = 1; i <= INTERPOLATION_POINTS; ++i) {
      double t = t_inc * i;
      Vector3d desk_int_p;

      // POSITION interpolation with acceleration phases.
      if(t>= t1 && t<=t1p){
        desk_int_p = desk_start_p + 0.5 * a1 * pow(t,2);
      } else if(t>t1p && t<=t1pp){
        desk_int_p = pt1p + v12 * (t-t1p);
      } else if(t>t1pp && t<= t2p){
        desk_int_p = pt1pp + v12*(t-t1pp) + 0.5*a2*pow(t-t1pp,2);					
      } else if(t>t2p && t<=t2pp){
        desk_int_p = pt2p + v23*(t-t2p);
      } else if(t>t2pp && t<=t3p){
        desk_int_p = pt2pp + v23*(t-t2pp)+ 0.5*a3*pow(t-t2pp,2);
      } else if(t>t3p && t<=t3pp) {
        desk_int_p = pt3p + v34*(t-t3p);
      } else {
        desk_int_p = pt3pp + v34*(t-t3pp)+ 0.5*a4*pow(t-t3pp,2);
      }

      // ROTATION interpolation.
      Eigen::Quaterniond desk_int_r;
      if ( t <= t2 ) {
        desk_int_r = desk_start_r;
      } else if (  t > t2 && t  <= t3) {
        desk_int_r = desk_start_r.slerp((t-t2)/(t3-t2),desk_final_r);
      } else {
        desk_int_r = desk_final_r;
      }

      // Transform position and rotation from desk to base frame.
      Vector3d base_int_p = desk_to_base_ * desk_int_p;
      Eigen::Quaterniond base_int_r = Eigen::Quaterniond(desk_to_base_.rotation()) * desk_int_r;

      // Compute the inverse kinematics for the current interpolation point.
      KDL::JntArray joint_int_pr(6);
      int result = inv_kin(
        base_int_p.x(), base_int_p.y(), base_int_p.z(),
        base_int_r.x(), base_int_r.y(), base_int_r.z(), base_int_r.w(),
        joint_int_pr, q_prev
      );
      q_prev = joint_int_pr;

      if ( result != 0 ) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Error calculating inverse kinematics.\n Cartesian position: %f %f %f\n Rotation %f %f %f %f",
          base_int_p.x(), base_int_p.y(), base_int_p.z(),
          base_int_r.x(), base_int_r.y(), base_int_r.z(), base_int_r.w()
        );
        auto new_result = std::make_shared<GrabMove::Result>();
        new_result->error_code = GrabMove::Result::INVERSE_KINEMATICS_ERROR;
        goal_handle->abort(new_result);
        return;
      }

      // Add the computed joint positions to the trajectory.
      trajectory_msgs::msg::JointTrajectoryPoint traj_point;
      traj_point.positions.push_back(joint_int_pr(0));
      traj_point.positions.push_back(joint_int_pr(1));
      traj_point.positions.push_back(joint_int_pr(2));
      traj_point.positions.push_back(joint_int_pr(3));
      traj_point.positions.push_back(joint_int_pr(4));
      traj_point.positions.push_back(joint_int_pr(5));
      traj_point.time_from_start = rclcpp::Duration::from_seconds(t);

      trajectory.points.push_back(traj_point);
    }

    // Create a FollowJointTrajectory goal message.
    auto goal_msg = FollowJointTrajectory::Goal();
    goal_msg.trajectory = trajectory;
    goal_msg.goal_time_tolerance.nanosec = 50000000;

    // Define goal options for the trajectory action client.
    auto send_goal_options = rclcpp_action::Client<FollowJointTrajectory>::SendGoalOptions();
    send_goal_options.goal_response_callback =
      [this](const GoalHandleFollowJointTrajectory::SharedPtr &goal_handle) {
        if (!goal_handle) {
            RCLCPP_ERROR(this->get_logger(), "Goal was rejected by the server");
        } else {
            RCLCPP_INFO(this->get_logger(), "Goal accepted by the server, waiting for result");
        }
      };

    send_goal_options.result_callback =
      [this, goal_handle](const GoalHandleFollowJointTrajectory::WrappedResult &result) {
        auto new_result = std::make_shared<GrabMove::Result>();
        new_result->error_code = result.result->error_code;
        switch (result.code) {
          case rclcpp_action::ResultCode::SUCCEEDED:
            RCLCPP_INFO(this->get_logger(), "Goal succeeded");
            if (goal_handle) {
                goal_handle->succeed(new_result);
            }
            break;
          case rclcpp_action::ResultCode::ABORTED:
            RCLCPP_ERROR(this->get_logger(), "Goal was aborted, %d", result.result->error_code);
            RCLCPP_ERROR(this->get_logger(), "Error: %s", result.result->error_string.c_str());
            if (goal_handle) {
                goal_handle->abort(new_result);
            }
            break;

          case rclcpp_action::ResultCode::CANCELED:
            RCLCPP_WARN(this->get_logger(), "Goal was canceled");
            if (goal_handle) {
                goal_handle->canceled(new_result);
            }
            break;

          default:
            RCLCPP_ERROR(this->get_logger(), "Unknown result code");
            if (goal_handle) {
                goal_handle->abort(new_result);
            }
            break;
        }
      };

    send_goal_options.feedback_callback = 
      [this, goal_handle]( GoalHandleFollowJointTrajectory::SharedPtr, const std::shared_ptr<const FollowJointTrajectory::Feedback> feedback) {
        auto actual = feedback->actual;
        KDL::Frame pos_frame;
        dir_kin(
          actual.positions[0],
          actual.positions[1],
          actual.positions[2],
          actual.positions[3],
          actual.positions[4],
          actual.positions[5],
          pos_frame
        );

        auto rotation = pos_frame.M;
        double r_x = 0;
        double r_y = 0;
        double r_z = 0;
        double w = 0;
        rotation.GetQuaternion( r_x, r_y, r_z, w);

        auto vector = pos_frame.p;
        auto new_feedback = std::make_shared<GrabMove::Feedback>();
        new_feedback->x = vector.x();
        new_feedback->y = vector.y();
        new_feedback->z = vector.z();
        new_feedback->r_x = r_x;
        new_feedback->r_y = r_y;
        new_feedback->r_z = r_z;
        new_feedback->w = w;

        goal_handle->publish_feedback(new_feedback);
      };

    // Send the trajectory goal asynchronously.
    action_client_->async_send_goal(goal_msg, send_goal_options);
  };

  /**
   * @brief Callback function to process joint state messages.
   * @param msg Shared pointer to the received JointState message.
   *
   * This callback updates the current joint configuration (q_curr_) based on the received joint state message.
   * It also checks for NaN values and warns if not all joint values are present.
   */
  void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    // Check for NaN values.
    for (size_t i = 0; i < msg->position.size(); i++)
    {
      if (std::isnan(msg->position[i]))
      {
        RCLCPP_WARN(this->get_logger(), "Joint state message contains NaN values");
        return;
      }
    }

    // Update current joint configuration.
    for (size_t i = 0; i < joint_names_.size(); i++) {
      bool updated = false;
      
      for (size_t j = 0; j < msg->name.size(); j++) {
        if ( joint_names_[i] == msg->name[j] ) {
          q_curr_(i) = msg->position[j];
          updated = true;
          break;
        }
      }

      if (!updated) {
        RCLCPP_WARN(this->get_logger(), "Joint state message does not contain all joint values");
      }
    }
  }

  /**
   * @brief Computes the inverse kinematics.
   * @param x Desired x-position in the base frame.
   * @param y Desired y-position in the base frame.
   * @param z Desired z-position in the base frame.
   * @param r_x x-component of the desired quaternion.
   * @param r_y y-component of the desired quaternion.
   * @param r_z z-component of the desired quaternion.
   * @param w w-component of the desired quaternion.
   * @param q_out Output joint array with the computed joint angles.
   * @param q_curr Current joint configuration used as the initial guess.
   * @return int Returns 0 if successful; otherwise, returns a non-zero error code.
   */
  int inv_kin(double x, double y, double z, double r_x, double r_y, double r_z, double w,
              KDL::JntArray &q_out, KDL::JntArray &q_curr)
  {
    KDL::Vector des_v(x, y, z);
    KDL::Rotation des_r = KDL::Rotation::Quaternion(r_x, r_y, r_z, w);
    KDL::Frame des_p( des_r, des_v );

    q_out.resize(6);
    int result = ik_solver_->CartToJnt( q_curr, des_p, q_out );

    return result;
  }

  /**
   * @brief Computes the forward kinematics.
   * @param q0 Joint angle for joint 0.
   * @param q1 Joint angle for joint 1.
   * @param q2 Joint angle for joint 2.
   * @param q3 Joint angle for joint 3.
   * @param q4 Joint angle for joint 4.
   * @param q5 Joint angle for joint 5.
   * @param p_out Output KDL frame with the computed pose.
   * @return int Returns 0 if successful; otherwise, returns a non-zero error code.
   */
  int dir_kin(double q0, double q1, double q2, double q3, double q4, double q5, KDL::Frame& p_out) {
    KDL::JntArray q_in(6);
    q_in(0) = q0;
    q_in(1) = q1;
    q_in(2) = q2;
    q_in(3) = q3;
    q_in(4) = q4;
    q_in(5) = q5;

    p_out = KDL::Frame();
    return fk_solver_->JntToCart(q_in, p_out);
  }

};  // class GrabMoveActionServer

}  // namespace ur5_motion_cpp

// Register the component with the class_loader.
RCLCPP_COMPONENTS_REGISTER_NODE(ur5_motion_cpp::GrabMoveActionServer)

/** @} */
