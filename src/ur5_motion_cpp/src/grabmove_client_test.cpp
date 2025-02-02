#include <functional>
#include <future>
#include <memory>
#include <string>
#include <sstream>

#include "ur5_motion_interfaces/action/grab_move.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace ur5_motion_cpp
{
class GrabMoveActionClient : public rclcpp::Node
{
public:
  using GrabMove = ur5_motion_interfaces::action::GrabMove;
  using GoalHandleGrabMove = rclcpp_action::ClientGoalHandle<GrabMove>;

  explicit GrabMoveActionClient(const rclcpp::NodeOptions & options)
  : Node("grabmove_action__client", options)
  {
    this->client_ptr_ = rclcpp_action::create_client<GrabMove>(
      this,
      "grabmove");

    auto timer_callback_lambda = [this](){ return this->send_goal(); };
    this->timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      timer_callback_lambda);
  }

  void send_goal()
  {
    using namespace std::placeholders;

    this->timer_->cancel();

    if (!this->client_ptr_->wait_for_action_server()) {
      RCLCPP_ERROR(this->get_logger(), "Action server not available after waiting");
      rclcpp::shutdown();
    }

    auto goal_msg = GrabMove::Goal();
    goal_msg.x = 0.2;
    goal_msg.y = 0.2;
    goal_msg.z = 0.6;
    goal_msg.r_x = 1;
    goal_msg.r_y = 0;
    goal_msg.r_z = 0;
    goal_msg.w = 1;

    RCLCPP_INFO(this->get_logger(), "Sending goal");

    auto send_goal_options = rclcpp_action::Client<GrabMove>::SendGoalOptions();
    send_goal_options.goal_response_callback = [this](const GoalHandleGrabMove::SharedPtr & goal_handle)
    {
      if (!goal_handle) {
        RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
      } else {
        RCLCPP_INFO(this->get_logger(), "Goal accepted by server, waiting for result");
      }
    };

    send_goal_options.feedback_callback = [this](
      GoalHandleGrabMove::SharedPtr,
      const std::shared_ptr<const GrabMove::Feedback> feedback)
    {
      RCLCPP_INFO(
        this->get_logger(),
        "Received goal request with x:%f y:%f z:%f r_x:%f r_y:%f r_z:%f, w:%f",
        feedback->x, feedback->y, feedback->z, feedback->r_x, feedback->r_y, feedback->r_z, feedback->w
      );
    };

    send_goal_options.result_callback = [this](const GoalHandleGrabMove::WrappedResult & result)
    {
      switch (result.code) {
        case rclcpp_action::ResultCode::SUCCEEDED:
          break;
        case rclcpp_action::ResultCode::ABORTED:
          RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
          return;
        case rclcpp_action::ResultCode::CANCELED:
          RCLCPP_ERROR(this->get_logger(), "Goal was canceled");
          return;
        default:
          RCLCPP_ERROR(this->get_logger(), "Unknown result code");
          return;
      }
      RCLCPP_INFO(this->get_logger(),"Finished");
      rclcpp::shutdown();
    };
    this->client_ptr_->async_send_goal(goal_msg, send_goal_options);
  }

private:
  rclcpp_action::Client<GrabMove>::SharedPtr client_ptr_;
  rclcpp::TimerBase::SharedPtr timer_;
};  // class GrabMoveActionClient

}  // namespace ur5_motion_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(ur5_motion_cpp::GrabMoveActionClient)