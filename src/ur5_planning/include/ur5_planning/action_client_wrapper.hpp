#ifndef ACTION_CLIENT_WRAPPER_HPP_
#define ACTION_CLIENT_WRAPPER_HPP_

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

template<typename ActionT>
class ActionClientWrapper {
public:
    using GoalHandle = typename rclcpp_action::ClientGoalHandle<ActionT>;
    using Result = typename rclcpp_action::ClientGoalHandle<ActionT>::WrappedResult;

    explicit ActionClientWrapper(
        rclcpp::Node::SharedPtr node,
        const std::string& action_name,
        std::chrono::seconds timeout = std::chrono::seconds(60))
        : node_(node), timeout_(timeout)
    {
        client_ = rclcpp_action::create_client<ActionT>(node, action_name);
    }

    bool sendGoal(const typename ActionT::Goal& goal, Result& result) {
        // Wait for action server
        if (!client_->wait_for_action_server(timeout_)) {
            RCLCPP_ERROR(node_->get_logger(), "Action server not available");
            return false;
        }

        // Send goal
        auto goal_handle_future = client_->async_send_goal(goal);
        
        // Wait for goal acceptance
        if (rclcpp::spin_until_future_complete(node_, goal_handle_future) != rclcpp::FutureReturnCode::SUCCESS) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to send goal");
            return false;
        }
        
        auto goal_handle = goal_handle_future.get();
        if (!goal_handle) {
            RCLCPP_ERROR(node_->get_logger(), "Goal was rejected");
            return false;
        }

        // Wait for result
        auto result_future = client_->async_get_result(goal_handle);
        if (rclcpp::spin_until_future_complete(node_, result_future) != rclcpp::FutureReturnCode::SUCCESS) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to get result");
            return false;
        }

        result = result_future.get();
        return result.code == rclcpp_action::ResultCode::SUCCEEDED;
    }

private:
    rclcpp::Node::SharedPtr node_;
    typename rclcpp_action::Client<ActionT>::SharedPtr client_;
    std::chrono::seconds timeout_;
};

#endif // ACTION_CLIENT_WRAPPER_HPP_