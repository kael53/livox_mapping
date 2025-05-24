#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>  // Include for pcl::toROSMsg

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_pcl_out1;
rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_livox_msg1;
uint64_t TO_MERGE_CNT = 1;
constexpr bool b_dbg_line = false;
std::vector<livox_ros_driver2::msg::CustomMsg::SharedPtr> livox_data;

void LivoxMsgCbk1(const livox_ros_driver2::msg::CustomMsg::SharedPtr livox_msg_in) {
  try {
    livox_data.push_back(livox_msg_in);
    if (livox_data.size() < TO_MERGE_CNT) return;

    pcl::PointCloud<PointType> pcl_in;

    for (size_t j = 0; j < livox_data.size(); j++) {
      auto& livox_msg = livox_data[j];
      if (livox_msg->points.empty()) {
        RCLCPP_WARN(rclcpp::get_logger("livox_repub"), "Empty point cloud received");
        continue;
      }

      auto time_end = livox_msg->points.back().offset_time;
      for (unsigned int i = 0; i < livox_msg->point_num; ++i) {
        PointType pt;
        pt.x = livox_msg->points[i].x;
        pt.y = livox_msg->points[i].y;
        pt.z = livox_msg->points[i].z;
        float s = livox_msg->points[i].offset_time / (float)time_end;

        pt.intensity = livox_msg->points[i].line + livox_msg->points[i].reflectivity / 10000.0;  // The integer part is line number and the decimal part is timestamp
        pt.curvature = s * 0.1;
        pcl_in.push_back(pt);
      }
    }

    if (!pcl_in.empty()) {
      unsigned long timebase_ns = livox_data[0]->timebase;
      rclcpp::Time timestamp(timebase_ns);

      sensor_msgs::msg::PointCloud2 pcl_ros_msg;
      pcl::toROSMsg(pcl_in, pcl_ros_msg);
      pcl_ros_msg.header.stamp = timestamp;
      pcl_ros_msg.header.frame_id = "/livox";  // Removed leading slash for ROS2 convention
      pub_pcl_out1->publish(pcl_ros_msg);
    }
    livox_data.clear();
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("livox_repub"), "Error in LivoxMsgCbk1: %s", e.what());
  }
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("livox_repub");

  pub_pcl_out1 = node->create_publisher<sensor_msgs::msg::PointCloud2>("/livox_pcl0", 100);
  sub_livox_msg1 = node->create_subscription<livox_ros_driver2::msg::CustomMsg>("/livox/lidar", 100, LivoxMsgCbk1);

  RCLCPP_INFO(rclcpp::get_logger("livox_repub"), "Starting livox_repub node");

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
