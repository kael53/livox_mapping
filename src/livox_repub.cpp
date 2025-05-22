#include <rclcpp/rclcpp.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "livox_ros_driver2/msg/custom_msg.hpp"

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

class LivoxRepub : public rclcpp::Node
{
public:
  LivoxRepub() : Node("livox_repub")
  {
    pub_pcl_out0 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/livox_pcl0", 10);
    pub_pcl_out1 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/livox_pcl1", 10);
    sub_livox_msg1 = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
      "/livox/lidar", 10, std::bind(&LivoxRepub::LivoxMsgCbk1, this, std::placeholders::_1));
  }

private:
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_pcl_out0;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_pcl_out1;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_livox_msg1;
  uint64_t TO_MERGE_CNT = 1;
  constexpr static bool b_dbg_line = false;
  std::vector<std::shared_ptr<livox_ros_driver2::msg::CustomMsg const>> livox_data;

  void LivoxMsgCbk1(const std::shared_ptr<livox_ros_driver2::msg::CustomMsg const>& livox_msg_in) {
  livox_data.push_back(livox_msg_in);
  if (livox_data.size() < TO_MERGE_CNT) return;

  pcl::PointCloud<PointType> pcl_in;

  for (size_t j = 0; j < livox_data.size(); j++) {
    auto& livox_msg = livox_data[j];
    auto time_end = livox_msg->points.back().offset_time;
    for (unsigned int i = 0; i < livox_msg->point_num; ++i) {
      PointType pt;
      pt.x = livox_msg->points[i].x;
      pt.y = livox_msg->points[i].y;
      pt.z = livox_msg->points[i].z;
      float s = livox_msg->points[i].offset_time / (float)time_end;

      pt.intensity = livox_msg->points[i].line +livox_msg->points[i].reflectivity /10000.0 ; // The integer part is line number and the decimal part is timestamp
      pt.curvature = s*0.1;
      pcl_in.push_back(pt);
    }
  }

    unsigned long timebase_ns = livox_data[0]->timebase;
    rclcpp::Time timestamp(timebase_ns, RCL_SYSTEM_TIME);

    sensor_msgs::msg::PointCloud2 pcl_ros_msg;
    pcl::toROSMsg(pcl_in, pcl_ros_msg);
    pcl_ros_msg.header.stamp = timestamp;
    pcl_ros_msg.header.frame_id = "/livox";
    pub_pcl_out1->publish(pcl_ros_msg);
    livox_data.clear();
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  RCLCPP_INFO(rclcpp::get_logger("livox_repub"), "Starting livox_repub node");
  auto node = std::make_shared<LivoxRepub>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
