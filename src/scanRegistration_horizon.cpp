// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <eigen3/Eigen/Core>

typedef pcl::PointXYZINormal PointType;

class ScanRegistrationNode : public rclcpp::Node
{
public:
  explicit ScanRegistrationNode()
    : Node("scan_registration")
  {
    // Declare parameters
    this->declare_parameter("n_scans", 6);
    N_SCANS_ = this->get_parameter("n_scans").as_int();

    // Create publishers with ROS2 QoS that matches ROS1 behavior
    rclcpp::QoS qos(10);  // Original ROS1 queue size
    qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
    qos.durability(rclcpp::DurabilityPolicy::Volatile);
    qos.history(rclcpp::HistoryPolicy::Keep_Last);

    pub_laser_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/velodyne_cloud_2", qos);
    pub_corner_points_sharp_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/laser_cloud_sharp", qos);
    pub_surf_points_flat_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/laser_cloud_flat", qos);
    pub_laser_cloud_temp_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/velodyne_cloud_registered", qos);

    // Create subscription with proper QoS settings
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/livox_pcl0", qos,
      std::bind(&ScanRegistrationNode::laserCloudHandler, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Scan registration node initialized with %d scans", N_SCANS_);
  }

private:
  int scan_id_{0};
  int N_SCANS_{6};
  int cloud_feature_flag_[32000]{0};

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_corner_points_sharp_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_surf_points_flat_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_temp_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;

  std::vector<std::shared_ptr<sensor_msgs::msg::PointCloud2>> msg_window_;
  cv::Mat mat_a1_{3, 3, CV_32F, cv::Scalar::all(0)};
  cv::Mat mat_d1_{1, 3, CV_32F, cv::Scalar::all(0)};
  cv::Mat mat_v1_{3, 3, CV_32F, cv::Scalar::all(0)};

  bool plane_judge(const std::vector<PointType>& point_list, const int plane_threshold)
  {
    if (point_list.empty()) {
      RCLCPP_WARN(this->get_logger(), "Empty point list provided to plane_judge");
      return false;
    }

    int num = point_list.size();
    float cx = 0;
    float cy = 0;
    float cz = 0;
    for (const auto& point : point_list) {
      cx += point.x;
      cy += point.y;
      cz += point.z;
    }
    cx /= num;
    cy /= num;
    cz /= num;

    // mean square error
    float a11 = 0;
    float a12 = 0;
    float a13 = 0;
    float a22 = 0;
    float a23 = 0;
    float a33 = 0;

    for (const auto& point : point_list) {
      float ax = point.x - cx;
      float ay = point.y - cy;
      float az = point.z - cz;

      a11 += ax * ax;
      a12 += ax * ay;
      a13 += ax * az;
      a22 += ay * ay;
      a23 += ay * az;
      a33 += az * az;
    }

    a11 /= num;
    a12 /= num;
    a13 /= num;
    a22 /= num;
    a23 /= num;
    a33 /= num;

    mat_a1_.at<float>(0, 0) = a11;
    mat_a1_.at<float>(0, 1) = a12;
    mat_a1_.at<float>(0, 2) = a13;
    mat_a1_.at<float>(1, 0) = a12;
    mat_a1_.at<float>(1, 1) = a22;
    mat_a1_.at<float>(1, 2) = a23;
    mat_a1_.at<float>(2, 0) = a13;
    mat_a1_.at<float>(2, 1) = a23;
    mat_a1_.at<float>(2, 2) = a33;

    cv::eigen(mat_a1_, mat_d1_, mat_v1_);
    return mat_d1_.at<float>(0, 0) > plane_threshold * mat_d1_.at<float>(0, 1);
  }

  void laserCloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg)
  {
    try {
      pcl::PointCloud<PointType> laserCloudIn;
      pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);

      int cloudSize = std::min(static_cast<int>(laserCloudIn.points.size()), 32000);
      
      RCLCPP_DEBUG(this->get_logger(), "Processing cloud with size: %d", cloudSize);

      std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS_);
      for (int i = 0; i < cloudSize; i++) {
        const auto& point_in = laserCloudIn.points[i];
        PointType point;
        point.x = point_in.x;
        point.y = point_in.y;
        point.z = point_in.z;
        point.intensity = point_in.intensity;
        point.curvature = point_in.curvature;
        
        int scanID = (N_SCANS_ == 6) ? static_cast<int>(point.intensity) : 0;
        if (scanID >= 0 && scanID < N_SCANS_) {
          laserCloudScans[scanID].push_back(point);
        }
      }

      auto laserCloud = std::make_shared<pcl::PointCloud<PointType>>();
      for (const auto& scan : laserCloudScans) {
        *laserCloud += scan;
      }

      cloudSize = laserCloud->size();
      std::fill_n(cloud_feature_flag_, cloudSize, 0);
      
      // Extract features
      pcl::PointCloud<PointType> cornerPointsSharp;
      pcl::PointCloud<PointType> surfPointsFlat;
      
      const double depth_threshold = 0.1;
      const int window_size = 5;

      for (int i = window_size; i < cloudSize - window_size; i++) {
        const auto& current_point = laserCloud->points[i];
        float depth = std::sqrt(current_point.x * current_point.x +
                              current_point.y * current_point.y +
                              current_point.z * current_point.z);

        // Calculate left neighborhood curvature
        Eigen::Vector3f left_diff = Eigen::Vector3f::Zero();
        for (int j = -4; j <= 0; j++) {
          if (j == -2) {
            left_diff -= 4.0f * Eigen::Vector3f(
              laserCloud->points[i + j].x,
              laserCloud->points[i + j].y,
              laserCloud->points[i + j].z);
          } else {
            left_diff += Eigen::Vector3f(
              laserCloud->points[i + j].x,
              laserCloud->points[i + j].y,
              laserCloud->points[i + j].z);
          }
        }
        float left_curvature = left_diff.squaredNorm();

        // Calculate right neighborhood curvature
        Eigen::Vector3f right_diff = Eigen::Vector3f::Zero();
        for (int j = 0; j <= 4; j++) {
          if (j == 2) {
            right_diff -= 4.0f * Eigen::Vector3f(
              laserCloud->points[i + j].x,
              laserCloud->points[i + j].y,
              laserCloud->points[i + j].z);
          } else {
            right_diff += Eigen::Vector3f(
              laserCloud->points[i + j].x,
              laserCloud->points[i + j].y,
              laserCloud->points[i + j].z);
          }
        }
        float right_curvature = right_diff.squaredNorm();

        // Feature extraction based on curvature
        if (left_curvature < 0.01) {
          std::vector<PointType> left_points;
          for (int j = -4; j < 0; j++) {
            left_points.push_back(laserCloud->points[i + j]);
          }
          
          if (left_curvature < 0.001 && plane_judge(left_points, 1000)) {
            cloud_feature_flag_[i - 2] = 1;  // surf point flag
            surfPointsFlat.push_back(laserCloud->points[i - 2]);
          }
        }

        if (right_curvature < 0.01) {
          std::vector<PointType> right_points;
          for (int j = 1; j < 5; j++) {
            right_points.push_back(laserCloud->points[i + j]);
          }
          
          if (right_curvature < 0.001 && plane_judge(right_points, 1000)) {
            cloud_feature_flag_[i + 2] = 1;  // surf point flag
            surfPointsFlat.push_back(laserCloud->points[i + 2]);
          }
          i += 3;  // Skip points to avoid duplicate processing
        }

        // Extract corner points based on high curvature
        if (left_curvature > 0.1 && right_curvature > 0.1) {
          cornerPointsSharp.push_back(current_point);
        }
      }

      // Publish results
      sensor_msgs::msg::PointCloud2 outMsg;
      
      pcl::toROSMsg(*laserCloud, outMsg);
      outMsg.header = laserCloudMsg->header;
      pub_laser_cloud_->publish(outMsg);

      pcl::toROSMsg(cornerPointsSharp, outMsg);
      outMsg.header = laserCloudMsg->header;
      pub_corner_points_sharp_->publish(outMsg);

      pcl::toROSMsg(surfPointsFlat, outMsg);
      outMsg.header = laserCloudMsg->header;
      pub_surf_points_flat_->publish(outMsg);

    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Error in laserCloudHandler: %s", e.what());
    }
  }

  void laserCloudHandler_temp(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg)
  {
    try {
      auto laserCloudIn = std::make_shared<pcl::PointCloud<PointType>>();

      // Maintain a window of messages
      if (msg_window_.size() < 2) {
        msg_window_.push_back(laserCloudMsg);
      } else {
        msg_window_.erase(msg_window_.begin());
        msg_window_.push_back(laserCloudMsg);
      }

      // Combine points from the message window
      for (const auto& msg : msg_window_) {
        pcl::PointCloud<PointType> temp;
        pcl::fromROSMsg(*msg, temp);
        *laserCloudIn += temp;
      }

      // Publish combined cloud
      sensor_msgs::msg::PointCloud2 laserCloudOutMsg;
      pcl::toROSMsg(*laserCloudIn, laserCloudOutMsg);
      laserCloudOutMsg.header = laserCloudMsg->header;
      laserCloudOutMsg.header.frame_id = "livox";
      pub_laser_cloud_temp_->publish(laserCloudOutMsg);

    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Error in laserCloudHandler_temp: %s", e.what());
    }
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ScanRegistrationNode>();
  RCLCPP_INFO(node->get_logger(), "Starting Scan Registration Node");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
