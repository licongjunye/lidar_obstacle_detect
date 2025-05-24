/*
 * @Author: xiaohu
 * @Date: 2022-04-02 00:26:55
 * @Last Modified by: xiaohu
 * @Last Modified time: 2022-04-02 01:12:59
 */

#ifndef LIDAR_DETECTION_TRACK_H_
#define LIDAR_DETECTION_TRACK_H_

#include <iostream>
#include <string>
#include <vector>
#include <mutex>

#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include "autoware_msgs/Centroids.h"
#include "autoware_msgs/CloudCluster.h"
#include "autoware_msgs/CloudClusterArray.h"
#include "autoware_msgs/DetectedObject.h"
#include "autoware_msgs/DetectedObjectArray.h"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../lib/ground_detector/patchwork/patchwork.h"
#include "../lib/visualization/visualize_detected_objects.h"
#include "../lib/pre_process/roi_clip/roi_clip.h"
#include "../lib/euclidean_cluster/euclidean_cluster.h"
#include "../lib/pre_process/voxel_grid_filter/voxel_grid_filter.h"
#include "../lib/bounding_box/bounding_box.h"
#include "../lib/visualization/visualize_detected_objects.h"

class lidarObstacleDetection
{
public:
    lidarObstacleDetection(ros::NodeHandle nh, ros::NodeHandle pnh);
    ~lidarObstacleDetection(){};

    RoiClip roi_clip_;
    VoxelGridFilter voxel_grid_filter_;
    PatchWork patch_work_;
    EuclideanCluster cluster_;

    BoundingBox bounding_box_;
    VisualizeDetectedObjects vdo_;

    ros::Publisher _pub_clip_cloud;
    ros::Publisher _pub_ground_cloud;
    ros::Publisher _pub_noground_cloud;
    ros::Publisher _pub_cluster_cloud;
    ros::Publisher _pub_clusters_message;

    ros::Publisher _pub_detected_objects;
    ros::Publisher _pub_detected_3Dobjects;

    ros::Publisher _pub_cluster_visualize_markers;
    ros::Publisher _pub_3Dobjects_visualize_markers;

    ros::Publisher _pub_cluster_visualize_markers_true;

    VisualizeDetectedObjects vdto;

private:
    void ClusterCallback(const sensor_msgs::PointCloud2ConstPtr &in_sensor_cloud);
    void publishDetectedObjects(const autoware_msgs::CloudClusterArray &in_clusters, autoware_msgs::DetectedObjectArray &detected_objects);
    
    /**
     * @brief 计算两个检测框的IoU（Intersection over Union）
     * 
     * @param box1 第一个检测框
     * @param box2 第二个检测框
     * @return double IoU值
     */
    double calculateIoU(const autoware_msgs::DetectedObject& box1, const autoware_msgs::DetectedObject& box2);
    
    /**
     * @brief 计算两组检测框的平均IoU
     * 
     * @param test_objects 测试检测框
     * @param true_objects 真实检测框
     * @return double 平均IoU值
     */
    double calculateAverageIoU(const autoware_msgs::DetectedObjectArray& test_objects, 
                              const autoware_msgs::DetectedObjectArray& true_objects);
};

#endif
