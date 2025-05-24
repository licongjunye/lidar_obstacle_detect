/*
 * @Author: xiaohu
 * @Date: 2022-04-02 00:26:55
 * @Last Modified by: xiaohu
 * @Last Modified time: 2022-04-02 01:12:59
 */

#include "lidar_obstacle_detection.h"

// 时间统计
int64_t euclidean_time = 0.;
int64_t total_time = 0.;
int counter = 0;

int64_t gtm()
{
	struct timeval tm;
	gettimeofday(&tm, 0);
	int64_t re = (((int64_t)tm.tv_sec) * 1000 * 1000 + tm.tv_usec);
	return re;
}

void publishCloud(
	const ros::Publisher *in_publisher, std_msgs::Header header,
	const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_to_publish_ptr)
{
	sensor_msgs::PointCloud2 cloud_msg;
	pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
	cloud_msg.header = header;
	in_publisher->publish(cloud_msg);
}

lidarObstacleDetection::lidarObstacleDetection(ros::NodeHandle nh, ros::NodeHandle pnh)
	: roi_clip_(nh, pnh), voxel_grid_filter_(nh, pnh), cluster_(nh, pnh), bounding_box_(pnh)
{
	ros::Subscriber sub = nh.subscribe("/velodyne_points", 1, &lidarObstacleDetection::ClusterCallback, this);

	_pub_clip_cloud = nh.advertise<sensor_msgs::PointCloud2>("/points_clip", 1);
	_pub_noground_cloud = nh.advertise<sensor_msgs::PointCloud2>("/nopoints_ground", 1);
	_pub_cluster_cloud = nh.advertise<sensor_msgs::PointCloud2>("/points_cluster", 1);
	_pub_clusters_message = nh.advertise<autoware_msgs::CloudClusterArray>("/detection/lidar_detector/cloud_clusters", 1);
	_pub_detected_objects = nh.advertise<autoware_msgs::DetectedObjectArray>("/detection/lidar_detector/objects", 1);
	_pub_cluster_visualize_markers = nh.advertise<visualization_msgs::MarkerArray>("/visualize/cluster_markers", 1);
	_pub_cluster_visualize_markers_true = nh.advertise<visualization_msgs::MarkerArray>("/visualize/cluster_markers_true", 1);
	ros::spin();
}

/**
 * @brief 聚类回调函数
 *
 * 当接收到点云数据时，调用此函数进行聚类处理。
 *
 * @param in_sensor_cloud 输入的点云数据
 */
void lidarObstacleDetection::ClusterCallback(
	const sensor_msgs::PointCloud2ConstPtr &in_sensor_cloud)
{
	std_msgs::Header header = in_sensor_cloud->header;
	pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr clip_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr noground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr ground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

	pcl::fromROSMsg(*in_sensor_cloud, *in_cloud_ptr);

	// 提取ROI
	int64_t tm0 = gtm();
	roi_clip_.GetROI(in_cloud_ptr, clip_cloud_ptr);
	// std::cout << "clip_cloud_ptr " << clip_cloud_ptr->points.size() << std::endl;
	int64_t tm1 = gtm();
	// ROS_INFO("ROI_Clip cost time:%ld ms", (tm1 - tm0) / 1000);

	// 下采样
	voxel_grid_filter_.downsample(clip_cloud_ptr, downsampled_cloud_ptr);

	// 地面分割
	patch_work_.estimate_ground(downsampled_cloud_ptr, ground_cloud_ptr, noground_cloud_ptr);
	int64_t tm2 = gtm();
	// ROS_INFO("remove ground cost time:%ld ms", (tm2 - tm1) / 1000);

	// 聚类
	pcl::PointCloud<pcl::PointXYZI>::Ptr outCloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
	std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pointsVector;

	cluster_.segmentByDistance(noground_cloud_ptr, outCloudPtr, pointsVector);

	// 真值聚类
	pcl::PointCloud<pcl::PointXYZI>::Ptr outCloudPtr_true(new pcl::PointCloud<pcl::PointXYZI>);
	std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pointsVector_true;

	cluster_.segmentByDistance_true(noground_cloud_ptr, outCloudPtr_true, pointsVector_true);

	
	int64_t tm3 = gtm();
	if(cluster_.cluster_method_ == EuclideanCluster::EUCLIDEAN_CLUSTER)
	{
		ROS_INFO("euclidean cluster cost time:%ld ms", (tm3 - tm2) / 1000);
	}
	else if(cluster_.cluster_method_ == EuclideanCluster::DBSCAN_CLUSTER)
	{
		ROS_INFO("dbscan cluster cost time:%ld ms", (tm3 - tm2) / 1000);
	}
	else if(cluster_.cluster_method_ == EuclideanCluster::MEANSHIFT_CLUSTER)
	{
		ROS_INFO("meanshift cluster cost time:%ld ms", (tm3 - tm2) / 1000);
	}
	else if(cluster_.cluster_method_ == EuclideanCluster::KMEANS_CLUSTER)
	{
		ROS_INFO("kmeans cluster cost time:%ld ms", (tm3 - tm2) / 1000);
	}
	else if(cluster_.cluster_method_ == EuclideanCluster::FAST_EUCLIDEAN_CLUSTER)
	{
		ROS_INFO("fastEuclidean clustercost time:%ld ms", (tm3 - tm2) / 1000);
	}else if(cluster_.cluster_method_ == EuclideanCluster::BRUTE_FORCE_CLUSTER)
	{
		ROS_INFO("brute force euclidean clustercost time:%ld ms", (tm3 - tm2) / 1000);	
	}
	// ROS_INFO("total cost time:%ld ms", (tm3 - tm0) / 1000);

	// 获取bounding_box信息
	autoware_msgs::CloudClusterArray inOutClusters;
	bounding_box_.getBoundingBox(header, pointsVector, inOutClusters);
	autoware_msgs::DetectedObjectArray detected_objects;
	publishDetectedObjects(inOutClusters, detected_objects);

	// 可视化
	visualization_msgs::MarkerArray visualize_markers;
	vdo_.visualizeDetectedObjs(detected_objects, visualize_markers);
	_pub_cluster_visualize_markers.publish(visualize_markers);

	// 真值可视化
	autoware_msgs::CloudClusterArray inOutClusters_true;
	bounding_box_.getBoundingBox(header, pointsVector_true, inOutClusters_true);
	autoware_msgs::DetectedObjectArray detected_objects_true;
	publishDetectedObjects(inOutClusters_true, detected_objects_true);
	
	// 计算重合度
	double average_iou = calculateAverageIoU(detected_objects, detected_objects_true);
	ROS_INFO("Average IoU between test and truth: %.1f%%", average_iou * 100.0);
	
	// 设置绿色
	std_msgs::ColorRGBA green_color;
	green_color.r = 0.0;
	green_color.g = 1.0;
	green_color.b = 0.0;
	green_color.a = 0.8;
	
	// 为每个检测到的对象设置绿色
	for(auto& obj : detected_objects_true.objects) {
		obj.color = green_color;
	}
	
	visualization_msgs::MarkerArray visualize_markers_true;
	vdo_.visualizeDetectedObjs(detected_objects_true, visualize_markers_true);
	_pub_cluster_visualize_markers_true.publish(visualize_markers_true);

	// 发布topic

	publishCloud(&_pub_clip_cloud, in_sensor_cloud->header, clip_cloud_ptr);
	publishCloud(&_pub_noground_cloud, in_sensor_cloud->header, noground_cloud_ptr);
	publishCloud(&_pub_cluster_cloud, in_sensor_cloud->header, outCloudPtr);

	euclidean_time += tm3 - tm2;
	total_time += tm3 - tm0;
	counter++;
	if (counter % 100 == 0)
	{
		ROS_INFO(
			"[INFO] euclidean cluster average time per hundred times:%ld ms",
			euclidean_time / 100000);
		ROS_INFO("[INFO] total average time per hundred times:%ld ms",
				 total_time / 100000);
		euclidean_time = 0.;
		total_time = 0.;
	}
}

/**
 * @brief 发布检测到的障碍物对象
 *
 * 将给定的 CloudClusterArray 数组中的障碍物对象转换为 DetectedObjectArray 数组，并发布出去。
 *
 * @param in_clusters 输入的 CloudClusterArray 数组
 * @param detected_objects 输出的 DetectedObjectArray 数组
 */
void lidarObstacleDetection::publishDetectedObjects(
	const autoware_msgs::CloudClusterArray &in_clusters,
	autoware_msgs::DetectedObjectArray &detected_objects)
{
	detected_objects.header = in_clusters.header;
	for (size_t i = 0; i < in_clusters.clusters.size(); i++)
	{
		autoware_msgs::DetectedObject detected_object;
		detected_object.header = in_clusters.header;
		detected_object.label = "unknown";
		detected_object.score = 1.;
		detected_object.space_frame = in_clusters.header.frame_id;
		detected_object.pose = in_clusters.clusters[i].bounding_box.pose;

		detected_object.dimensions = in_clusters.clusters[i].dimensions;
		detected_object.pointcloud = in_clusters.clusters[i].cloud;
		detected_object.convex_hull = in_clusters.clusters[i].convex_hull;
		detected_object.valid = true;

		detected_objects.objects.push_back(detected_object);
	}

	_pub_detected_objects.publish(detected_objects);
}

/**
 * @brief 计算两个检测框的IoU（Intersection over Union）
 * 
 * @param box1 第一个检测框
 * @param box2 第二个检测框
 * @return double IoU值
 */
double lidarObstacleDetection::calculateIoU(const autoware_msgs::DetectedObject& box1, const autoware_msgs::DetectedObject& box2) {
    // 计算两个框的中心点距离
    double dx = box1.pose.position.x - box2.pose.position.x;
    double dy = box1.pose.position.y - box2.pose.position.y;
    double dz = box1.pose.position.z - box2.pose.position.z;
    double distance = sqrt(dx*dx + dy*dy + dz*dz);
    
    // 如果中心点距离太远，直接返回0
    if (distance > 5.0) {  // 5米阈值，可以根据实际情况调整
        return 0.0;
    }
    
    // 计算两个框的体积
    double volume1 = box1.dimensions.x * box1.dimensions.y * box1.dimensions.z;
    double volume2 = box2.dimensions.x * box2.dimensions.y * box2.dimensions.z;
    
    // 计算相交体积（简化计算，使用中心点距离和尺寸）
    double intersection_x = std::max(0.0, std::min(box1.dimensions.x/2 + box2.dimensions.x/2 - std::abs(dx), 
                                                 std::min(box1.dimensions.x, box2.dimensions.x)));
    double intersection_y = std::max(0.0, std::min(box1.dimensions.y/2 + box2.dimensions.y/2 - std::abs(dy), 
                                                 std::min(box1.dimensions.y, box2.dimensions.y)));
    double intersection_z = std::max(0.0, std::min(box1.dimensions.z/2 + box2.dimensions.z/2 - std::abs(dz), 
                                                 std::min(box1.dimensions.z, box2.dimensions.z)));
    
    double intersection_volume = intersection_x * intersection_y * intersection_z;
    double union_volume = volume1 + volume2 - intersection_volume;
    
    return intersection_volume / union_volume;
}

/**
 * @brief 计算两组检测框的平均IoU
 * 
 * @param test_objects 测试检测框
 * @param true_objects 真实检测框
 * @return double 平均IoU值
 */
double lidarObstacleDetection::calculateAverageIoU(const autoware_msgs::DetectedObjectArray& test_objects, 
                          const autoware_msgs::DetectedObjectArray& true_objects) {
    if (test_objects.objects.empty() || true_objects.objects.empty()) {
        return 0.0;
    }
    
    double total_iou = 0.0;
    int matched_pairs = 0;
    
    // 为每个测试框找到最佳匹配的真实框
    std::vector<bool> true_matched(true_objects.objects.size(), false);
    
    for (const auto& test_obj : test_objects.objects) {
        double best_iou = 0.0;
        int best_match_idx = -1;
        
        for (size_t i = 0; i < true_objects.objects.size(); ++i) {
            if (!true_matched[i]) {
                double iou = calculateIoU(test_obj, true_objects.objects[i]);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_match_idx = i;
                }
            }
        }
        
        if (best_match_idx != -1 && best_iou > 0.5) {  // IoU阈值设为0.5
            total_iou += best_iou;
            true_matched[best_match_idx] = true;
            matched_pairs++;
        }
    }
    
    return matched_pairs > 0 ? total_iou / matched_pairs : 0.0;
}
