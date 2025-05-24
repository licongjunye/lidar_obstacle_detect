/*
 * @Author: xiaohu
 * @Date: 2022-04-02 00:26:55
 * @Last Modified by: xiaohu
 * @Last Modified time: 2022-04-02 01:12:59
 */

#ifndef EUCLIDEAN_CLUSTER_H_
#define EUCLIDEAN_CLUSTER_H_

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <future>
#include <unordered_map>

#include <ros/ros.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_ros/point_cloud.h>


class EuclideanCluster
{
public:
  EuclideanCluster(ros::NodeHandle nh, ros::NodeHandle pnh);
  ~EuclideanCluster(){};
  //欧式聚类
  void cluster_vector(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, std::vector<pcl::PointIndices> &indices);
  //DBSCAN聚类
  void dbscan_vector(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, std::vector<pcl::PointIndices> &indices);
  //Kmeans++聚类
  void kmeans_vector(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, std::vector<pcl::PointIndices> &indices);
  //MeanShift聚类
  void meanShift_vector(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, std::vector<pcl::PointIndices> &indices);
  //快速欧式聚类
  void fastEuclideanCluster(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, std::vector<pcl::PointIndices> &indices);
  
  void segmentByDistance(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, pcl::PointCloud<pcl::PointXYZI>::Ptr &out_cloud_ptr,
                         std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &points_vector);

  void segmentByDistance_true(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, pcl::PointCloud<pcl::PointXYZI>::Ptr &out_cloud_ptr,
                         std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &points_vector);

  void clusterIndicesMultiThread(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr, double in_max_cluster_distance,
                                 std::promise<std::vector<pcl::PointIndices>> &promiseObj);

  // 暴力搜索欧式聚类
  void bruteForceCluster(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, std::vector<pcl::PointIndices> &indices);
  // 网格划分欧式聚类
  void gridBasedCluster(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, std::vector<pcl::PointIndices> &indices);

  int cluster_method_;
  enum ClusterMethod
  {
      EUCLIDEAN_CLUSTER = 1,
      DBSCAN_CLUSTER = 2,
      KMEANS_CLUSTER = 3,
      MEANSHIFT_CLUSTER = 4,
      FAST_EUCLIDEAN_CLUSTER = 5,
      BRUTE_FORCE_CLUSTER = 6 // 暴力搜索欧式聚类
  };
  

private:
  double cluster_tolerance_;
  int min_cluster_size_;
  int max_cluster_size_;
  std::vector<double> clustering_distances_;
  std::vector<double> clustering_ranges_;
  bool use_multiple_thres_;
  double dbscan_eps_;
  int dbscan_min_pts_;
  
  // Kmeans++聚类参数
  int kmeans_max_iteration_;  // 最大迭代次数
  int kmeans_cluster_num_;    // 聚类个数

  // MeanShift参数
  double meanshift_bandwidth_;  // 修改为double类型
  int meanshift_max_iteration_;
  double meanshift_convergence_threshold_;  // 修改为double类型

  // 快速欧式聚类参数
  double fast_cluster_tolerance_;  // 搜索半径
  int fast_cluster_max_size_;      // 邻域搜索最大点数
  int fast_cluster_min_size_;      // 聚类最小点数
  int fast_cluster_max_cluster_;   // 聚类最大点数

  // 高斯核函数
  inline float kernel(float x)
  {
      return 2 * sqrt(x) * exp(-0.5 * x);
  }

  std::mutex mutex_; //先定义互斥锁
};

#endif
