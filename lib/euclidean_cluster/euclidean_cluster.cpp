/*
 * @Author: xiaohu
 * @Date: 2022-04-02 00:26:55
 * @Last Modified by: xiaohu
 * @Last Modified time: 2022-04-02 01:12:59
 */

#include "euclidean_cluster.h"

EuclideanCluster::EuclideanCluster(ros::NodeHandle nh, ros::NodeHandle pnh)
{
    pnh.param("cluster_tolerance", cluster_tolerance_, 0.35);
    pnh.param("min_cluster_size", min_cluster_size_, 5);
    pnh.param("max_cluster_size", max_cluster_size_, 20000);
    pnh.param("use_multiple_thres", use_multiple_thres_, false);
    pnh.param("cluster_method", cluster_method_, 1);
    pnh.param("eps", dbscan_eps_, 2.0);
    pnh.param("min_pts", dbscan_min_pts_, 50);
    
    // Kmeans++参数初始化
    pnh.param("kmeans_max_iteration", kmeans_max_iteration_, 100);
    pnh.param("kmeans_cluster_num", kmeans_cluster_num_, 50);

    // MeanShift参数初始化
    pnh.param("meanshift_bandwidth", meanshift_bandwidth_, 7.0);
    pnh.param("meanshift_max_iteration", meanshift_max_iteration_, 100);
    pnh.param("meanshift_convergence_threshold", meanshift_convergence_threshold_, 0.01);

    // 快速欧式聚类参数初始化
    pnh.param("fast_cluster_tolerance", fast_cluster_tolerance_, 0.02);
    pnh.param("fast_cluster_max_size", fast_cluster_max_size_, 50);
    pnh.param("fast_cluster_min_size", fast_cluster_min_size_, 100);
    pnh.param("fast_cluster_max_cluster", fast_cluster_max_cluster_, 25000);

    clustering_distances_ = {0.5, 1.1, 1.6, 2.1, 2.6};
    clustering_ranges_ = {15, 30, 45, 60};
}

/******  欧式聚类分割 *******/
void EuclideanCluster::cluster_vector(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, std::vector<pcl::PointIndices> &indices)
{
    //设置查找方式－kdtree
    pcl::search::Search<pcl::PointXYZI>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZI>>(new pcl::search::KdTree<pcl::PointXYZI>);

    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(cluster_tolerance_); 
    ec.setMinClusterSize(min_cluster_size_);   
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(in);
    ec.extract(indices);
}

/******  DBSCAN聚类分割 *******/
void EuclideanCluster::dbscan_vector(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, std::vector<pcl::PointIndices> &indices)
{
    // 预分配内存
    indices.clear();
    indices.reserve(in->size() / dbscan_min_pts_);  // 预估聚类数量
    
    // 记录点是否已处理
    std::vector<bool> cloud_processed(in->size(), false);
    
    // 创建KD树（只创建一次）
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
    tree->setInputCloud(in);
    
    // 预分配搜索结果的存储空间
    std::vector<pcl::Indices> k_indices_array(omp_get_max_threads());
    std::vector<std::vector<float>> k_distances_array(omp_get_max_threads());
    for (int i = 0; i < omp_get_max_threads(); ++i) {
        k_indices_array[i].reserve(in->size());
        k_distances_array[i].reserve(in->size());
    }
    
    // 使用OpenMP并行处理
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        pcl::Indices& k_indices = k_indices_array[thread_id];
        std::vector<float>& k_distances = k_distances_array[thread_id];
        
        // 每个线程维护自己的聚类结果
        std::vector<pcl::PointIndices> local_indices;
        local_indices.reserve(in->size() / (dbscan_min_pts_ * omp_get_max_threads()));
        
        #pragma omp for schedule(dynamic, 1000)
        for (size_t i = 0; i < in->size(); ++i)
        {
            if (cloud_processed[i])
            {
                continue;
            }
            
            // 清空上一次的搜索结果
            k_indices.clear();
            k_distances.clear();
            
            // 搜索当前点的邻域
            if (tree->radiusSearch(in->points[i], dbscan_eps_, k_indices, k_distances) < dbscan_min_pts_)
            {
                continue;
            }
            
            // 创建新的聚类
            pcl::PointIndices cluster;
            cluster.indices.reserve(k_indices.size());
            
            // 将核心点加入聚类
            cluster.indices.push_back(i);
            cloud_processed[i] = true;
            
            // 使用队列进行广度优先搜索
            std::queue<size_t> search_queue;
            for (size_t j = 0; j < k_indices.size(); ++j)
            {
                search_queue.push(k_indices[j]);
            }
            
            // 扩展聚类
            while (!search_queue.empty())
            {
                size_t current_idx = search_queue.front();
                search_queue.pop();
                
                if (cloud_processed[current_idx])
                {
                    continue;
                }
                
                // 将未处理的点加入聚类
                cluster.indices.push_back(current_idx);
                cloud_processed[current_idx] = true;
                
                // 搜索当前点的邻域
                pcl::Indices neighbor_indices;
                std::vector<float> neighbor_distances;
                if (tree->radiusSearch(in->points[current_idx], dbscan_eps_, neighbor_indices, neighbor_distances) >= dbscan_min_pts_)
                {
                    // 将新的未处理点加入搜索队列
                    for (size_t k = 0; k < neighbor_indices.size(); ++k)
                    {
                        if (!cloud_processed[neighbor_indices[k]])
                        {
                            search_queue.push(neighbor_indices[k]);
                        }
                    }
                }
            }
            
            // 如果聚类大小满足要求，则加入本地结果
            if (cluster.indices.size() >= static_cast<size_t>(min_cluster_size_) && 
                cluster.indices.size() <= static_cast<size_t>(max_cluster_size_))
            {
                local_indices.push_back(cluster);
            }
        }
        
        // 合并本地结果到全局结果
        #pragma omp critical
        {
            indices.insert(indices.end(), local_indices.begin(), local_indices.end());
        }
    }
    
    // 按聚类大小排序（可选）
    std::sort(indices.begin(), indices.end(), 
              [](const pcl::PointIndices& a, const pcl::PointIndices& b) {
                  return a.indices.size() > b.indices.size();
              });
}

void EuclideanCluster::kmeans_vector(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, std::vector<pcl::PointIndices> &indices)
{
    // 预分配内存
    indices.clear();
    indices.reserve(kmeans_cluster_num_);
    
    if (in->empty())
    {
        return;
    }
    
    const size_t num_points = in->size();
    const size_t num_threads = omp_get_max_threads();
    
    // 预分配搜索结果的存储空间
    std::vector<std::vector<double>> dists_array(num_threads);
    std::vector<std::vector<Eigen::Vector4f>> centers_array(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        dists_array[i].reserve(kmeans_cluster_num_);
        centers_array[i].reserve(kmeans_cluster_num_);
    }
    
    // 将点云数据转换为Eigen向量数组，提高访问效率
    std::vector<Eigen::Vector4f> points_vec(num_points);
    #pragma omp parallel for schedule(dynamic, 1000)
    for (size_t i = 0; i < num_points; ++i) {
        points_vec[i] = in->points[i].getVector4fMap();
    }
    
    // -------------------------最远点采样选取聚类中心点----------------------------
    std::vector<int> selected_indices;
    selected_indices.reserve(kmeans_cluster_num_);
    std::vector<float> distances(num_points, std::numeric_limits<float>::infinity());
    size_t farthest_index = 0;
    
    // 并行计算初始距离
    #pragma omp parallel for schedule(dynamic, 1000)
    for (size_t i = 0; i < num_points; ++i) {
        distances[i] = points_vec[i].head<3>().squaredNorm();
    }
    
    // 选择初始聚类中心
    for (size_t i = 0; i < kmeans_cluster_num_; i++) 
    {
        selected_indices.push_back(farthest_index);
        const Eigen::Vector4f& selected = points_vec[farthest_index];
        double max_dist = 0;
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            double local_max_dist = 0;
            size_t local_farthest_index = 0;
            
            #pragma omp for schedule(dynamic, 1000)
            for (size_t j = 0; j < num_points; j++)
            {
                float dist = (points_vec[j].head<3>() - selected.head<3>()).squaredNorm();
                distances[j] = std::min(distances[j], dist);
                if (distances[j] > local_max_dist) 
                {
                    local_max_dist = distances[j];
                    local_farthest_index = j;
                }
            }
            
            #pragma omp critical
            {
                if (local_max_dist > max_dist)
                {
                    max_dist = local_max_dist;
                    farthest_index = local_farthest_index;
                }
            }
        }
    }
    
    // 获取聚类中心点
    std::vector<Eigen::Vector4f> centers(kmeans_cluster_num_);
    for (size_t i = 0; i < kmeans_cluster_num_; ++i) {
        centers[i] = points_vec[selected_indices[i]];
    }
    
    // -----------------------------------进行KMeans聚类--------------------------------
    int iterations = 0;
    double sum_diff = 0.2;
    
    // 预分配内存
    std::vector<std::vector<pcl::PointIndices>> local_indices_array(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        local_indices_array[i].resize(kmeans_cluster_num_);
    }
    
    // 如果大于迭代次数或者两次重心之差小于0.02就停止
    while (!(iterations >= kmeans_max_iteration_ || sum_diff <= 0.02))
    {
        sum_diff = 0;
        
        // 并行计算点到中心的距离并分配簇
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            std::vector<double>& dists = dists_array[thread_id];
            std::vector<pcl::PointIndices>& local_indices = local_indices_array[thread_id];
            
            // 清空本地索引
            for (auto& cluster : local_indices) {
                cluster.indices.clear();
            }
            
            #pragma omp for schedule(dynamic, 1000)
            for (size_t i = 0; i < num_points; ++i)
            {
                dists.clear();
                dists.reserve(kmeans_cluster_num_);
                
                // 计算到所有中心的距离
                const Eigen::Vector4f& point = points_vec[i];
                for (size_t j = 0; j < kmeans_cluster_num_; ++j)
                {
                    float dist = (point.head<3>() - centers[j].head<3>()).squaredNorm();
                    dists.emplace_back(dist);
                }
                
                // 找到最近的簇
                int min_idx = std::distance(dists.begin(), std::min_element(dists.begin(), dists.end()));
                local_indices[min_idx].indices.push_back(i);
            }
        }
        
        // 合并本地结果
        indices.clear();
        indices.resize(kmeans_cluster_num_);
        for (const auto& local_indices : local_indices_array) {
            for (size_t i = 0; i < kmeans_cluster_num_; ++i) {
                indices[i].indices.insert(indices[i].indices.end(), 
                                       local_indices[i].indices.begin(), 
                                       local_indices[i].indices.end());
            }
        }
        
        // 重新计算簇中心点
        std::vector<Eigen::Vector4f> new_centers(kmeans_cluster_num_);
        std::vector<int> cluster_sizes(kmeans_cluster_num_, 0);
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            std::vector<Eigen::Vector4f>& local_centers = centers_array[thread_id];
            local_centers.resize(kmeans_cluster_num_);
            std::fill(local_centers.begin(), local_centers.end(), Eigen::Vector4f::Zero());
            
            #pragma omp for schedule(dynamic)
            for (size_t k = 0; k < kmeans_cluster_num_; ++k)
            {
                if (!indices[k].indices.empty())
                {
                    for (int idx : indices[k].indices) {
                        local_centers[k] += points_vec[idx];
                    }
                }
            }
            
            #pragma omp critical
            {
                for (size_t k = 0; k < kmeans_cluster_num_; ++k) {
                    if (!indices[k].indices.empty()) {
                        new_centers[k] += local_centers[k];
                        cluster_sizes[k] = indices[k].indices.size();
                    } else {
                        new_centers[k] = centers[k];
                    }
                }
            }
        }
        
        // 计算新的中心点
        #pragma omp parallel for schedule(dynamic)
        for (size_t k = 0; k < kmeans_cluster_num_; ++k) {
            if (cluster_sizes[k] > 0) {
                new_centers[k] /= cluster_sizes[k];
            }
        }
        
        // 计算聚类中心点的变化量
        #pragma omp parallel for reduction(+:sum_diff)
        for (size_t s = 0; s < kmeans_cluster_num_; ++s)
        {
            float dist = (new_centers[s].head<3>() - centers[s].head<3>()).squaredNorm();
            sum_diff += dist;
        }
        
        centers = std::move(new_centers);
        ++iterations;
    }
    
    // 过滤掉太小的簇
    indices.erase(
        std::remove_if(indices.begin(), indices.end(),
            [this](const pcl::PointIndices& cluster) {
                return cluster.indices.size() < static_cast<size_t>(min_cluster_size_) ||
                       cluster.indices.size() > static_cast<size_t>(max_cluster_size_);
            }),
        indices.end()
    );
    
    // 按簇大小排序
    std::sort(indices.begin(), indices.end(),
        [](const pcl::PointIndices& a, const pcl::PointIndices& b) {
            return a.indices.size() > b.indices.size();
        });
}

void EuclideanCluster::meanShift_vector(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, std::vector<pcl::PointIndices> &indices)
{
    // 预分配内存
    indices.clear();
    indices.reserve(in->size() / min_cluster_size_);
    
    if (in->empty())
    {
        return;
    }
    
    const size_t num_points = in->size();
    const size_t num_threads = omp_get_max_threads();
    const size_t chunk_size = 1000;  // 动态调度的块大小
    
    // 创建KD树
    pcl::KdTreeFLANN<pcl::PointXYZI> tree;
    tree.setInputCloud(in);
    
    // 预分配搜索结果的存储空间
    std::vector<std::vector<int>> nn_indices_array(num_threads);
    std::vector<std::vector<float>> nn_dists_array(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        nn_indices_array[i].reserve(num_points);
        nn_dists_array[i].reserve(num_points);
    }
    
    // 初始化标签
    std::vector<int> labels(num_points, 0);
    int segLab = 1;
    
    // 使用OpenMP并行处理
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        std::vector<int>& nn_indices = nn_indices_array[thread_id];
        std::vector<float>& nn_dists = nn_dists_array[thread_id];
        
        #pragma omp for schedule(dynamic, chunk_size)
        for (size_t i = 0; i < num_points; ++i)
        {
            if (labels[i] != 0)
            {
                continue;
            }
            
            // 搜索邻域点
            if (tree.radiusSearch(in->points[i], meanshift_bandwidth_, nn_indices, nn_dists) <= 0)
            {
                continue;
            }
            
            // 获取当前点的坐标
            const Eigen::Vector3f& point = in->points[i].getVector3fMap();
            Eigen::Vector3f centroid = point;
            
            // MeanShift迭代
            bool converged = false;
            int iteration = 0;
            
            while (!converged && iteration < meanshift_max_iteration_)
            {
                // 计算加权平均位置
                Eigen::Vector3f new_centroid = Eigen::Vector3f::Zero();
                float total_weight = 0.0f;
                
                for (size_t j = 0; j < nn_indices.size(); ++j)
                {
                    const int idx_j = nn_indices[j];
                    const Eigen::Vector3f& point_j = in->points[idx_j].getVector3fMap();
                    const float dist = nn_dists[j];
                    const float weight = kernel(dist / meanshift_bandwidth_);
                    new_centroid += weight * point_j;
                    total_weight += weight;
                }
                
                if (total_weight > 0)
                {
                    new_centroid /= total_weight;
                }
                
                // 检查收敛
                if ((new_centroid - centroid).norm() < meanshift_convergence_threshold_)
                {
                    converged = true;
                }
                else
                {
                    centroid = new_centroid;
                }
                
                ++iteration;
            }
            
            // 标记聚类
            #pragma omp critical
            {
                for (size_t j = 0; j < nn_indices.size(); ++j)
                {
                    if (labels[nn_indices[j]] == 0)
                    {
                        labels[nn_indices[j]] = segLab;
                    }
                }
                ++segLab;
            }
        }
    }
    
    // 根据标签组织聚类结果
    std::unordered_map<int, int> label_map;
    int index = 0;
    
    for (size_t i = 0; i < num_points; ++i)
    {
        int label = labels[i];
        if (label == 0) continue;
        
        if (label_map.find(label) == label_map.end())
        {
            label_map[label] = index++;
            indices.push_back(pcl::PointIndices());
        }
        indices[label_map[label]].indices.push_back(i);
    }
    
    // 过滤掉太小的簇
    indices.erase(
        std::remove_if(indices.begin(), indices.end(),
            [this](const pcl::PointIndices& cluster) {
                return cluster.indices.size() < static_cast<size_t>(min_cluster_size_) ||
                       cluster.indices.size() > static_cast<size_t>(max_cluster_size_);
            }),
        indices.end()
    );
    
    // 按簇大小排序
    std::sort(indices.begin(), indices.end(),
        [](const pcl::PointIndices& a, const pcl::PointIndices& b) {
            return a.indices.size() > b.indices.size();
        });
}

void EuclideanCluster::fastEuclideanCluster(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, std::vector<pcl::PointIndices> &indices)
{
    // 预分配内存
    indices.clear();
    
    if (in->empty())
    {
        ROS_WARN("Input cloud is empty!");
        return;
    }
    
    // 设置默认参数
    if (fast_cluster_min_size_ <= 0) fast_cluster_min_size_ = 100;
    if (fast_cluster_max_size_ <= 0) fast_cluster_max_size_ = 50;
    if (fast_cluster_max_cluster_ <= 0) fast_cluster_max_cluster_ = 25000;
    if (fast_cluster_tolerance_ <= 0) fast_cluster_tolerance_ = 0.02;
    
    // 初始化标签
    std::vector<int> labels(in->size(), 0);  // 初始化所有点标签为0
    int segLab = 1;  // 聚类标签从1开始
    
    // 创建KD树
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(in);
    
    // 预分配搜索结果的存储空间
    pcl::Indices nn_indices;
    std::vector<float> nn_dists;
    nn_indices.reserve(fast_cluster_max_size_);
    nn_dists.reserve(fast_cluster_max_size_);
    
    // 对每个点进行聚类
    for (size_t i = 0; i < in->size(); ++i)
    {
        // 如果点未被标记
        if (labels[i] == 0)
        {
            // 搜索邻域点
            kdtree.radiusSearch(in->points[i], fast_cluster_tolerance_, nn_indices, nn_dists, fast_cluster_max_size_);
            
            // 获取邻域点的标签
            std::vector<int> nLabs;
            for (size_t j = 0; j < nn_indices.size(); ++j)
            {
                if (labels[nn_indices[j]] != 0)
                {
                    nLabs.push_back(labels[nn_indices[j]]);
                }
            }
            
            // 确定最小标签
            int minSegLab = 0;
            if (!nLabs.empty())
            {
                minSegLab = *std::min_element(nLabs.begin(), nLabs.end());
            }
            else
            {
                minSegLab = segLab;
            }
            
            // 合并标签
            for (size_t j = 0; j < nLabs.size(); ++j)
            {
                if (nLabs[j] > minSegLab)
                {
                    for (size_t k = 0; k < labels.size(); ++k)
                    {
                        if (labels[k] == nLabs[j])
                        {
                            labels[k] = minSegLab;
                        }
                    }
                }
            }
            
            // 标记所有邻域点
            for (size_t nnIdx = 0; nnIdx < nn_indices.size(); ++nnIdx)
            {
                labels[nn_indices[nnIdx]] = minSegLab;
            }
            
            ++segLab;
        }
    }
    
    // 根据标签组织聚类结果
    std::unordered_map<int, int> segID;
    std::vector<std::vector<int>> clusterIndices;
    int index = 1;
    
    for (size_t i = 0; i < in->size(); ++i)
    {
        int label = labels[i];
        if (label == 0) continue;  // 跳过未标记的点
        
        if (segID.find(label) == segID.end())
        {
            segID[label] = index;
            clusterIndices.push_back(std::vector<int>());
            ++index;
        }
        clusterIndices[segID[label] - 1].push_back(i);
    }
    
    // 筛选符合点数阈值的类别
    for (size_t i = 0; i < clusterIndices.size(); ++i)
    {
        if (clusterIndices[i].size() >= static_cast<size_t>(fast_cluster_min_size_) && 
            clusterIndices[i].size() <= static_cast<size_t>(fast_cluster_max_cluster_))
        {
            pcl::PointIndices cluster;
            cluster.indices = clusterIndices[i];
            indices.push_back(cluster);
        }
    }
    
    // 按簇大小排序
    std::sort(indices.begin(), indices.end(),
        [](const pcl::PointIndices& a, const pcl::PointIndices& b) {
            return a.indices.size() > b.indices.size();
        });
}

/**
 * @brief 暴力搜索欧式聚类
 * @param in 输入点云
 * @param indices 聚类结果
 */
void EuclideanCluster::bruteForceCluster(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, std::vector<pcl::PointIndices> &indices)
{
    // 预分配内存
    indices.clear();
    
    if (in->empty()) return;
    
    // 初始化标签
    std::vector<int> labels(in->size(), -1);
    int current_label = 0;
    
    // 将点云数据转换为Eigen向量数组，提高访问效率
    std::vector<Eigen::Vector4f> points_vec(in->size());
    #pragma omp parallel for schedule(dynamic, 1000)
    for (size_t i = 0; i < in->size(); ++i) {
        points_vec[i] = in->points[i].getVector4fMap();
    }
    
    // 对每个点进行聚类
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        std::vector<pcl::PointIndices> local_indices;
        local_indices.reserve(in->size() / min_cluster_size_);
        
        #pragma omp for schedule(dynamic, 1000)
        for (size_t i = 0; i < in->size(); ++i)
        {
            if (labels[i] != -1) continue;
            
            // 创建新的聚类
            pcl::PointIndices cluster;
            cluster.indices.push_back(i);
            labels[i] = current_label;
            
            // 使用队列进行广度优先搜索
            std::queue<size_t> search_queue;
            search_queue.push(i);
            
            while (!search_queue.empty())
            {
                size_t current_idx = search_queue.front();
                search_queue.pop();
                
                // 暴力搜索邻域点
                for (size_t j = 0; j < in->size(); ++j)
                {
                    if (current_idx == j || labels[j] != -1) continue;
                    
                    // 计算平方距离
                    float dist_squared = (points_vec[current_idx].head<3>() - points_vec[j].head<3>()).squaredNorm();
                    
                    // 如果距离小于阈值，加入聚类
                    if (dist_squared < cluster_tolerance_ * cluster_tolerance_ * 1.5)
                    {
                        cluster.indices.push_back(j);
                        labels[j] = current_label;
                        search_queue.push(j);
                    }
                }
            }
            
            // 如果聚类大小满足要求，添加到结果中
            if (cluster.indices.size() >= min_cluster_size_ && 
                cluster.indices.size() <= max_cluster_size_)
            {
                local_indices.push_back(cluster);
            }
        }
        
        // 合并本地结果到全局结果
        #pragma omp critical
        {
            indices.insert(indices.end(), local_indices.begin(), local_indices.end());
        }
    }
}

/**
 * @brief 网格划分欧式聚类
 * @param in 输入点云
 * @param indices 聚类结果
 */
void EuclideanCluster::gridBasedCluster(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, std::vector<pcl::PointIndices> &indices)
{
    // 预分配内存
    indices.clear();
    
    if (in->empty()) return;
    
    // 计算点云边界
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float min_z = std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();
    float max_z = -std::numeric_limits<float>::max();
    
    for (const auto& point : in->points)
    {
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
        min_z = std::min(min_z, point.z);
        max_x = std::max(max_x, point.x);
        max_y = std::max(max_y, point.y);
        max_z = std::max(max_z, point.z);
    }
    
    // 创建网格
    float grid_size = cluster_tolerance_;
    int nx = static_cast<int>((max_x - min_x) / grid_size) + 1;
    int ny = static_cast<int>((max_y - min_y) / grid_size) + 1;
    int nz = static_cast<int>((max_z - min_z) / grid_size) + 1;
    
    // 使用std::vector存储网格
    std::vector<std::vector<int>> grid(nx * ny * nz);
    
    // 将点分配到网格
    for (size_t i = 0; i < in->size(); ++i)
    {
        int gx = static_cast<int>((in->points[i].x - min_x) / grid_size);
        int gy = static_cast<int>((in->points[i].y - min_y) / grid_size);
        int gz = static_cast<int>((in->points[i].z - min_z) / grid_size);
        
        int key = gx + gy * nx + gz * nx * ny;
        grid[key].push_back(i);
    }
    
    // 初始化标签
    std::vector<int> labels(in->size(), -1);
    int current_label = 0;
    
    // 对每个网格进行聚类
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        std::vector<pcl::PointIndices> local_indices;
        local_indices.reserve(in->size() / min_cluster_size_);
        
        #pragma omp for schedule(dynamic, 1000)
        for (size_t i = 0; i < grid.size(); ++i)
        {
            const auto& points = grid[i];
            if (points.empty()) continue;
            
            // 创建新的聚类
            pcl::PointIndices cluster;
            
            // 使用队列进行广度优先搜索
            std::queue<int> search_queue;
            for (int idx : points)
            {
                if (labels[idx] != -1) continue;
                search_queue.push(idx);
                labels[idx] = current_label;
                cluster.indices.push_back(idx);
            }
            
            while (!search_queue.empty())
            {
                int current_idx = search_queue.front();
                search_queue.pop();
                
                // 检查当前网格和相邻网格中的点
                for (int dx = -1; dx <= 1; ++dx)
                {
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        for (int dz = -1; dz <= 1; ++dz)
                        {
                            int nx = i + dx + dy * nx + dz * nx * ny;
                            if (nx < 0 || nx >= grid.size()) continue;
                            
                            for (int idx : grid[nx])
                            {
                                if (labels[idx] != -1) continue;
                                
                                // 计算欧氏距离
                                float dx = in->points[current_idx].x - in->points[idx].x;
                                float dy = in->points[current_idx].y - in->points[idx].y;
                                float dz = in->points[current_idx].z - in->points[idx].z;
                                float dist = sqrt(dx*dx + dy*dy + dz*dz);
                                
                                if (dist < cluster_tolerance_ * 1.5)
                                {
                                    cluster.indices.push_back(idx);
                                    labels[idx] = current_label;
                                    search_queue.push(idx);
                                }
                            }
                        }
                    }
                }
            }
            
            // 如果聚类大小满足要求，添加到结果中
            if (cluster.indices.size() >= min_cluster_size_ && 
                cluster.indices.size() <= max_cluster_size_)
            {
                local_indices.push_back(cluster);
            }
        }
        
        // 合并本地结果到全局结果
        #pragma omp critical
        {
            indices.insert(indices.end(), local_indices.begin(), local_indices.end());
        }
    }
}

/**
 * @brief 根据距离进行欧几里得聚类分割
 *
 * 根据给定的点云数据，根据距离进行欧几里得聚类分割，将分割后的点云数据存储在指定的输出点云指针和点云向量中。
 *
 * @param in 输入的点云数据指针
 * @param out_cloud_ptr 输出的点云数据指针
 * @param points_vector 存储分割后点云数据的向量
 */
void EuclideanCluster::segmentByDistance(const pcl::PointCloud<pcl::PointXYZI>::Ptr in,
                                         pcl::PointCloud<pcl::PointXYZI>::Ptr &out_cloud_ptr, std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &points_vector)
{
    std::vector<pcl::PointIndices> cluster_indices;

    if (cluster_method_ == EUCLIDEAN_CLUSTER)
    {
        // 欧式聚类
        if (!use_multiple_thres_)
        {
            cluster_vector(in, cluster_indices);
        }
        else
        {
            std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloud_segments_array(5);
            for (unsigned int i = 0; i < cloud_segments_array.size(); i++)
            {
                pcl::PointCloud<pcl::PointXYZI>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZI>);
                cloud_segments_array[i] = tmp_cloud;
            }

            for (unsigned int i = 0; i < in->points.size(); i++)
            {
                pcl::PointXYZI current_point;
                current_point.x = in->points[i].x;
                current_point.y = in->points[i].y;
                current_point.z = in->points[i].z;
                current_point.intensity = in->points[i].intensity;

                float origin_distance = sqrt(pow(current_point.x, 2) + pow(current_point.y, 2));

                if (origin_distance < clustering_ranges_[0])
                {
                    cloud_segments_array[0]->points.push_back(current_point);
                }
                else if (origin_distance < clustering_ranges_[1])
                {
                    cloud_segments_array[1]->points.push_back(current_point);
                }
                else if (origin_distance < clustering_ranges_[2])
                {
                    cloud_segments_array[2]->points.push_back(current_point);
                }
                else if (origin_distance < clustering_ranges_[3])
                {
                    cloud_segments_array[3]->points.push_back(current_point);
                }
                else
                {
                    cloud_segments_array[4]->points.push_back(current_point);
                }
            }

            std::vector<std::thread> thread_vec(cloud_segments_array.size());
            for (unsigned int i = 0; i < cloud_segments_array.size(); i++)
            {
                // 这种获取多线程返回值写法，运行速度慢，大家有兴趣自行更改，我懒改了，这是粗版demo
                std::promise<std::vector<pcl::PointIndices>> promiseObj;
                std::shared_future<std::vector<pcl::PointIndices>> futureObj = promiseObj.get_future();
                thread_vec[i] = std::thread(&EuclideanCluster::clusterIndicesMultiThread, this, cloud_segments_array[i], std::ref(clustering_distances_[i]), std::ref(promiseObj));
                cluster_indices = futureObj.get();
                for (int j = 0; j < cluster_indices.size(); j++)
                {
                    pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
                    pcl::copyPointCloud(*cloud_segments_array[i], cluster_indices[j], *temp_cloud_ptr);
                    *out_cloud_ptr += *temp_cloud_ptr;
                    points_vector.push_back(temp_cloud_ptr);
                }
            }

            for (int i = 0; i < thread_vec.size(); i++)
            {
                thread_vec[i].join();
            }
        }
    }
    else if (cluster_method_ == DBSCAN_CLUSTER)
    {
        // DBSCAN聚类 
        dbscan_vector(in, cluster_indices);
    }
    else if (cluster_method_ == KMEANS_CLUSTER)
    {
        // Kmeans++聚类
        kmeans_vector(in, cluster_indices);
    }
    else if (cluster_method_ == MEANSHIFT_CLUSTER)
    {
        // MeanShift聚类
        meanShift_vector(in, cluster_indices);
    }
    else if (cluster_method_ == FAST_EUCLIDEAN_CLUSTER)
    {
        // 快速欧式聚类
        fastEuclideanCluster(in, cluster_indices);
    }
    else if (cluster_method_ == BRUTE_FORCE_CLUSTER)
    {
        // 暴力搜索欧式聚类
        bruteForceCluster(in, cluster_indices);
    }

    if(cluster_indices.size() > 0)
    {
        for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
        {
            pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::copyPointCloud(*in, it->indices, *temp_cloud_ptr);
            *out_cloud_ptr += *temp_cloud_ptr;
            points_vector.push_back(temp_cloud_ptr);
        }
    }
    else
    {
        ROS_WARN("No clusters found!");
    }
}


void EuclideanCluster::segmentByDistance_true(const pcl::PointCloud<pcl::PointXYZI>::Ptr in, pcl::PointCloud<pcl::PointXYZI>::Ptr &out_cloud_ptr,
                         std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &points_vector)
{
    std::vector<pcl::PointIndices> cluster_indices;
    fastEuclideanCluster(in, cluster_indices);

     if(cluster_indices.size() > 0)
    {
        for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
        {
            pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::copyPointCloud(*in, it->indices, *temp_cloud_ptr);
            *out_cloud_ptr += *temp_cloud_ptr;
            points_vector.push_back(temp_cloud_ptr);
        }
    }
    else
    {
        ROS_WARN("No clusters found!");
    }
}


/**
 * @brief 使用多线程进行欧几里得聚类
 *
 * 根据给定的点云数据，使用多线程进行欧几里得聚类，并将聚类结果通过 std::promise 返回。
 *
 * @param in_cloud_ptr 输入的点云数据指针
 * @param in_max_cluster_distance 最大的聚类距离
 * @param promiseObj 用于返回聚类结果的 std::promise 对象
 */
void EuclideanCluster::clusterIndicesMultiThread(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr, double in_max_cluster_distance,
                                                 std::promise<std::vector<pcl::PointIndices>> &promiseObj)
{
    // make it flat
    // for (size_t i = 0; i < cloud_2d->points.size(); i++)
    // {
    //     cloud_2d->points[i].z = 0;
    // }
    pcl::search::Search<pcl::PointXYZI>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZI>>(new pcl::search::KdTree<pcl::PointXYZI>);
    std::vector<pcl::PointIndices> indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(cluster_tolerance_); 
    ec.setMinClusterSize(min_cluster_size_);   
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(in_cloud_ptr);
    ec.extract(indices);

    promiseObj.set_value(indices);
}


