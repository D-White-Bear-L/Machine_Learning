import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建输出文件夹
output_folder = "c:\\programme\\data_science\\artificial intelligence\\homwork_ex\\hw\\hw03\\birch_output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 加载数据集
try:
    # 尝试从本地加载数据集
    data = pd.read_csv("c:\\programme\\data_science\\artificial intelligence\\homwork_ex\\hw\\hw03\\Mall_Customers.csv")
    print("成功加载本地Mall Customer Segmentation数据集")
except:
    # 如果本地没有，从网络下载
    print("本地未找到数据集，正在尝试下载...")
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv"
        urllib.request.urlretrieve(url, "c:\\programme\\data_science\\artificial intelligence\\homwork_ex\\hw\\hw03\\Mall_Customers.csv")
        data = pd.read_csv("c:\\programme\\data_science\\artificial intelligence\\homwork_ex\\hw\\hw03\\Mall_Customers.csv")
        print("成功下载并加载Mall Customer Segmentation数据集")
    except:
        # 如果下载失败，创建一个示例数据集
        print("下载失败，创建示例数据集...")
        np.random.seed(42)
        n_samples = 200
        
        # 创建示例数据
        age = np.random.randint(18, 70, n_samples)
        annual_income = np.random.randint(15, 100, n_samples)
        spending_score = np.random.randint(1, 100, n_samples)
        gender = np.random.choice(['Male', 'Female'], n_samples)
        customer_id = [f'C{i+1}' for i in range(n_samples)]
        
        # 创建DataFrame
        data = pd.DataFrame({
            'CustomerID': customer_id,
            'Genre': gender,
            'Age': age,
            'Annual Income (k$)': annual_income,
            'Spending Score (1-100)': spending_score
        })
        print("已创建示例Mall Customer Segmentation数据集")

# 查看数据集基本信息
print("\n数据集基本信息:")
print(f"数据集形状: {data.shape}")
print("\n数据集前5行:")
print(data.head())

# 数据预处理
# 提取需要用于聚类的特征 - 只使用数值特征
X = data[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']]

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# BIRCH算法参数选择
# 可视化不同threshold参数的效果
threshold_range = [0.1, 0.3, 0.5, 0.7]
plt.figure(figsize=(15, 10))

for i, threshold in enumerate(threshold_range):
    # 使用BIRCH算法
    birch = Birch(n_clusters=None, threshold=threshold, branching_factor=50)
    birch_labels = birch.fit_predict(X_scaled)
    
    # 计算聚类数量
    n_clusters = len(set(birch_labels))
    
    # 绘制聚类结果
    plt.subplot(2, 2, i+1)
    plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], 
                c=birch_labels, cmap='viridis', s=50, alpha=0.8)
    plt.title(f'BIRCH (threshold={threshold})\n聚类数: {n_clusters}')
    plt.xlabel('年收入 (k$)')
    plt.ylabel('消费评分 (1-100)')
    plt.colorbar(label='聚类')
    plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'birch_threshold_comparison.png'))
plt.show()

# 可视化不同branching_factor参数的效果
branching_factor_range = [20, 50, 100, 200]
plt.figure(figsize=(15, 10))

for i, branching_factor in enumerate(branching_factor_range):
    # 使用BIRCH算法
    birch = Birch(n_clusters=None, threshold=0.5, branching_factor=branching_factor)
    birch_labels = birch.fit_predict(X_scaled)
    
    # 计算聚类数量
    n_clusters = len(set(birch_labels))
    
    # 绘制聚类结果
    plt.subplot(2, 2, i+1)
    plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], 
                c=birch_labels, cmap='viridis', s=50, alpha=0.8)
    plt.title(f'BIRCH (branching_factor={branching_factor})\n聚类数: {n_clusters}')
    plt.xlabel('年收入 (k$)')
    plt.ylabel('消费评分 (1-100)')
    plt.colorbar(label='聚类')
    plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'birch_branching_factor_comparison.png'))
plt.show()

# 确定最佳聚类数量
max_clusters = 10
silhouette_scores = []

for n_clusters in range(2, max_clusters + 1):
    # 使用BIRCH算法
    birch = Birch(n_clusters=n_clusters, threshold=0.5, branching_factor=50)
    birch_labels = birch.fit_predict(X_scaled)
    
    # 计算轮廓系数
    score = silhouette_score(X_scaled, birch_labels)
    silhouette_scores.append(score)
    print(f"聚类数 {n_clusters}, 轮廓系数: {score:.4f}")

# 绘制轮廓系数图
plt.figure(figsize=(10, 6))
plt.plot(range(2, max_clusters + 1), silhouette_scores, 'bo-')
plt.xlabel('聚类数量')
plt.ylabel('轮廓系数')
plt.title('BIRCH: 不同聚类数量的轮廓系数')
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'birch_silhouette_scores.png'))
plt.show()

# 选择最佳聚类数量
best_n_clusters = np.argmax(silhouette_scores) + 2  # +2 因为从2开始
print(f"\n最佳聚类数量: {best_n_clusters}")

# 使用最佳参数运行BIRCH
birch = Birch(n_clusters=best_n_clusters, threshold=0.5, branching_factor=50)
birch_labels = birch.fit_predict(X_scaled)

# 将聚类结果添加到原始数据中
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = birch_labels

# 可视化BIRCH聚类结果
plt.figure(figsize=(12, 10))

# 绘制散点图
scatter = plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], 
                     c=birch_labels, cmap='viridis', alpha=0.8, s=100)

plt.colorbar(scatter, label='聚类')
plt.title(f"BIRCH聚类结果 (聚类数: {best_n_clusters})")
plt.xlabel('年收入 (k$)')
plt.ylabel('消费评分 (1-100)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'birch_clusters.png'))
plt.show()

# 3D可视化
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
scatter = ax.scatter(X['Annual Income (k$)'], 
                    X['Spending Score (1-100)'], 
                    X['Age'],
                    c=birch_labels, cmap='viridis', 
                    s=80, alpha=0.8)

ax.set_xlabel('年收入 (k$)')
ax.set_ylabel('消费评分 (1-100)')
ax.set_zlabel('年龄')
ax.set_title(f"BIRCH聚类结果 - 3D视图 (聚类数: {best_n_clusters})")
plt.colorbar(scatter, label='聚类', pad=0.1)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'birch_3d_visualization.png'))
plt.show()

# 分析每个聚类的特征
def analyze_clusters(data, clusters):
    """分析每个聚类的特征分布"""
    # 添加聚类标签
    data_copy = data.copy()
    data_copy['Cluster'] = clusters
    
    # 计算每个聚类的特征均值 - 只使用数值型列
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cluster_means = data_copy.groupby('Cluster')[numeric_cols].mean()
    
    # 绘制热图
    plt.figure(figsize=(12, 8))
    sns.heatmap(cluster_means, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('各聚类的特征均值')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cluster_features.png'))
    plt.show()
    
    # 绘制箱线图
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):
        if i < 4:  # 限制最多显示4个特征
            plt.subplot(2, 2, i+1)
            sns.boxplot(x='Cluster', y=feature, data=data_copy)
            plt.title(f'聚类的{feature}分布')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cluster_boxplots.png'))
    plt.show()
    
    # 分析性别分布
    if 'Genre' in data.columns:
        gender_distribution = pd.crosstab(data_copy['Cluster'], 
                                         data_copy['Genre'], 
                                         normalize='index')
        
        plt.figure(figsize=(10, 6))
        gender_distribution.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title('各聚类的性别分布')
        plt.xlabel('聚类')
        plt.ylabel('比例')
        plt.legend(title='性别')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'cluster_gender.png'))
        plt.show()
    
    return cluster_means

# 分析聚类特征
cluster_means = analyze_clusters(data, birch_labels)
print("\n各聚类的特征均值:")
print(cluster_means)

# 计算每个聚类的样本数量
cluster_counts = pd.Series(birch_labels).value_counts().sort_index()
print("\n各聚类的样本数量:")
print(cluster_counts)

# 绘制聚类大小分布
plt.figure(figsize=(10, 6))
cluster_counts.plot(kind='bar')
plt.title('各聚类的样本数量')
plt.xlabel('聚类')
plt.ylabel('样本数量')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'cluster_sizes.png'))
plt.show()

# 保存聚类结果
data_with_clusters.to_csv(os.path.join(output_folder, 'birch_clustering_results.csv'), index=False)
print(f"\n聚类结果已保存至: {os.path.join(output_folder, 'birch_clustering_results.csv')}")

# BIRCH算法优势总结
print("\nBIRCH聚类算法优势:")
print("1. 高效处理大型数据集，时间复杂度为O(n)")
print("2. 只需要对数据进行一次扫描")
print("3. 对异常值不敏感")
print("4. 可以增量学习，适合流数据")
print("5. 内存占用小，适合有限内存环境")