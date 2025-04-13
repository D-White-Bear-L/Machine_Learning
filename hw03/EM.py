import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建输出文件夹
output_folder = "c:\\programme\\data_science\\artificial intelligence\\homwork_ex\\hw\\hw03\\em_output"
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

# 确定最佳组件数量（聚类数）
max_components = 10
bic_scores = []
aic_scores = []
silhouette_scores_list = []

for n_components in range(2, max_components + 1):
    # 使用EM算法（高斯混合模型）
    gmm = GaussianMixture(n_components=n_components, 
                          covariance_type='full', 
                          random_state=42,
                          n_init=10,
                          max_iter=100)
    gmm.fit(X_scaled)
    
    # 计算BIC和AIC
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))
    
    # 预测聚类标签
    labels = gmm.predict(X_scaled)
    
    # 计算轮廓系数
    silhouette_avg = silhouette_score(X_scaled, labels)
    silhouette_scores_list.append(silhouette_avg)
    
    print(f"组件数: {n_components}, BIC: {gmm.bic(X_scaled):.2f}, AIC: {gmm.aic(X_scaled):.2f}, 轮廓系数: {silhouette_avg:.4f}")

# 绘制BIC和AIC曲线
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(range(2, max_components + 1), bic_scores, 'bo-')
plt.xlabel('组件数量')
plt.ylabel('BIC得分')
plt.title('BIC得分 vs 组件数量')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(range(2, max_components + 1), aic_scores, 'ro-')
plt.xlabel('组件数量')
plt.ylabel('AIC得分')
plt.title('AIC得分 vs 组件数量')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(range(2, max_components + 1), silhouette_scores_list, 'go-')
plt.xlabel('组件数量')
plt.ylabel('轮廓系数')
plt.title('轮廓系数 vs 组件数量')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'em_component_selection.png'))
plt.show()

# 选择最佳组件数量
# BIC和AIC越小越好，轮廓系数越大越好
best_bic_components = np.argmin(bic_scores) + 2  # +2 因为从2开始
best_aic_components = np.argmin(aic_scores) + 2
best_silhouette_components = np.argmax(silhouette_scores_list) + 2

print(f"\n基于BIC的最佳组件数量: {best_bic_components}")
print(f"基于AIC的最佳组件数量: {best_aic_components}")
print(f"基于轮廓系数的最佳组件数量: {best_silhouette_components}")

# 使用BIC选择的最佳组件数量
best_n_components = best_bic_components

# 使用最佳组件数量运行EM算法
gmm = GaussianMixture(n_components=best_n_components, 
                      covariance_type='full', 
                      random_state=42,
                      n_init=10,
                      max_iter=100)
gmm.fit(X_scaled)

# 预测聚类标签
labels = gmm.predict(X_scaled)

# 计算每个样本属于每个聚类的概率
probabilities = gmm.predict_proba(X_scaled)

# 将聚类结果添加到原始数据中
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = labels

# 添加每个样本属于其聚类的概率
data_with_clusters['Cluster_Probability'] = np.max(probabilities, axis=1)

# 可视化EM聚类结果
plt.figure(figsize=(12, 10))

# 绘制散点图
scatter = plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], 
                     c=labels, cmap='viridis', alpha=0.8, s=100)

# 绘制每个高斯分布的中心
means = gmm.means_
plt.scatter(means[:, 0] * scaler.scale_[0] + scaler.mean_[0], 
            means[:, 1] * scaler.scale_[1] + scaler.mean_[1], 
            c='red', s=200, alpha=0.8, marker='X')

plt.colorbar(scatter, label='聚类')
plt.title(f"EM聚类结果 (组件数: {best_n_components})")
plt.xlabel('年收入 (k$)')
plt.ylabel('消费评分 (1-100)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'em_clusters.png'))
plt.show()

# 可视化聚类概率
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], 
                     c=data_with_clusters['Cluster_Probability'], cmap='viridis', 
                     alpha=0.8, s=100)
plt.colorbar(scatter, label='聚类概率')
plt.title("EM聚类概率")
plt.xlabel('年收入 (k$)')
plt.ylabel('消费评分 (1-100)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'em_probabilities.png'))
plt.show()

# 3D可视化
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
scatter = ax.scatter(X['Annual Income (k$)'], 
                    X['Spending Score (1-100)'], 
                    X['Age'],
                    c=labels, cmap='viridis', 
                    s=80, alpha=0.8)

# 绘制每个高斯分布的中心
ax.scatter(means[:, 0] * scaler.scale_[0] + scaler.mean_[0], 
           means[:, 1] * scaler.scale_[1] + scaler.mean_[1], 
           means[:, 2] * scaler.scale_[2] + scaler.mean_[2], 
           c='red', s=200, alpha=0.8, marker='X')

ax.set_xlabel('年收入 (k$)')
ax.set_ylabel('消费评分 (1-100)')
ax.set_zlabel('年龄')
ax.set_title(f"EM聚类结果 - 3D视图 (组件数: {best_n_components})")
plt.colorbar(scatter, label='聚类', pad=0.1)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'em_3d_visualization.png'))
plt.show()

# 可视化每个高斯分布的协方差矩阵
def plot_gaussian_ellipsoid(gmm, X, feature_indices=(0, 1), ax=None, colors=None):
    """绘制二维高斯分布的椭圆"""
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    
    if ax is None:
        ax = plt.gca()
    
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, gmm.n_components))
    
    for n, (color, mean, covar) in enumerate(zip(colors, gmm.means_, gmm.covariances_)):
        # 提取二维特征的均值
        mean_2d = mean[list(feature_indices)]
        
        # 提取二维特征的协方差矩阵
        if covar.ndim == 1:
            # 对角协方差矩阵
            covar_2d = np.diag([covar[i] for i in feature_indices])
        elif covar.ndim == 2:
            # 完整协方差矩阵
            idx = np.ix_(feature_indices, feature_indices)
            covar_2d = covar[idx]
        
        # 计算特征值和特征向量
        v, w = np.linalg.eigh(covar_2d)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # 转换为度
        
        # 标准差
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        
        # 绘制椭圆
        for nsig in range(1, 4):
            # 计算椭圆中心点的原始坐标
            center_x = mean_2d[0] * scaler.scale_[feature_indices[0]] + scaler.mean_[feature_indices[0]]
            center_y = mean_2d[1] * scaler.scale_[feature_indices[1]] + scaler.mean_[feature_indices[1]]
            
            ell = Ellipse((center_x, center_y), 
                         nsig * v[0], nsig * v[1],
                         angle=angle, color=color, alpha=0.2)
            ell.set_clip_box(ax.bbox)
            ax.add_artist(ell)
        
        # 绘制中心点
        ax.scatter(center_x, center_y, 
                  c=color, s=200, alpha=0.8, marker='X')

# 绘制高斯分布椭圆
plt.figure(figsize=(12, 10))
ax = plt.gca()

# 绘制数据点
scatter = plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], 
                     c=labels, cmap='viridis', alpha=0.5, s=80)

# 绘制高斯分布椭圆
colors = plt.cm.viridis(np.linspace(0, 1, best_n_components))
plot_gaussian_ellipsoid(gmm, X_scaled, feature_indices=(0, 1), ax=ax, colors=colors)

plt.colorbar(scatter, label='聚类')
plt.title(f"EM聚类结果与高斯分布 (组件数: {best_n_components})")
plt.xlabel('年收入 (k$)')
plt.ylabel('消费评分 (1-100)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'em_gaussian_ellipsoids.png'))
plt.show()

# 分析每个聚类的特征
def analyze_clusters(data, clusters, probabilities):
    """分析每个聚类的特征分布"""
    # 添加聚类标签和概率
    data_copy = data.copy()
    data_copy['Cluster'] = clusters
    data_copy['Probability'] = np.max(probabilities, axis=1)
    
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
    
    # 分析聚类概率分布
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='Probability', data=data_copy)
    plt.title('各聚类的概率分布')
    plt.xlabel('聚类')
    plt.ylabel('概率')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cluster_probabilities.png'))
    plt.show()
    
    return cluster_means

# 分析聚类特征
cluster_means = analyze_clusters(data, labels, probabilities)
print("\n各聚类的特征均值:")
print(cluster_means)

# 计算每个聚类的样本数量
cluster_counts = pd.Series(labels).value_counts().sort_index()
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
data_with_clusters.to_csv(os.path.join(output_folder, 'em_clustering_results.csv'), index=False)
print(f"\n聚类结果已保存至: {os.path.join(output_folder, 'em_clustering_results.csv')}")

# EM算法优势总结
print("\nEM算法(高斯混合模型)优势:")
print("1. 提供软聚类结果，每个样本属于每个聚类的概率")
print("2. 可以处理不同形状和大小的聚类")
print("3. 可以处理重叠的聚类")
print("4. 基于概率模型，有坚实的统计基础")
print("5. 可以使用BIC和AIC等信息准则自动选择最佳聚类数")