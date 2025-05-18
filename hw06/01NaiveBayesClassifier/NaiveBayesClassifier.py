import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建输出目录
output_folder = os.path.join('hw06', '01NaiveBayesClassifier', 'output')
os.makedirs(output_folder, exist_ok=True)

# 自定义朴素贝叶斯分类器
class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None  # 存储类别
        self.class_priors = None  # 存储先验概率
        self.means = None  # 存储每个特征的均值
        self.variances = None  # 存储每个特征的方差
        
    def fit(self, X, y):
        """
        训练朴素贝叶斯分类器
        
        参数:
        X: 特征矩阵，形状为 (n_samples, n_features)
        y: 目标变量，形状为 (n_samples,)
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # 初始化参数
        self.class_priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        
        # 计算每个类别的先验概率、均值和方差
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[i] = X_c.shape[0] / n_samples
            self.means[i, :] = X_c.mean(axis=0)
            self.variances[i, :] = X_c.var(axis=0) + 1e-9  # 添加小值避免方差为0
            
        return self
    
    def _calculate_likelihood(self, x, mean, var):
        """
        计算高斯概率密度函数
        
        参数:
        x: 单个特征值
        mean: 该特征的均值
        var: 该特征的方差
        
        返回:
        概率密度值
        """
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean)**2 / (2 * var))
    
    def _calculate_class_probability(self, x, class_idx):
        """
        计算样本属于某个类别的概率
        
        参数:
        x: 特征向量
        class_idx: 类别索引
        
        返回:
        概率值（对数形式，避免数值下溢）
        """
        log_prob = np.log(self.class_priors[class_idx])
        
        for j in range(len(x)):
            likelihood = self._calculate_likelihood(x[j], self.means[class_idx, j], self.variances[class_idx, j])
            log_prob += np.log(likelihood + 1e-9)  # 添加小值避免对0取对数
            
        return log_prob
    
    def predict_proba(self, X):
        """
        预测样本属于每个类别的概率
        
        参数:
        X: 特征矩阵，形状为 (n_samples, n_features)
        
        返回:
        概率矩阵，形状为 (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probs = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            for j in range(n_classes):
                probs[i, j] = self._calculate_class_probability(X[i], j)
                
        # 将对数概率转换为正常概率并归一化
        probs = np.exp(probs)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X):
        """
        预测样本的类别
        
        参数:
        X: 特征矩阵，形状为 (n_samples, n_features)
        
        返回:
        预测的类别，形状为 (n_samples,)
        """
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练自定义朴素贝叶斯分类器
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = nb_classifier.predict(X_test)
y_prob = nb_classifier.predict_proba(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"朴素贝叶斯分类器准确率: {accuracy:.4f}")

# 打印分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=class_names))

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('混淆矩阵')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))
plt.show()

# 可视化特征分布
plt.figure(figsize=(15, 10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    for j in range(3):
        plt.hist(X[y == j, i], alpha=0.5, label=class_names[j])
    plt.title(f'特征分布: {feature_names[i]}')
    plt.xlabel('特征值')
    plt.ylabel('频率')
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'feature_distributions.png'))
plt.show()

# 可视化决策边界（选择两个特征）
def plot_decision_boundary(X, y, model, feature_idx, title):
    # 提取两个特征
    X_selected = X[:, feature_idx]
    
    # 创建网格
    h = 0.1  # 网格步长
    x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
    y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 准备网格点的特征
    grid_points = np.zeros((xx.ravel().shape[0], X.shape[1]))
    # 填充选择的两个特征
    grid_points[:, feature_idx[0]] = xx.ravel()
    grid_points[:, feature_idx[1]] = yy.ravel()
    # 填充其他特征（使用平均值）
    for i in range(X.shape[1]):
        if i not in feature_idx:
            grid_points[:, i] = X[:, i].mean()
    
    # 预测网格点的类别
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    
    # 绘制训练点
    for i, c in enumerate(np.unique(y)):
        plt.scatter(X[y == c, feature_idx[0]], X[y == c, feature_idx[1]], 
                   label=class_names[i], edgecolor='k')
    
    plt.title(title)
    plt.xlabel(feature_names[feature_idx[0]])
    plt.ylabel(feature_names[feature_idx[1]])
    plt.legend()
    plt.savefig(os.path.join(output_folder, f'decision_boundary_{feature_idx[0]}_{feature_idx[1]}.png'))
    plt.show()

# 绘制不同特征组合的决策边界
feature_combinations = [(0, 1), (2, 3), (0, 2), (1, 3)]
for feature_idx in feature_combinations:
    plot_decision_boundary(X, y, nb_classifier, feature_idx, 
                          f'朴素贝叶斯决策边界: {feature_names[feature_idx[0]]} vs {feature_names[feature_idx[1]]}')

# 可视化每个类别的概率分布
plt.figure(figsize=(12, 6))
for i, class_name in enumerate(class_names):
    plt.subplot(1, 3, i+1)
    class_probs = y_prob[:, i]
    for j, target_class in enumerate(np.unique(y_test)):
        plt.hist(class_probs[y_test == target_class], alpha=0.5, bins=20, 
                label=f'真实类别: {class_names[target_class]}')
    plt.title(f'类别 {class_name} 的概率分布')
    plt.xlabel(f'属于类别 {class_name} 的概率')
    plt.ylabel('样本数量')
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'probability_distributions.png'))
plt.show()

# 与sklearn的朴素贝叶斯分类器比较
from sklearn.naive_bayes import GaussianNB

# 训练sklearn的高斯朴素贝叶斯分类器
sk_nb = GaussianNB()
sk_nb.fit(X_train, y_train)
sk_y_pred = sk_nb.predict(X_test)
sk_accuracy = accuracy_score(y_test, sk_y_pred)

print("\n模型比较:")
print(f"自定义朴素贝叶斯分类器准确率: {accuracy:.4f}")
print(f"Sklearn朴素贝叶斯分类器准确率: {sk_accuracy:.4f}")

# 可视化比较结果
plt.figure(figsize=(8, 6))
models = ['自定义朴素贝叶斯', 'Sklearn朴素贝叶斯']
accuracies = [accuracy, sk_accuracy]
plt.bar(models, accuracies, color=['blue', 'green'])
plt.ylim(0.8, 1.0)  # 设置y轴范围以便更好地显示差异
plt.title('模型准确率比较')
plt.ylabel('准确率')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
plt.savefig(os.path.join(output_folder, 'model_comparison.png'))
plt.show()

# 总结朴素贝叶斯分类器的优缺点
print("\n朴素贝叶斯分类器优点:")
print("1. 简单易实现，计算效率高")
print("2. 对小规模数据表现良好")
print("3. 对缺失数据不敏感")
print("4. 能处理多分类问题")
print("5. 对不相关特征的鲁棒性较好")

print("\n朴素贝叶斯分类器缺点:")
print("1. 假设特征之间相互独立，这在实际应用中往往不成立")
print("2. 对数据分布敏感，如果特征不符合高斯分布，性能可能下降")
print("3. 零频率问题：如果某个类别的特征在训练集中没有出现，会导致概率为零")
print("4. 估计的概率可能不准确，不适合需要精确概率的任务")