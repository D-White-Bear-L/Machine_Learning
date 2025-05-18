import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建输出目录
output_folder = os.path.join('hw05', '01perception', 'output')
os.makedirs(output_folder, exist_ok=True)

# 设置随机种子以确保结果可重现
np.random.seed(42)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        # 初始化权重和偏置
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 确保y是-1和1
        y_ = np.where(y <= 0, -1, 1)
        
        # 训练历史记录
        self.errors_ = []
        
        # 训练感知器
        for _ in range(self.n_iterations):
            errors = 0
            for idx, x_i in enumerate(X):
                # 计算预测值
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = 1 if linear_output >= 0 else -1
                
                # 更新权重
                if y_[idx] != y_predicted:
                    update = self.learning_rate * y_[idx]
                    self.weights += update * x_i
                    self.bias += update
                    errors += 1
            
            self.errors_.append(errors)
            
            # 如果没有错误，提前停止
            if errors == 0:
                break
                
        return self
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 由于感知器只能处理二分类问题，我们选择两个类别进行分类
# 选择前两个特征和前两个类别
X = X[:100, :2]  # 只取前两个特征：萼片长度和萼片宽度
y = y[:100]      # 只取前两个类别：Setosa和Versicolor

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练感知器模型
perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)
perceptron.fit(X_train, y_train)

# 在测试集上评估模型
accuracy = perceptron.score(X_test, y_test)
print(f"感知器模型准确率: {accuracy:.4f}")

# 获取预测结果
y_pred = perceptron.predict(X_test)

# 打印分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=[iris.target_names[0], iris.target_names[1]]))

# 可视化训练过程中的错误数量
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
plt.title('感知器训练过程中的错误数量')
plt.xlabel('迭代次数')
plt.ylabel('错误数量')
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'perceptron_errors.png'))
plt.show()

# 可视化决策边界
# 可视化决策边界
def plot_decision_boundary(X, y, model, title):
    # 设置网格的最小和最大值，更合理地设置范围
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # 创建网格点
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # 对网格点进行预测
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和散点图
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolor='k', s=100)
    
    # 计算决策边界中心点作为箭头起点
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    # 添加权重向量，从中心点开始，并调整箭头大小
    # 归一化权重向量以便更好地显示
    weight_norm = np.sqrt(model.weights[0]**2 + model.weights[1]**2)
    scale_factor = min(x_max - x_min, y_max - y_min) / 5  # 调整箭头长度为图表尺寸的1/5
    
    if weight_norm > 0:  # 避免除以零
        arrow_dx = scale_factor * model.weights[0] / weight_norm
        arrow_dy = scale_factor * model.weights[1] / weight_norm
        plt.arrow(center_x, center_y, arrow_dx, arrow_dy, 
                head_width=scale_factor/10, head_length=scale_factor/8, 
                fc='r', ec='r', width=scale_factor/30)
    
    plt.title(title)
    plt.xlabel('萼片长度 (cm)')
    plt.ylabel('萼片宽度 (cm)')
    plt.savefig(os.path.join(output_folder, 'perceptron_decision_boundary.png'))
    plt.show()

# 绘制决策边界
plot_decision_boundary(X, y, perceptron, '感知器的决策边界')

# 学习率对感知器性能的影响
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0]
accuracies = []

plt.figure(figsize=(12, 8))

for i, lr in enumerate(learning_rates):
    model = Perceptron(learning_rate=lr, n_iterations=1000)
    model.fit(X_train, y_train)
    
    # 记录准确率
    acc = model.score(X_test, y_test)
    accuracies.append(acc)
    
    # 绘制训练过程中的错误数量
    plt.subplot(2, 3, i+1)
    plt.plot(range(1, len(model.errors_) + 1), model.errors_, marker='.')
    plt.title(f'学习率 = {lr}')
    plt.xlabel('迭代次数')
    plt.ylabel('错误数量')
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'perceptron_learning_rates.png'))
plt.show()

# 绘制学习率与准确率的关系
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, accuracies, marker='o', linestyle='-')
plt.title('学习率对感知器准确率的影响')
plt.xlabel('学习率')
plt.ylabel('准确率')
plt.xscale('log')
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'perceptron_accuracy_vs_learning_rate.png'))
plt.show()

print("\n不同学习率的准确率:")
for lr, acc in zip(learning_rates, accuracies):
    print(f"学习率 {lr}: 准确率 {acc:.4f}")

# 迭代次数对感知器性能的影响
iterations = [10, 50, 100, 500, 1000]
accuracies = []

plt.figure(figsize=(12, 8))

for i, n_iter in enumerate(iterations):
    model = Perceptron(learning_rate=0.01, n_iterations=n_iter)
    model.fit(X_train, y_train)
    
    # 记录准确率
    acc = model.score(X_test, y_test)
    accuracies.append(acc)
    
    # 绘制训练过程中的错误数量
    plt.subplot(2, 3, i+1)
    plt.plot(range(1, len(model.errors_) + 1), model.errors_, marker='.')
    plt.title(f'迭代次数 = {n_iter}')
    plt.xlabel('迭代次数')
    plt.ylabel('错误数量')
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'perceptron_iterations.png'))
plt.show()

# 绘制迭代次数与准确率的关系
plt.figure(figsize=(10, 6))
plt.plot(iterations, accuracies, marker='o', linestyle='-')
plt.title('迭代次数对感知器准确率的影响')
plt.xlabel('迭代次数')
plt.ylabel('准确率')
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'perceptron_accuracy_vs_iterations.png'))
plt.show()

print("\n不同迭代次数的准确率:")
for n_iter, acc in zip(iterations, accuracies):
    print(f"迭代次数 {n_iter}: 准确率 {acc:.4f}")

# 感知器的优缺点总结
print("\n感知器算法优点:")
print("1. 简单易实现，计算效率高")
print("2. 对于线性可分的数据，能够找到一个完美的分类超平面")
print("3. 是神经网络和深度学习的基础")

print("\n感知器算法缺点:")
print("1. 只能解决线性可分的问题")
print("2. 对非线性问题无能为力")
print("3. 对初始权重和学习率敏感")
print("4. 无法处理多分类问题（需要扩展）")