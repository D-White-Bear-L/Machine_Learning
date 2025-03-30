import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from collections import Counter
import graphviz
import os

class CARTDecisionTree:
    def __init__(self, max_depth=None):
        self.tree = None
        self.feature_names = None
        self.max_depth = max_depth
        
    def calculate_gini(self, y):
        """计算基尼不纯度"""
        if len(y) == 0:
            return 0
        
        # 计算各类别的概率
        counter = Counter(y)
        gini = 1.0
        for count in counter.values():
            p = count / len(y)
            gini -= p ** 2
        
        return gini
    
    def calculate_gini_gain(self, X, y, feature_idx, threshold):
        """计算特征的基尼增益"""
        # 计算父节点的基尼不纯度
        gini_parent = self.calculate_gini(y)
        
        # 按特征值划分数据集
        left_indices = X[:, feature_idx] <= threshold
        right_indices = ~left_indices
        
        left_y = y[left_indices]
        right_y = y[right_indices]
        
        # 如果划分后的子集为空，返回0
        if len(left_y) == 0 or len(right_y) == 0:
            return 0
        
        # 计算子集的基尼不纯度，并加权求和
        n = len(y)
        gini_children = (len(left_y) / n) * self.calculate_gini(left_y) + \
                        (len(right_y) / n) * self.calculate_gini(right_y)
        
        # 计算基尼增益
        gini_gain = gini_parent - gini_children
        return gini_gain
    
    def find_best_split(self, X, y, available_features):
        """找到最佳分裂特征和阈值"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in available_features:
            # 获取特征的所有唯一值
            unique_values = np.sort(np.unique(X[:, feature_idx]))
            
            # 计算相邻值的中点作为可能的阈值
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                gain = self.calculate_gini_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, X, y, available_features, depth=0):
        """递归构建决策树"""
        # 如果所有样本属于同一类别，返回该类别
        if len(set(y)) == 1:
            return y[0]
        
        # 如果没有可用特征或达到最大深度，返回多数类
        if len(available_features) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return Counter(y).most_common(1)[0][0]
        
        # 选择最佳分裂特征和阈值
        best_feature, best_threshold, best_gain = self.find_best_split(X, y, available_features)
        
        # 如果无法找到有效的分裂，返回多数类
        if best_feature is None or best_gain <= 0:
            return Counter(y).most_common(1)[0][0]
        
        # 创建以最佳特征为根的树
        tree = {
            'feature': self.feature_names[best_feature],
            'index': best_feature,
            'threshold': best_threshold,
            'gini': self.calculate_gini(y),
            'samples': len(y),
            'value': [np.sum(y == 0), np.sum(y == 1)],
            'left': None,
            'right': None
        }
        
        # 按特征值划分数据集
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        # 递归构建左子树和右子树
        tree['left'] = self.build_tree(X[left_indices], y[left_indices], available_features, depth + 1)
        tree['right'] = self.build_tree(X[right_indices], y[right_indices], available_features, depth + 1)
        
        return tree
    
    def fit(self, X, y, feature_names):
        """训练决策树"""
        self.feature_names = feature_names
        available_features = set(range(X.shape[1]))
        self.tree = self.build_tree(X, y, available_features)
        return self
    
    def predict_one(self, x, tree=None):
        """预测单个样本的类别"""
        if tree is None:
            tree = self.tree
        
        # 如果树是叶节点（类别），直接返回
        if not isinstance(tree, dict):
            return tree
        
        # 获取样本在分裂特征上的值
        feature_idx = tree['index']
        threshold = tree['threshold']
        
        # 根据特征值选择子树
        if x[feature_idx] <= threshold:
            return self.predict_one(x, tree['left'])
        else:
            return self.predict_one(x, tree['right'])
    
    def predict(self, X):
        """预测多个样本的类别"""
        return np.array([self.predict_one(x) for x in X])
    
    def print_tree(self, tree=None, indent=""):
        """打印决策树结构"""
        if tree is None:
            tree = self.tree
        
        if not isinstance(tree, dict):
            print(f"{indent}决策: {'是' if tree == 1 else '否'}")
            return
        
        # 获取节点信息
        feature = tree['feature']
        threshold = tree['threshold']
        gini = tree['gini']
        samples = tree['samples']
        value = tree['value']
        
        print(f"{indent}{feature} <= {threshold:.3f}")
        print(f"{indent}  ├─ gini = {gini:.3f}")
        print(f"{indent}  ├─ samples = {samples}")
        print(f"{indent}  ├─ value = {value}")
        
        # 递归打印子树
        print(f"{indent}  ├─ 左子树 (是):")
        self.print_tree(tree['left'], indent + "  │  ")
        print(f"{indent}  ├─ 右子树 (否):")
        self.print_tree(tree['right'], indent + "  │  ")
    
    def visualize_tree(self, output_file=None):
        """使用graphviz可视化决策树"""
        dot = graphviz.Digraph(comment='CART Decision Tree')
        
        # 设置中文字体支持
        dot.attr(fontname='SimHei')
        
        def add_nodes(tree, parent=None, edge_label=None, node_id=None):
            if node_id is None:
                node_id = [0]
            
            if not isinstance(tree, dict):
                # 叶节点
                class_name = "是" if tree == 1 else "否"
                dot.node(str(node_id[0]), f'决策: {class_name}', shape='box', style='filled', color='lightblue')
                if parent is not None:
                    dot.edge(str(parent), str(node_id[0]), label=edge_label)
                current_id = node_id[0]
                node_id[0] += 1
                return current_id
            
            # 内部节点
            feature = tree['feature']
            threshold = tree['threshold']
            gini = tree['gini']
            samples = tree['samples']
            value = tree['value']
            
            label = f"{feature} <= {threshold:.3f}\\ngini = {gini:.3f}\\nsamples = {samples}\\nvalue = {value}"
            dot.node(str(node_id[0]), label, shape='ellipse', style='filled', color='lightgreen')
            
            if parent is not None:
                dot.edge(str(parent), str(node_id[0]), label=edge_label)
            
            current_id = node_id[0]
            node_id[0] += 1
            
            # 添加左子树
            add_nodes(tree['left'], current_id, "是", node_id)
            
            # 添加右子树
            add_nodes(tree['right'], current_id, "否", node_id)
            
            return current_id
        
        add_nodes(self.tree)
        
        if output_file:
            dot.render(output_file, format='png', cleanup=True)
        
        return dot
    
    def visualize_tree_matplotlib(self, output_file=None):
        """使用matplotlib可视化决策树"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # 计算树的深度和宽度
        def get_tree_depth_and_width(tree, depth=0):
            if not isinstance(tree, dict):
                return depth, 1
            
            left_depth, left_width = get_tree_depth_and_width(tree['left'], depth + 1)
            right_depth, right_width = get_tree_depth_and_width(tree['right'], depth + 1)
            
            return max(left_depth, right_depth), left_width + right_width
        
        max_depth, max_width = get_tree_depth_and_width(self.tree)
        
        # 计算节点位置
        def calculate_node_positions(tree, x=0.5, y=0.9, width=1.0, depth=0, positions=None):
            if positions is None:
                positions = {}
            
            if not isinstance(tree, dict):
                positions[id(tree)] = (x, y, tree)
                return positions, width
            
            positions[id(tree)] = (x, y, tree)
            
            # 计算子节点位置
            child_y = y - 0.15
            left_width = width * 0.5
            right_width = width * 0.5
            
            # 计算左子树位置
            left_x = x - width / 4
            positions, _ = calculate_node_positions(tree['left'], left_x, child_y, left_width, depth + 1, positions)
            
            # 计算右子树位置
            right_x = x + width / 4
            positions, _ = calculate_node_positions(tree['right'], right_x, child_y, right_width, depth + 1, positions)
            
            return positions, width
        
        # 计算所有节点的位置
        node_positions, _ = calculate_node_positions(self.tree)
        
        # 绘制节点和连接线
        def draw_tree(tree, node_positions):
            # 绘制所有的连接线
            for node_id, (x, y, node) in node_positions.items():
                if isinstance(node, dict):
                    # 获取左右子树的位置
                    if id(node['left']) in node_positions:
                        left_x, left_y, _ = node_positions[id(node['left'])]
                        # 绘制连接线
                        ax.plot([x, left_x], [y, left_y], 'k-')
                        # 添加边标签
                        mid_x = (x + left_x) / 2
                        mid_y = (y + left_y) / 2
                        ax.text(mid_x, mid_y, "是", ha='center', va='center', 
                               fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
                    
                    if id(node['right']) in node_positions:
                        right_x, right_y, _ = node_positions[id(node['right'])]
                        # 绘制连接线
                        ax.plot([x, right_x], [y, right_y], 'k-')
                        # 添加边标签
                        mid_x = (x + right_x) / 2
                        mid_y = (y + right_y) / 2
                        ax.text(mid_x, mid_y, "否", ha='center', va='center', 
                               fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
            
            # 绘制所有的节点
            for node_id, (x, y, node) in node_positions.items():
                if isinstance(node, dict):
                    # 内部节点
                    feature = node['feature']
                    threshold = node['threshold']
                    gini = node['gini']
                    samples = node['samples']
                    value = node['value']
                    
                    rect = Rectangle((x-0.1, y-0.05), 0.2, 0.1, 
                                    fill=True, edgecolor='black', facecolor='lightgreen', 
                                    linewidth=1, alpha=0.8)
                    ax.add_patch(rect)
                    ax.text(x, y+0.02, f"{feature} <= {threshold:.3f}", ha='center', va='center', fontsize=9)
                    ax.text(x, y-0.01, f"gini = {gini:.2f}", ha='center', va='center', fontsize=8)
                    ax.text(x, y-0.03, f"samples = {samples}", ha='center', va='center', fontsize=8)
                    ax.text(x, y-0.05, f"value = {value}", ha='center', va='center', fontsize=8)
                else:
                    # 叶节点
                    class_name = "是" if node == 1 else "否"
                    rect = Rectangle((x-0.1, y-0.05), 0.2, 0.1, 
                                    fill=True, edgecolor='black', facecolor='lightblue', 
                                    linewidth=1, alpha=0.8)
                    ax.add_patch(rect)
                    ax.text(x, y, f"class = {class_name}", ha='center', va='center', fontsize=9)
        
        # 绘制树
        draw_tree(self.tree, node_positions)
        
        # 调整图形大小
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # 保存图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"决策树已保存为: {output_file}")
        
        plt.show()
        return fig, ax
    
    def plot_feature_importance(self, X, y, output_file=None):
        """绘制特征重要性图"""
        # 计算特征重要性
        importances = self.calculate_feature_importance(X, y)
        
        # 按重要性排序
        sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 绘制条形图
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_importances.keys(), sorted_importances.values(), color='skyblue')
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.title('特征重要性')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 在条形上方显示具体数值
        for i, (feature, importance) in enumerate(sorted_importances.items()):
            plt.text(i, importance + 0.005, f'{importance:.3f}', ha='center')
        
        if output_file:
            plt.savefig(output_file)
        
        plt.show()
        
        return sorted_importances
    
    def calculate_feature_importance(self, X, y):
        """计算特征重要性"""
        importances = {feature: 0 for feature in self.feature_names}
        
        def traverse_tree(tree, samples_count):
            if not isinstance(tree, dict):
                return
            
            feature = tree['feature']
            gini_decrease = tree['gini'] * tree['samples'] / samples_count
            
            if isinstance(tree['left'], dict):
                gini_decrease -= tree['left']['gini'] * tree['left']['samples'] / samples_count
            
            if isinstance(tree['right'], dict):
                gini_decrease -= tree['right']['gini'] * tree['right']['samples'] / samples_count
            
            importances[feature] += gini_decrease
            
            traverse_tree(tree['left'], samples_count)
            traverse_tree(tree['right'], samples_count)
        
        traverse_tree(self.tree, len(y))
        
        # 归一化特征重要性
        total = sum(importances.values())
        if total > 0:
            for feature in importances:
                importances[feature] /= total
        
        return importances

# 主程序
if __name__ == "__main__":
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建数据集
    # 年龄: 0-<=30, 1-31...40, 2->40
    # 收入: 0-high, 1-medium, 2-low
    # 学生: 0-no, 1-yes
    # 信用评级: 0-fair, 1-excellent
    # 购买电脑: 0-no, 1-yes
    X = np.array([
        [0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [2, 1, 0, 0],
        [2, 2, 1, 0], [2, 2, 1, 1], [1, 2, 1, 1], [0, 1, 0, 0],
        [0, 2, 1, 0], [2, 1, 1, 0], [0, 1, 1, 1], [1, 1, 0, 1],
        [1, 0, 1, 0], [2, 1, 0, 1]
    ])
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])
    
    # 特征名称
    feature_names = ["age", "income", "student", "credit_rating"]
    
    # 创建输出文件夹
    output_folder = r"c:\programme\data_science\artificial intelligence\homwork_ex\hw\hw02\cart_output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 创建并训练CART决策树
    tree = CARTDecisionTree(max_depth=3)
    tree.fit(X, y, feature_names)
    
    # 打印决策树
    print("CART决策树结构:")
    tree.print_tree()
    
    # 可视化决策树
    output_path = os.path.join(output_folder, "decision_tree")
    tree.visualize_tree(output_path)
    print(f"\n决策树已保存为图片: {output_path}.png")
    
    # 使用matplotlib可视化决策树
    matplotlib_output_path = os.path.join(output_folder, "decision_tree_matplotlib.png")
    tree.visualize_tree_matplotlib(matplotlib_output_path)
    print(f"\n使用matplotlib的决策树已保存为图片: {matplotlib_output_path}")
    
    # 绘制特征重要性图
    importance_chart_path = os.path.join(output_folder, "feature_importance_chart.png")
    tree.plot_feature_importance(X, y, importance_chart_path)
    print(f"特征重要性图已保存为: {importance_chart_path}")
    
    # 进行预测
    X_test = np.array([[0, 1, 0, 0], [2, 2, 1, 0]])  # 测试数据
    predictions = tree.predict(X_test)
    print("\n预测结果:", ["否" if p == 0 else "是" for p in predictions])