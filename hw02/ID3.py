import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from collections import Counter
import graphviz

class ID3DecisionTree:
    def __init__(self):
        self.tree = None
        self.feature_names = None
        
    def calculate_entropy(self, y):
        """计算数据集的熵"""
        if len(y) == 0:
            return 0
        
        # 计算各类别的概率
        counter = Counter(y)
        entropy = 0
        for count in counter.values():
            p = count / len(y)
            entropy -= p * math.log2(p)
        
        return entropy
    
    def calculate_information_gain(self, X, y, feature_idx):
        """计算特征的信息增益"""
        # 计算数据集的熵
        entropy_parent = self.calculate_entropy(y)
        
        # 按特征值划分数据集
        feature_values = set(X[:, feature_idx])
        entropy_children = 0
        
        for value in feature_values:
            # 获取特征值为value的子集
            indices = X[:, feature_idx] == value
            subset_y = y[indices]
            
            # 计算子集的熵，并加权求和
            weight = len(subset_y) / len(y)
            entropy_children += weight * self.calculate_entropy(subset_y)
        
        # 计算信息增益
        information_gain = entropy_parent - entropy_children
        return information_gain
    
    def find_best_feature(self, X, y, available_features):
        """找到最佳分裂特征"""
        best_gain = -1
        best_feature = None
        
        for feature_idx in available_features:
            gain = self.calculate_information_gain(X, y, feature_idx)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
        
        return best_feature, best_gain
    
    def build_tree(self, X, y, available_features):
        """递归构建决策树"""
        # 如果所有样本属于同一类别，返回该类别
        if len(set(y)) == 1:
            return y[0]
        
        # 如果没有可用特征，返回多数类
        if len(available_features) == 0:
            return Counter(y).most_common(1)[0][0]
        
        # 选择最佳分裂特征
        best_feature, _ = self.find_best_feature(X, y, available_features)
        
        # 创建以最佳特征为根的树
        tree = {self.feature_names[best_feature]: {}}
        
        # 对最佳特征的每个取值，递归构建子树
        feature_values = set(X[:, best_feature])
        for value in feature_values:
            # 获取特征值为value的子集
            indices = X[:, best_feature] == value
            subset_X = X[indices]
            subset_y = y[indices]
            
            # 创建新的可用特征集合（排除当前特征）
            new_available_features = available_features.copy()
            new_available_features.remove(best_feature)
            
            # 递归构建子树
            subtree = self.build_tree(subset_X, subset_y, new_available_features)
            tree[self.feature_names[best_feature]][value] = subtree
        
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
        
        # 获取根节点的特征
        feature = list(tree.keys())[0]
        feature_idx = self.feature_names.index(feature)
        
        # 获取样本在该特征上的值
        value = x[feature_idx]
        
        # 如果该值不在树中，返回多数类
        if value not in tree[feature]:
            # 简单处理：返回第一个子树的结果
            subtree = list(tree[feature].values())[0]
            if isinstance(subtree, dict):
                return self.predict_one(x, subtree)
            else:
                return subtree
        
        # 递归预测
        subtree = tree[feature][value]
        return self.predict_one(x, subtree)
    
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
        
        # 获取根节点的特征
        feature = list(tree.keys())[0]
        print(f"{indent}{feature}")
        
        # 递归打印子树
        for value, subtree in tree[feature].items():
            value_text = self.get_feature_value_text(feature, value)
            print(f"{indent}  ├─ {value_text}")
            self.print_tree(subtree, indent + "  │  ")
    
    def get_feature_value_text(self, feature, value):
        """获取特征值的文字说明"""
        feature_value_map = {
            "age": {0: "<=30", 1: "31...40", 2: ">40"},
            "income": {0: "high", 1: "medium", 2: "low"},
            "student": {0: "no", 1: "yes"},
            "credit_rating": {0: "fair", 1: "excellent"}
        }
        
        if feature in feature_value_map and value in feature_value_map[feature]:
            return feature_value_map[feature][value]
        return str(value)
    
    def calculate_all_gains(self, X, y):
        """计算所有特征的信息增益"""
        gains = {}
        for i, feature in enumerate(self.feature_names):
            gain = self.calculate_information_gain(X, y, i)
            gains[feature] = gain
        return gains
    
    def visualize_with_sklearn(self, X, y, output_file=None):
        """使用scikit-learn的决策树可视化功能"""
        from sklearn import tree
        from sklearn.tree import DecisionTreeClassifier
        import matplotlib.pyplot as plt
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 训练sklearn决策树
        clf = DecisionTreeClassifier(criterion='entropy')
        clf.fit(X, y)
        
        # 创建图形
        plt.figure(figsize=(15, 10))
        
        # 绘制决策树
        tree.plot_tree(clf, feature_names=self.feature_names, 
                      class_names=["否", "是"], 
                      filled=True, rounded=True, fontsize=10)
        
        # 保存图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"决策树已保存为: {output_file}")
        
        plt.show()
    
    def visualize_tree_matplotlib(self, output_file=None):
        """使用matplotlib可视化决策树，类似于sklearn的决策树可视化"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.path import Path
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        
        # 计算树的深度和宽度
        def get_tree_depth_and_width(tree, depth=0):
            if not isinstance(tree, dict):
                return depth, 1
            
            feature = list(tree.keys())[0]
            max_depth = depth
            total_width = 0
            
            for value, subtree in tree[feature].items():
                sub_depth, sub_width = get_tree_depth_and_width(subtree, depth + 1)
                max_depth = max(max_depth, sub_depth)
                total_width += sub_width
            
            return max_depth, max(total_width, 1)
        
        max_depth, max_width = get_tree_depth_and_width(self.tree)
        
        # 计算节点位置
        def calculate_node_positions(tree, x=0.5, y=0.9, width=1.0, depth=0, positions=None):
            if positions is None:
                positions = {}
            
            if not isinstance(tree, dict):
                positions[id(tree)] = (x, y, tree)
                return positions, width
            
            feature = list(tree.keys())[0]
            positions[id(tree)] = (x, y, feature)
            
            # 计算子节点位置
            values = list(tree[feature].keys())
            n_children = len(values)
            
            # 计算子节点的水平间距
            child_y = y - 0.2
            total_width = 0
            child_positions = []
            
            # 第一次遍历计算每个子树的宽度
            for value in values:
                subtree = tree[feature][value]
                _, subtree_width = calculate_node_positions(subtree, 0, 0, width / n_children, depth + 1, {})
                total_width += subtree_width
                child_positions.append(subtree_width)
            
            # 第二次遍历根据宽度分配位置
            current_x = x - total_width / 2
            for i, value in enumerate(values):
                subtree = tree[feature][value]
                child_width = child_positions[i]
                child_x = current_x + child_width / 2
                positions, _ = calculate_node_positions(subtree, child_x, child_y, child_width, depth + 1, positions)
                current_x += child_width
            
            return positions, total_width
        
        # 计算所有节点的位置
        node_positions, _ = calculate_node_positions(self.tree)
        
        # 绘制节点和连接线
        def draw_tree(tree, node_positions):
            # 绘制所有的连接线
            for node_id, (x, y, node_value) in node_positions.items():
                if isinstance(node_value, dict):
                    feature = list(node_value.keys())[0]
                    for value, subtree in node_value[feature].items():
                        if id(subtree) in node_positions:
                            child_x, child_y, _ = node_positions[id(subtree)]
                            # 绘制连接线
                            ax.plot([x, child_x], [y, child_y], 'k-')
                            
                            # 添加边标签
                            value_text = self.get_feature_value_text(feature, value)
                            mid_x = (x + child_x) / 2
                            mid_y = (y + child_y) / 2
                            ax.text(mid_x, mid_y, value_text, ha='center', va='center', 
                                   fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
            
            # 绘制所有的节点
            for node_id, (x, y, node_value) in node_positions.items():
                if isinstance(node_value, dict):
                    # 内部节点
                    feature = list(node_value.keys())[0]
                    rect = patches.Rectangle((x-0.1, y-0.05), 0.2, 0.1, 
                                            fill=True, edgecolor='black', facecolor='lightgreen', 
                                            linewidth=1, alpha=0.8)
                    ax.add_patch(rect)
                    ax.text(x, y, f"{feature}", ha='center', va='center', fontsize=10)
                else:
                    # 叶节点
                    class_name = "是" if node_value == 1 else "否"
                    rect = patches.Rectangle((x-0.1, y-0.05), 0.2, 0.1, 
                                            fill=True, edgecolor='black', facecolor='lightblue', 
                                            linewidth=1, alpha=0.8)
                    ax.add_patch(rect)
                    ax.text(x, y, f"class = {class_name}", ha='center', va='center', fontsize=10)
        
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

    def visualize_tree(self, output_file=None):
        """使用graphviz可视化决策树"""
        dot = graphviz.Digraph(comment='ID3 Decision Tree')
        
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
            feature = list(tree.keys())[0]
            dot.node(str(node_id[0]), feature, shape='ellipse', style='filled', color='lightgreen')
            
            if parent is not None:
                dot.edge(str(parent), str(node_id[0]), label=edge_label)
            
            current_id = node_id[0]
            node_id[0] += 1
            
            for value, subtree in tree[feature].items():
                value_text = self.get_feature_value_text(feature, value)
                add_nodes(subtree, current_id, value_text, node_id)
            
            return current_id
        
        add_nodes(self.tree)
        
        if output_file:
            dot.render(output_file, format='png', cleanup=True)
        
        return dot
    
    def plot_information_gain(self, X, y, output_file=None):
        """绘制信息增益比较图"""
        gains = self.calculate_all_gains(X, y)
        
        # 按信息增益排序
        sorted_gains = dict(sorted(gains.items(), key=lambda x: x[1], reverse=True))
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 绘制条形图
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_gains.keys(), sorted_gains.values(), color='skyblue')
        plt.xlabel('特征')
        plt.ylabel('信息增益')
        plt.title('各特征的信息增益比较')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 在条形上方显示具体数值
        for i, (feature, gain) in enumerate(sorted_gains.items()):
            plt.text(i, gain + 0.005, f'{gain:.3f}', ha='center')
        
        if output_file:
            plt.savefig(output_file)
        
        plt.show()
        
        return sorted_gains

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
    import os
    output_folder = r"c:\programme\data_science\artificial intelligence\homwork_ex\hw\hw02\id3_output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 创建并训练ID3决策树
    tree = ID3DecisionTree()
    tree.fit(X, y, feature_names)
    
    # 打印决策树
    print("ID3决策树结构:")
    tree.print_tree()
    
    # 计算数据集的熵
    entropy = tree.calculate_entropy(y)
    print(f"\n数据集的熵: {entropy:.3f}位")
    
    # 计算各特征的信息增益
    print("\n各特征的信息增益:")
    gains = tree.calculate_all_gains(X, y)
    for feature, gain in sorted(gains.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {gain:.3f}位")
    
    # 可视化决策树
    output_path = os.path.join(output_folder, "decision_tree")
    tree.visualize_tree(output_path)
    print(f"\n决策树已保存为图片: {output_path}.png")
    
    # 绘制信息增益比较图
    gain_chart_path = os.path.join(output_folder, "information_gain_chart")
    tree.plot_information_gain(X, y, gain_chart_path)
    print(f"信息增益比较图已保存为: {gain_chart_path}.png")
    
    # 进行预测
    X_test = np.array([[0, 1, 0, 0], [2, 2, 1, 0]])  # 测试数据
    predictions = tree.predict(X_test)
    print("\n预测结果:", ["否" if p == 0 else "是" for p in predictions])
    
    # 在主程序末尾添加
    # 使用scikit-learn可视化决策树
    sklearn_output_path = os.path.join(output_folder, "decision_tree_sklearn.png")
    tree.visualize_with_sklearn(X, y, sklearn_output_path)
    print(f"\n使用scikit-learn的决策树已保存为图片: {sklearn_output_path}")