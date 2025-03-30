import numpy as np
import pandas as pd
from collections import Counter
import math

class ID3DecisionTree:
    def __init__(self, max_depth=None):
        """
        初始化ID3决策树
        参数:
            max_depth: 树的最大深度，用于防止过拟合
        """
        self.max_depth = max_depth  # 控制树的最大深度，防止过拟合
        self.tree = None  # 存储训练好的决策树
    
    def fit(self, X, y, feature_names=None):
        """
        训练ID3决策树
        参数:
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名称列表
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]  # 如果未提供特征名称，则自动生成
        
        # 将数据转换为DataFrame以便于处理
        data = pd.DataFrame(X, columns=feature_names)  # 特征数据转为DataFrame
        data['target'] = y  # 添加目标变量列
        
        # 构建决策树
        self.tree = self._build_tree(data, data.columns[:-1], 'target', 0)  # 递归构建决策树，从深度0开始
        return self
    
    def predict(self, X):
        """
        使用训练好的决策树进行预测
        参数:
            X: 特征矩阵
        返回:
            预测结果
        """
        if self.tree is None:
            raise Exception("模型尚未训练，请先调用fit方法")  # 确保模型已训练
        
        # 将输入转换为DataFrame
        if not isinstance(X, pd.DataFrame):
            # 使用原始特征名称创建DataFrame
            original_feature_names = ["天气", "温度", "湿度", "风力"]  # 使用与训练时相同的原始特征名称
            X = pd.DataFrame(X, columns=original_feature_names)  # 转换为DataFrame便于处理
        
        # 对每个样本进行预测
        predictions = []
        for _, sample in X.iterrows():
            predictions.append(self._predict_sample(sample, self.tree))  # 对每个样本递归预测
        
        return np.array(predictions)  # 返回numpy数组形式的预测结果
    
    def _get_feature_names(self, tree, feature_names=None):
        """
        从决策树中提取所有特征名称
        
        参数:
            tree: 决策树或子树
            feature_names: 已收集的特征名称列表
        
        返回:
            特征名称列表
        """
        if feature_names is None:
            feature_names = []  # 初始化特征名称列表
        
        # 如果是叶节点，直接返回
        if not isinstance(tree, dict):
            return feature_names  # 叶节点不包含特征，直接返回
        
        # 获取当前节点的特征并添加到列表中
        feature = list(tree.keys())[0]  # 获取当前节点的特征名
        if feature not in feature_names:
            feature_names.append(feature)  # 避免重复添加特征
        
        # 递归处理子树
        for subtree in tree[feature].values():
            if isinstance(subtree, dict):
                # 修复：保存递归调用的结果
                feature_names = self._get_feature_names(subtree, feature_names)  # 递归收集子树中的特征
        
        return feature_names
    
    def _predict_sample(self, sample, tree):
        """
        对单个样本进行预测
        
        参数:
            sample: 单个样本
            tree: 决策树或子树
        
        返回:
            预测的类别
        """
        # 如果是叶节点，直接返回类别
        if not isinstance(tree, dict):
            return tree  # 到达叶节点，返回类别值
        
        # 获取当前节点的特征
        feature = list(tree.keys())[0]  # 获取决策特征
        
        # 获取样本在该特征上的值
        value = sample[feature]  # 获取样本在当前特征上的取值
        
        # 如果该值不在决策树中，返回最常见的类别
        if value not in tree[feature]:
            # 找出所有叶节点的值
            leaf_values = []
            for subtree in tree[feature].values():
                if not isinstance(subtree, dict):
                    leaf_values.append(subtree)  # 收集直接子节点中的叶节点值
                else:
                    # 递归收集所有叶节点
                    self._collect_leaf_values(subtree, leaf_values)  # 递归收集更深层的叶节点值
            
            # 返回最常见的类别
            return Counter(leaf_values).most_common(1)[0][0]  # 返回出现频率最高的类别
        
        # 递归预测
        return self._predict_sample(sample, tree[feature][value])  # 根据特征值递归向下预测
    
    def _collect_leaf_values(self, tree, values):
        """
        收集树中所有叶节点的值
        
        参数:
            tree: 决策树或子树
            values: 存储叶节点值的列表
        """
        if not isinstance(tree, dict):
            values.append(tree)  # 如果是叶节点，添加其值
            return
        
        feature = list(tree.keys())[0]  # 获取当前节点的特征
        for subtree in tree[feature].values():
            if not isinstance(subtree, dict):
                values.append(subtree)  # 收集直接子节点中的叶节点值
            else:
                self._collect_leaf_values(subtree, values)  # 递归收集更深层的叶节点值
    
    def _build_tree(self, data, features, target, depth):
        """
        递归构建决策树
        
        参数:
            data: 数据集
            features: 可用特征列表
            target: 目标变量名称
            depth: 当前深度
        
        返回:
            决策树或叶节点
        """
        # 获取目标变量的所有取值及其计数
        target_counts = Counter(data[target])  # 统计目标变量各类别的出现次数
        
        # 如果只有一种类别，返回该类别
        if len(target_counts) == 1:
            return list(target_counts.keys())[0]  # 数据集中只有一个类别，直接返回该类别
        
        # 如果没有可用特征或达到最大深度，返回最常见的类别
        if len(features) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return target_counts.most_common(1)[0][0]  # 返回出现频率最高的类别作为叶节点
        
        # 计算信息增益，选择最佳特征
        best_feature = self._choose_best_feature(data, features, target)  # 选择信息增益最大的特征
        
        # 创建以最佳特征为根节点的树
        tree = {best_feature: {}}  # 以最佳特征创建新的子树
        
        # 对最佳特征的每个取值，递归构建子树
        for value in data[best_feature].unique():  # 遍历特征的所有可能取值
            # 获取特征值为value的子数据集
            sub_data = data[data[best_feature] == value].drop(best_feature, axis=1)  # 按特征值分割数据集
            
            # 如果子数据集为空，使用父节点中最常见的类别
            if len(sub_data) == 0:
                tree[best_feature][value] = target_counts.most_common(1)[0][0]  # 处理空子集情况
            else:
                # 递归构建子树
                remaining_features = [f for f in features if f != best_feature]  # 移除已使用的特征
                tree[best_feature][value] = self._build_tree(sub_data, remaining_features, target, depth + 1)  # 递归构建子树
        
        return tree
    
    def _choose_best_feature(self, data, features, target):
        """
        选择具有最大信息增益的特征
        
        参数:
            data: 数据集
            features: 可用特征列表
            target: 目标变量名称
        
        返回:
            最佳特征名称
        """
        # 计算数据集的熵
        base_entropy = self._calculate_entropy(data[target])  # 计算当前数据集的熵
        
        # 初始化最大信息增益和对应的特征
        max_info_gain = -float('inf')  # 初始化为负无穷大
        best_feature = None  # 初始化最佳特征为空
        
        # 计算每个特征的信息增益
        for feature in features:
            # 计算特征的信息增益
            info_gain = self._calculate_info_gain(data, feature, target, base_entropy)  # 计算当前特征的信息增益
            
            # 更新最大信息增益和对应的特征
            if info_gain > max_info_gain:
                max_info_gain = info_gain  # 更新最大信息增益
                best_feature = feature  # 更新最佳特征
        
        return best_feature
    
    def _calculate_entropy(self, y):
        """
        计算熵
        
        参数:
            y: 目标变量
        
        返回:
            熵值
        """
        # 计算每个类别的概率
        counts = Counter(y)  # 统计各类别出现次数
        probs = [count / len(y) for count in counts.values()]  # 计算各类别概率
        
        # 计算熵
        entropy = -sum(p * math.log2(p) for p in probs)  # 应用熵的公式: -∑(p*log2(p))
        return entropy
    
    def _calculate_info_gain(self, data, feature, target, base_entropy):
        """
        计算特征的信息增益
        
        参数:
            data: 数据集
            feature: 特征名称
            target: 目标变量名称
            base_entropy: 数据集的熵
        
        返回:
            信息增益
        """
        # 获取特征的所有取值
        feature_values = data[feature].unique()  # 获取特征的所有可能取值
        
        # 计算条件熵
        conditional_entropy = 0
        for value in feature_values:
            # 获取特征值为value的子数据集
            sub_data = data[data[feature] == value]  # 按特征值分割数据
            
            # 计算子数据集的权重
            weight = len(sub_data) / len(data)  # 子数据集占总数据集的比例
            
            # 计算子数据集的熵
            sub_entropy = self._calculate_entropy(sub_data[target])  # 计算子数据集的熵
            
            # 累加条件熵
            conditional_entropy += weight * sub_entropy  # 加权累加各子集熵
        
        # 计算信息增益
        info_gain = base_entropy - conditional_entropy  # 信息增益 = 原熵 - 条件熵
        return info_gain
    
    def _calculate_tree_dimensions(self, tree, depth=0):
        """
        计算树的最大深度和宽度
        
        参数:
            tree: 决策树或子树
            depth: 当前深度
        
        返回:
            (最大深度, 最大宽度)
        """
        if not isinstance(tree, dict):
            return depth, 1
        
        feature = list(tree.keys())[0]
        max_depth = depth
        total_width = 0
        
        for value, subtree in tree[feature].items():
            sub_depth, sub_width = self._calculate_tree_dimensions(subtree, depth + 1)
            max_depth = max(max_depth, sub_depth)
            total_width += sub_width
        
        return max_depth, max(1, total_width)
    
    def print_tree(self):
        """
        打印决策树
        """
        if self.tree is None:
            print("模型尚未训练，请先调用fit方法")  # 检查模型是否已训练
            return
        
        self._print_tree(self.tree, 0)  # 从根节点开始打印，初始缩进级别为0
    
    def _print_tree(self, tree, level):
        """
        递归打印决策树
        
        参数:
            tree: 决策树或子树
            level: 当前层级
        """
        # 如果是叶节点，直接打印类别
        if not isinstance(tree, dict):
            print("  " * level + "-> " + str(tree))  # 打印叶节点的类别值
            return
        
        # 获取当前节点的特征
        feature = list(tree.keys())[0]  # 获取当前节点的决策特征
        print("  " * level + feature)  # 打印特征名称
        
        # 递归打印子树
        for value, subtree in tree[feature].items():
            print("  " * (level + 1) + str(value))  # 打印特征值
            self._print_tree(subtree, level + 2)  # 递归打印子树，增加缩进级别
    
    def visualize(self, output_file='decision_tree'):
        """
        将决策树可视化并保存为图片
        
        参数:
            output_file: 输出文件名（不包含扩展名）
        """
        if self.tree is None:
            print("模型尚未训练，请先调用fit方法")
            return
        
        try:
            # 使用matplotlib进行可视化
            import matplotlib.pyplot as plt
            
            # 设置中文字体支持
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            
            # 创建图形
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
            ax.axis('off')
            
            # 计算树的深度和宽度
            max_depth, max_width = self._calculate_tree_dimensions(self.tree)
            
            # 绘制树
            self._plot_tree(ax, self.tree, 0.5, 0.1, 0.9, 0.9, max_depth, plt=plt)
            
            # 保存图像
            plt.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{output_file}.pdf", bbox_inches='tight')
            
            print(f"决策树已保存为 {output_file}.png 和 {output_file}.pdf")
            
            # 显示图像
            plt.close()
            
        except ImportError as e:
            print(f"未找到matplotlib库或发生错误: {e}")
            print("使用备用方法生成文本形式的决策树...")
            self._save_text_tree(output_file)
        except Exception as e:
            print(f"可视化过程中发生错误: {e}")
            print("使用备用方法生成文本形式的决策树...")
            self._save_text_tree(output_file)
    
    def _plot_tree(self, ax, tree, x, y_top, x_width, y_height, max_depth, parent_x=None, parent_y=None, edge_label=None, plt=None):
        """
        递归绘制决策树
        
        参数:
            ax: matplotlib轴
            tree: 决策树或子树
            x: 当前节点的x坐标
            y_top: 顶部y坐标
            x_width: x方向的宽度
            y_height: y方向的高度
            max_depth: 树的最大深度
            parent_x: 父节点的x坐标
            parent_y: 父节点的y坐标
            edge_label: 边标签
            plt: matplotlib.pyplot模块
        """
        # 如果plt未传入，尝试导入
        if plt is None:
            import matplotlib.pyplot as plt
            
        # 计算当前节点的y坐标
        level_height = y_height / max_depth
        y = y_top - level_height / 2
        
        # 如果是叶节点
        if not isinstance(tree, dict):
            class_name = "是" if tree == 1 else "否"
            node_text = f'决策: {class_name} ({tree})'
            
            # 绘制节点
            rect = ax.add_patch(
                plt.Rectangle((x - 0.05, y - 0.02), 0.1, 0.04, 
                             linewidth=1, edgecolor='navy', facecolor='#AED6F1')
            )
            ax.text(x, y, node_text, ha='center', va='center', fontsize=9)
            
            # 如果有父节点，绘制连接线
            if parent_x is not None and parent_y is not None:
                ax.plot([parent_x, x], [parent_y, y - 0.02], 'k-', linewidth=1)
                
                # 添加边标签
                if edge_label:
                    mid_x = (parent_x + x) / 2
                    mid_y = (parent_y + y - 0.02) / 2
                    ax.text(mid_x, mid_y, edge_label, ha='center', va='center', 
                           fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            return
        
        # 获取当前节点的特征
        feature = list(tree.keys())[0]
        
        # 绘制当前节点
        rect = ax.add_patch(
            plt.Rectangle((x - 0.05, y - 0.02), 0.1, 0.04, 
                         linewidth=1, edgecolor='navy', facecolor='#E5F5FD')
        )
        ax.text(x, y, feature, ha='center', va='center', fontsize=10, weight='bold')
        
        # 如果有父节点，绘制连接线
        if parent_x is not None and parent_y is not None:
            ax.plot([parent_x, x], [parent_y, y - 0.02], 'k-', linewidth=1)
            
            # 添加边标签
            if edge_label:
                mid_x = (parent_x + x) / 2
                mid_y = (parent_y + y - 0.02) / 2
                ax.text(mid_x, mid_y, edge_label, ha='center', va='center', 
                       fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # 计算子树数量
        items = list(tree[feature].items())
        n_children = len(items)
        
        # 计算子树的x坐标间隔
        if n_children > 1:
            dx = x_width / n_children
        else:
            dx = x_width / 2
        
        # 递归绘制子树
        for i, (value, subtree) in enumerate(items):
            # 计算子树的x坐标
            child_x = x - x_width/2 + dx/2 + i*dx
            
            # 为特征值添加文字说明
            value_text = self._get_feature_value_text(feature, value)
            
            # 递归绘制子树
            self._plot_tree(ax, subtree, child_x, y_top - level_height, 
                          dx, y_height, max_depth, x, y, value_text, plt=plt)
    
    def _save_text_tree(self, output_file):
        """
        将决策树保存为文本文件
        
        参数:
            output_file: 输出文件名（不包含扩展名）
        """
        import io
        
        # 使用StringIO捕获打印输出
        buffer = io.StringIO()
        
        # 打印树结构到缓冲区
        buffer.write("=" * 50 + "\n")
        buffer.write("决策树结构:\n")
        buffer.write("=" * 50 + "\n")
        self._print_text_tree(self.tree, 0, buffer)
        buffer.write("=" * 50 + "\n")
        
        # 将缓冲区内容写入文件
        with open(f"{output_file}.txt", "w", encoding="utf-8") as f:
            f.write(buffer.getvalue())
        
        print(f"决策树已保存为文本文件: {output_file}.txt")
        
        # 关闭缓冲区
        buffer.close()
    
    def _print_text_tree(self, tree, level, buffer):
        """
        递归打印决策树到缓冲区
        
        参数:
            tree: 决策树或子树
            level: 当前层级
            buffer: 输出缓冲区
        """
        # 如果是叶节点，直接打印类别
        if not isinstance(tree, dict):
            class_name = "是" if tree == 1 else "否"
            buffer.write("  " * level + "└─ 决策: " + class_name + f" ({tree})\n")
            return
        
        # 获取当前节点的特征
        feature = list(tree.keys())[0]
        buffer.write("  " * level + f"【{feature}】\n")
        
        # 递归打印子树
        items = list(tree[feature].items())
        for i, (value, subtree) in enumerate(items):
            # 为特征值添加文字说明
            value_text = self._get_feature_value_text(feature, value)
            is_last = (i == len(items) - 1)
            prefix = "└─ " if is_last else "├─ "
            buffer.write("  " * (level + 1) + prefix + value_text + "\n")
            self._print_text_tree(subtree, level + 2, buffer)

    def _create_dot_data(self, tree, parent_node=None, edge_label=None, node_id=None):
        """
        递归创建DOT语言数据
        
        参数:
            tree: 决策树或子树
            parent_node: 父节点ID
            edge_label: 边标签
            node_id: 当前节点ID
        
        返回:
            DOT语言字符串
        """
        if node_id is None:
            node_id = [0]  # 使用列表作为可变对象
            dot_data = ['digraph Tree {',
                       'node [shape=box, fontname="Microsoft YaHei", fontsize=10, color="skyblue", style="filled", fillcolor="#E5F5FD"];',
                       'edge [fontname="Microsoft YaHei", fontsize=9, color="navy"];']
        else:
            dot_data = []
        
        # 如果是叶节点
        if not isinstance(tree, dict):
            class_name = "是" if tree == 1 else "否"
            node_label = f'决策: {class_name} ({tree})'
            dot_data.append(f'{node_id[0]} [label="{node_label}", shape=box, fillcolor="#AED6F1"];')
            
            if parent_node is not None and edge_label is not None:
                dot_data.append(f'{parent_node} -> {node_id[0]} [label="{edge_label}"];')
            
            node_id[0] += 1
            return dot_data
        
        # 获取当前节点的特征
        feature = list(tree.keys())[0]
        
        # 创建当前节点
        current_node = node_id[0]
        dot_data.append(f'{current_node} [label="{feature}"];')
        node_id[0] += 1
        
        # 如果有父节点，添加边
        if parent_node is not None and edge_label is not None:
            dot_data.append(f'{parent_node} -> {current_node} [label="{edge_label}"];')
        
        # 递归处理子树
        for value, subtree in tree[feature].items():
            # 为特征值添加文字说明
            value_text = self._get_feature_value_text(feature, value)
            
            # 递归创建子树
            child_dot_data = self._create_dot_data(subtree, current_node, value_text, node_id)
            dot_data.extend(child_dot_data)
        
        # 如果是根节点，添加结束括号
        if parent_node is None:
            dot_data.append('}')
        
        return dot_data
    
    def _get_feature_value_text(self, feature, value):
        """
        获取特征值的文字说明
        
        参数:
            feature: 特征名称
            value: 特征值
        
        返回:
            特征值的文字说明
        """
        feature_value_map = {
            "天气": {0: "晴朗", 1: "多云", 2: "雨天"},
            "温度": {0: "高", 1: "中", 2: "低"},
            "湿度": {0: "高", 1: "正常"},
            "风力": {0: "强", 1: "弱"}
        }
        
        if feature in feature_value_map and value in feature_value_map[feature]:
            return feature_value_map[feature][value]
        return f"{value}"

# 在主程序部分修改可视化调用
if __name__ == "__main__":
    # 创建一个简单的数据集
    # 天气: 0-晴朗, 1-多云, 2-雨天
    # 温度: 0-高, 1-中, 2-低
    # 湿度: 0-高, 1-正常
    # 风力: 0-强, 1-弱
    # 是否打网球: 0-否, 1-是
    X = np.array([
        [0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [2, 1, 0, 0],
        [2, 2, 1, 0], [2, 2, 1, 1], [1, 2, 1, 1], [0, 1, 0, 0],
        [0, 2, 1, 0], [2, 1, 1, 0], [0, 1, 1, 1], [1, 1, 0, 1],
        [1, 0, 1, 0], [2, 1, 0, 1]
    ])  # 特征矩阵，每行是一个样本，每列是一个特征
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])  # 目标变量，是否打网球
    
    # 特征名称
    feature_names = ["天气", "温度", "湿度", "风力"]  # 定义特征名称
    
    # 创建输出文件夹
    import os
    output_folder = r"c:\programme\data_science\artificial intelligence\homwork_ex\hw\hw02\design_output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 创建并训练ID3决策树
    tree = ID3DecisionTree(max_depth=3)  # 创建决策树，最大深度为3
    tree.fit(X, y, feature_names)  # 训练决策树
    
    # 打印决策树
    print("ID3决策树结构:")
    tree.print_tree()  # 可视化决策树结构
    
    # 进行预测
    X_test = np.array([[0, 1, 0, 1], [2, 2, 1, 0]])  # 测试数据
    predictions = tree.predict(X_test)  # 预测测试数据的类别
    print("\n预测结果:", predictions)  # 打印预测结果
    
    # 可视化决策树
    output_path = os.path.join(output_folder, "decision_tree")
    tree.visualize(output_path)
    
    # 添加提示信息
    print("\n如果要使用matplotlib可视化，请确保已安装matplotlib库")
    print("如果未安装，可以使用以下命令安装：pip install matplotlib")