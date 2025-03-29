# 🤖 机器学习

本项目包含机器学习的实验代码和示例，主要关注数据可视化、线性回归模型和分类模型的应用。

## 📚 主要内容

### hw01 内容

#### ex2.1 - 身高预测模型

这个实验展示了如何使用线性回归模型根据足长和步幅预测身高。

**主要功能**：
- 使用足长和步幅作为特征，身高作为目标变量
- 实现二元线性回归模型
- 分析模型参数（权重和截距）
- 可视化特征与目标变量之间的关系
- 通过3D图展示多变量关系

**技术要点**：
- 使用scikit-learn实现线性回归
- 使用matplotlib进行数据可视化
- 包含单变量和多变量分析

#### ch02-5-5 - 高级数据可视化技术

这个文件展示了多种高级数据可视化技术，适用于不同类型的数据分析场景。

**主要功能**：
- 散点图、气泡图和箱线图的创建
- 直方图和概率密度函数的可视化
- 平行坐标系图表的实现
- 股票数据的动态可视化
- 自定义平行坐标图的实现

**技术要点**：
- 使用matplotlib和seaborn进行基础可视化
- 使用pyecharts创建交互式图表
- 使用gif模块创建动态可视化
- 股票数据的获取和处理
- 自定义可视化函数的实现

### hw02 内容

#### 3-2 - 集成学习方法

这个实验展示了如何使用决策树和装袋法（Bagging）进行分类任务。

**主要功能**：
- 使用决策树分类器（CART）进行基础分类
- 实现装袋法（Bagging）提升分类性能
- 使用交叉验证评估模型性能
- 比较基础分类器和集成方法的性能差异

**技术要点**：
- 使用scikit-learn实现决策树分类器
- 使用BaggingClassifier实现集成学习
- 使用K折交叉验证评估模型
- 参数调优与性能分析

## 🛠️ 环境要求

本项目需要以下Python库：
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- pyecharts
- PIL/Pillow
- gif

项目根目录下提供了`requirements.txt`文件，其中包含了所有依赖库的推荐版本。您可以使用以下命令一键安装所有依赖：

```bash
pip install -r requirements.txt
```

## 📝 使用方法

1. 确保已安装所有必要的依赖库
2. 使用Jupyter Notebook或者python编译器（vs code 或 PyCharm）打开.ipynb文件
3. 按顺序执行单元格以查看结果

## 📂 数据文件

### hw01 数据
- 身高预测参照表-1.xlsx：包含足长、步幅和身高数据
- 其他数据通过API动态获取（如股票数据）

### hw02 数据
- 分类数据集：通过scikit-learn的datasets模块加载

## ⚠️ 注意事项

- 部分可视化需要特定的库支持
- 动态可视化可能需要较长时间生成
- 确保文件路径正确，特别是在读取Excel文件时
- 对于hw02中的集成学习，注意scikit-learn版本兼容性（1.0及以上版本使用estimator参数代替base_estimator）