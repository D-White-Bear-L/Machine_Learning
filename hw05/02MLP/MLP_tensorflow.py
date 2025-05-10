import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris # 加载鸢尾花数据
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建输出目录
output_folder = r'hw05\02MLP\MLP_output'
os.makedirs(output_folder, exist_ok=True)

# 设置随机种子以确保结果可重现
np.random.seed(42)
tf.random.set_seed(42)

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data.astype(np.float32)
y = iris.target.reshape(-1, 1)

# 对标签进行独热编码
try:
    # 尝试使用新参数名
    encoder = OneHotEncoder(sparse_output=False)
except TypeError:
    # 如果失败，使用旧参数名
    encoder = OneHotEncoder(sparse=False)
y_one_hot = encoder.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# 定义MLP模型
def create_mlp_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 创建模型
model = create_mlp_model()

# 打印模型结构
model.summary()

# 训练模型
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"测试集准确率: {accuracy:.4f}")

# 可视化训练过程
plt.figure(figsize=(12, 4))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('模型准确率')
plt.xlabel('轮次')
plt.ylabel('准确率')
plt.legend() # 绘制训练损失和验证损失曲线

plt.tight_layout()
plt.savefig(os.path.join(output_folder,'iris_mlp_training.png')) # 保存混淆矩阵为图片
plt.show()

# 使用模型进行预测
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# 打印分类报告
from sklearn.metrics import classification_report, confusion_matrix
print("\n分类报告:")
print(classification_report(true_classes, predicted_classes, target_names=iris.target_names))

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(true_classes, predicted_classes)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.colorbar()
tick_marks = np.arange(len(iris.target_names))
plt.xticks(tick_marks, iris.target_names, rotation=45)
plt.yticks(tick_marks, iris.target_names)

# 在混淆矩阵中添加文本标注
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


plt.tight_layout()
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.savefig(os.path.join(output_folder,'iris_mlp_confusion_matrix.png')) # 保存混淆矩阵为图片
plt.show()

# 保存模型
model.save(os.path.join(output_folder,'iris_mlp_model.h5'))
print("模型已保存为h5格式 'iris_mlp_model.h5'")