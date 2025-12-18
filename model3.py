import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D

from tensorflow.keras.metrics import Accuracy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import os


# 1. 加载数据
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# 2. 数据预处理
def preprocess_data(df, sequence_length=30, time_delay=5):
    # 选择特征列（去掉标签和时间相关列）
    feature_columns = [
        # "n_close",
        # "amount_delta",
        "n_midprice",
        # "n_bid1",
        # "n_bsize1",
        # "n_bid2",
        # "n_bsize2",
        # "n_bid3",
        # "n_bsize3",
        # "n_bid4",
        # "n_bsize4",
        # "n_bid5",
        # "n_bsize5",
        # "n_ask1",
        # "n_asize1",
        # "n_ask2",
        # "n_asize2",
        # "n_ask3",
        # "n_asize3",
        # "n_ask4",
        # "n_asize4",
        # "n_ask5",
        # "n_asize5",
    ]

    # 处理缺失值（用前一个值填充）
    df[feature_columns] = df[feature_columns].fillna(method="ffill")

    # 创建时间序列样本
    features = df[feature_columns]
    X, y = [], []
    for i in range(len(features) - sequence_length - time_delay):
        X.append(features[i : i + sequence_length])
        y.append(df.iloc[i + sequence_length + time_delay - 1]["n_midprice"])

    return np.array(X), np.array(y)


# 3. 构建LSTM模型
def build_lstm_model(input_shape):
    model = Sequential(
        [
            # Conv1D(filters=64, kernel_size=3, activation="tanh"),
            # Dropout(0.3),
            # Conv1D(filters=64, kernel_size=3, activation="tanh"),
            # Dropout(0.3),
            LSTM(256, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(256, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation="tanh"),
            Dropout(0.3),
            Dense(1, activation="tanh"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mse"],
    )

    return model


# 4. 主函数
def main():
    file_path = "./merged_data/merged_0.csv"

    # 加载数据
    df = load_data(file_path)
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print(df.head())

    # Visualize the data
    # Plot the closing price over time
    plt.figure(figsize=(14, 7))
    plt.plot(df["n_midprice"], label="Midprice")
    plt.title("Midprice Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    plt.show()

    # 预处理数据
    sequence_length = 100  # 使用过去30个tick的数据
    print("Preprocessing Data...")
    X, y = preprocess_data(df, sequence_length, 5)
    print("Preprocessing Data...Done")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # 时间序列不随机打乱
    )

    print(f"训练集形状: {X_train.shape}, {y_train.shape}")
    print(f"测试集形状: {X_test.shape}, {y_test.shape}")

    # 构建模型
    print("Building model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    print("Building model...Done")
    model.summary()

    # 训练模型
    history = model.fit(
        X_train,
        y_train,
        # validation_data=(X_test, y_test),
        epochs=10,
        batch_size=128,
        verbose=1,
    )

    # 预测示例
    y_pred = model.predict(X_test)
    print("Plotting test results...")
    plot_predict_curve(y_test, y_pred)
    draw_loss_curve(history=history)

    # Draw loss curve
    # draw_loss_curve(history)

    # 将 TensorFlow 张量转换为 NumPy 数组
    X_np = X_test[:, 99, 0]

    # 同样，需要 squeeze y_pred_np
    y_pred_np_squeezed = np.squeeze(y_pred)
    y_test_squeezed = np.squeeze(y_test)

    print("Shape of y_pred_np_squeezed:", y_pred_np_squeezed.shape)
    print("Calculating labels with NumPy...")
    pred_labels = change_label(y_pred_np_squeezed, X_np, 5)
    true_labels = change_label(y_test_squeezed, X_np, 5)
    print("Calculating labels...Done")

    # m = Accuracy()
    # m.update_state(df["label_5"], y_pred)
    print("Calculating scores...")
    score = calculate_f_beta_multiclass(true_labels, pred_labels)
    print("Calculating scores...Done")
    print(f"The F-beta Score: {score}")


def change_label(y, X, time_delay):  # y可为y_test或y_pred，Day为一个数[5,10,20,40,60]
    # X = X_test[:, 99, 0]
    if time_delay in [5, 10]:
        alpha = 0.0005
    elif time_delay in [20, 40, 60]:
        alpha = 0.001
    else:
        # 如果N不在预期值中，使用默认值或抛出异常
        raise ValueError(
            f"不支持的时间步长N={time_delay}，支持的值为[5, 10, 20, 40, 60]"
        )

    y = np.asarray(y)
    X = np.asarray(X)

    price_diff = y - X

    # 使用 np.select 进行高效的条件选择
    conditions = [price_diff < -alpha, price_diff > alpha]
    choices = [0, 2]
    labels = np.select(conditions, choices, default=1)

    return labels.tolist()


def calculate_f_beta_multiclass(true_labels, pred_labels, beta=0.5):
    # 确保输入是numpy数组
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # 计算recall：真实标签不是1的样本中预测正确的比例
    non_one_mask = true_labels != 1
    if np.sum(non_one_mask) > 0:
        recall = np.sum(
            true_labels[non_one_mask] == pred_labels[non_one_mask]
        ) / np.sum(non_one_mask)
    else:
        recall = 0.0

    # 计算precision：预测标签不是1的样本中预测正确的比例
    pred_non_one_mask = pred_labels != 1
    if np.sum(pred_non_one_mask) > 0:
        precision = np.sum(
            true_labels[pred_non_one_mask] == pred_labels[pred_non_one_mask]
        ) / np.sum(pred_non_one_mask)
    else:
        precision = 0.0

    # 计算F-beta分数
    if precision + recall == 0:
        f_beta = 0.0
    else:
        numerator = (1 + beta**2) * precision * recall
        denominator = (beta**2 * precision) + recall
        f_beta = numerator / denominator

    return f_beta


def plot_predict_curve(y_true, y_pred):
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label="True Curve")
    plt.plot(y_pred, label="Prediction Curve")
    plt.title("Prediction v.s. Truth")
    plt.xlabel("Ticks")
    plt.ylabel("Midprice")
    plt.legend()
    plt.grid(True)

    plt.show()


def draw_loss_curve(history):
    history_dict = history.history
    # 创建图表
    plt.figure(figsize=(15, 5))

    # 1. 绘制损失曲线
    plt.plot(history_dict["loss"], label="Training Loss")
    if "val_loss" in history_dict:
        plt.plot(history_dict["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


# 5. 批量处理多个文件
def process_multiple_files(file_patterns):
    all_X, all_y = [], []

    for file_path in file_patterns:
        if os.path.exists(file_path):
            df = load_data(file_path)
            X, y = preprocess_data(df)
            all_X.append(X)
            all_y.append(y)

    # 合并所有数据
    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)

    return X_combined, y_combined


if __name__ == "__main__":
    # 运行主函数
    main()

    # 如果需要处理多个文件
    # files = ['snapshot_sym1_date1_am.csv', 'snapshot_sym1_date1_pm.csv']
    # X_all, y_all = process_multiple_files(files)
