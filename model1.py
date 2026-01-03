import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
import os


# 1. 加载数据
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# 2. 数据预处理
def preprocess_data(df, sequence_length=30):
    # 选择特征列（去掉标签和时间相关列）
    feature_columns = [
        "n_close",
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

    # 标签列（这里以label5为例）
    label_column = "label_5"

    # 处理缺失值（用前一个值填充）
    df[feature_columns] = df[feature_columns].fillna(method="ffill")

    # 标准化特征
    # scaler = StandardScaler()
    # scaled_features = scaler.fit_transform(df[feature_columns])
    scaled_features = df[feature_columns]
    # 创建时间序列样本
    X, y = [], []
    for i in range(len(scaled_features) - sequence_length):
        X.append(scaled_features[i : i + sequence_length])
        y.append(df.iloc[i + sequence_length][label_column])

    return np.array(X), np.array(y)


# 3. 构建LSTM模型
def build_lstm_model(input_shape, num_classes=3):
    model = Sequential(
        [
            # Conv1D(filters=128, kernel_size=3, padding="same", activation="tanh"),
            # Dropout(0.3),
            LSTM(
                256,
                activation="tanh",
                return_sequences=True,
                input_shape=input_shape,
            ),
            Dropout(0.3),
            # LSTM(
            #     256,
            #     activation="tanh",
            #     return_sequences=True,
            # ),
            # Dropout(0.3),
            LSTM(256, activation="tanh", return_sequences=False),
            Dropout(0.3),
            Dense(128, activation="tanh", kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(64, activation="tanh"),
            Dense(num_classes, activation="softmax"),  # 3个类别：0,1,2
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def apply_undersampling(X_train, y_train):
    """对多数类进行欠采样"""
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)

    sampler = RandomUnderSampler(random_state=42, sampling_strategy="auto")
    X_resampled, y_resampled = sampler.fit_resample(X_train_reshaped, y_train)

    # 将数据reshape回原来的形状
    X_resampled = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])

    print(f"欠采样后训练集形状: {X_resampled.shape}, {y_resampled.shape}")
    print(f"类别分布: {np.bincount(y_resampled.astype(int))}")
    return X_resampled, y_resampled


# 4. 主函数
def main():
    file_path = "./merged_data/merged_0.csv"

    # 加载数据
    df = load_data(file_path)
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print(df.head())

    # 预处理数据
    sequence_length = 100  # 使用过去30个tick的数据
    X, y = preprocess_data(df, sequence_length)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # 时间序列不随机打乱
    )

    print(f"训练集形状: {X_train.shape}, {y_train.shape}")
    print(f"测试集形状: {X_test.shape}, {y_test.shape}")

    # 构建模型
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    model.summary()

    # 训练模型
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=5,
        batch_size=128,
        class_weight={0: 5, 1: 1, 2: 5},
        verbose=1,
    )

    # 预测示例
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    test_score = calculate_f_beta_multiclass(y_test, y_pred)
    print(f"The f beta score on test: {test_score}")

    # 或许是无用功，但还是应该试一试：
    y_pred = model.predict(X_train)
    y_pred = np.argmax(y_pred, axis=1)
    train_score = calculate_f_beta_multiclass(y_train, y_pred)
    print(f"The f beta score on train: {train_score}")

    # 保存模型
    # model.save('lstm_price_prediction_model.h5')
    # print("模型已保存为 'lstm_price_prediction_model.h5'")


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
