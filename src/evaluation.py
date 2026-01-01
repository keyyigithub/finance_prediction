import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import alpha
import tensorflow as tf


class FBetaScore(tf.keras.metrics.Metric):
    def __init__(self, beta=0.5, name="f_beta_score", **kwargs):
        super(FBetaScore, self).__init__(name=name, **kwargs)
        self.beta = beta
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 将预测转换为类别
        y_pred_labels = tf.argmax(y_pred, axis=-1)  # 如果是多分类
        # 或者 y_pred_labels = tf.round(y_pred) 如果是二分类

        y_true = tf.cast(y_true, tf.int32)
        y_pred_labels = tf.cast(y_pred_labels, tf.int32)

        # 创建掩码
        true_non_one_mask = tf.not_equal(y_true, 1)
        pred_non_one_mask = tf.not_equal(y_pred_labels, 1)

        # 计算真正例（true_positives）：预测为非1且正确的样本
        correct_predictions = tf.equal(y_true, y_pred_labels)
        true_positives = tf.logical_and(pred_non_one_mask, correct_predictions)
        true_positives = tf.cast(true_positives, tf.float32)

        # 计算假正例（false_positives）：预测为非1但错误的样本
        false_positives = tf.logical_and(
            pred_non_one_mask, tf.logical_not(correct_predictions)
        )
        false_positives = tf.cast(false_positives, tf.float32)

        # 计算假负例（false_negatives）：真实为非1但预测为1的样本
        false_negatives = tf.logical_and(true_non_one_mask, tf.equal(y_pred_labels, 1))
        false_negatives = tf.cast(false_negatives, tf.float32)

        # 更新状态变量
        self.true_positives.assign_add(tf.reduce_sum(true_positives))
        self.false_positives.assign_add(tf.reduce_sum(false_positives))
        self.false_negatives.assign_add(tf.reduce_sum(false_negatives))

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )

        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )

        # 计算F-beta分数
        numerator = (1 + self.beta**2) * precision * recall
        denominator = (self.beta**2 * precision) + recall

        f_beta = tf.where(tf.equal(denominator, 0), 0.0, numerator / denominator)

        return f_beta

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)


def get_label(
    y, X, time_delay, alpha1=0.0005, alpha2=0.001
):  # y可为y_test或y_pred，Day为一个数[5,10,20,40,60]
    # X = X_test[:, 99, 0]
    if time_delay in [5, 10]:
        alpha = alpha1
    elif time_delay in [20, 40, 60]:
        alpha = alpha2
    else:
        # 如果N不在预期值中，使用默认值或抛出异常
        raise ValueError(
            f"不支持的时间步长N={time_delay}，支持的值为[5, 10, 20, 40, 60]"
        )

    y = np.asarray(y)
    X = np.asarray(X)

    # Squeeze to prevent broadcast error
    y = np.squeeze(y)
    price_diff = y - X

    # 使用 np.select 进行高效的条件选择
    conditions = [price_diff < -alpha, price_diff > alpha]
    choices = [0, 2]
    labels = np.select(conditions, choices, default=1)

    return labels.tolist()


def calculate_f_beta_multiclass(true_labels, pred_labels, beta=0.5):
    # 计算recall：真实标签不是1的样本中预测正确的比例
    non_one_mask = true_labels != 1
    if np.sum(non_one_mask) > 0:
        recall = np.sum(
            true_labels[non_one_mask] == pred_labels[non_one_mask]
        ) / np.sum(non_one_mask)
    else:
        recall = 10.0

    print(f"Recall: {recall}")

    # 计算precision：预测标签不是1的样本中预测正确的比例
    pred_non_one_mask = pred_labels != 1
    if np.sum(pred_non_one_mask) > 0:
        precision = np.sum(
            true_labels[pred_non_one_mask] == pred_labels[pred_non_one_mask]
        ) / np.sum(pred_non_one_mask)
    else:
        precision = 10.0

    print(f"Precision: {precision}")
    # 计算F-beta分数
    if precision + recall == 0:
        f_beta = 0.0
    else:
        numerator = (1 + beta**2) * precision * recall
        denominator = (beta**2 * precision) + recall
        f_beta = numerator / denominator

    return f_beta


def check_feature_distributions(df: pd.DataFrame, features: list[str]):
    """
    检查特征分布，找出问题特征
    """
    results = []
    for feature in features:
        data = df[feature].values
        stats = {
            "feature": feature,
            "min": np.nanmin(data),
            "max": np.nanmax(data),
            "mean": np.nanmean(data),
            "std": np.nanstd(data),
            "has_nan": np.any(np.isnan(data)),
            "has_inf": np.any(np.isinf(data)),
            "negative_count": np.sum(data < 0),
            "zero_count": np.sum(data == 0),
            "less_than_minus1": np.sum(data < -1) if np.any(data < -1) else 0,
        }
        results.append(stats)

    return pd.DataFrame(results)


# converts the one-hot coding to labels, with threshold
# shape of y: (n_samples, 3)
def triple_one_hot_to_label(y: NDArray, threshold: float):
    y0, y1, y2 = y[:, 0], y[:, 1], y[:, 2]
    # 创建条件数组
    cond1 = y0 - y2 > threshold  # 取0的条件
    cond2 = y2 - y0 > threshold  # 取2的条件
    y1_is_max = (y1 >= y0) & (y1 >= y2)

    # 初始化结果数组，默认值为1
    result = np.ones(y.shape[0], dtype=int)

    # 对于y[1]不是最大值的情况
    mask = ~y1_is_max
    y0_masked = y0[mask]
    y2_masked = y2[mask]

    # 应用条件
    cond1 = y0_masked - y2_masked > threshold  # 取0的条件
    cond2 = y2_masked - y0_masked > threshold  # 取2的条件

    # 使用np.select进行条件选择
    subset_result = np.select([cond1, cond2], [0, 2], default=1)
    result[mask] = subset_result

    return result


def double_one_hot_to_label(y: NDArray, threshold=0.001):
    y0 = y[:, 0]
    y1 = y[:, 1]

    # 创建条件数组
    cond1 = y0 - y1 > threshold  # 取0的条件
    cond2 = y1 - y0 > threshold  # 取2的条件

    # 使用np.select进行条件选择
    return np.select([cond1, cond2], [0, 2], default=1)


def adaptive_threshold_double_one_hot_to_label(y: NDArray, adaptive_percentile=0.3):
    """
    自适应阈值方法：根据预测分布的百分位数确定阈值
    """
    y0 = y[:, 0]
    y1 = y[:, 1]

    # 计算预测置信度的差异
    conf_diff = np.abs(y0 - y1)

    # 使用百分位数作为阈值（例如，差异小于30%分位数的视为不确定）
    threshold = np.percentile(conf_diff, adaptive_percentile * 100)

    # 创建条件数组
    cond1 = y0 - y1 > threshold  # 取0的条件
    cond2 = y1 - y0 > threshold  # 取2的条件

    return np.select([cond1, cond2], [0, 2], default=1)


def label_to_double_one_hot(labels: NDArray):
    labels = np.asarray(labels)

    # 定义编码映射
    encoding_map = np.array(
        [[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]],  # label 0  # label 1  # label 2
        dtype=np.float32,
    )

    # 验证标签范围
    if np.any((labels < 0) | (labels > 2)):
        raise ValueError("标签值必须在[0, 2]范围内")

    # 直接使用索引获取编码
    # 注意：这种方法要求labels是整数，且范围在0-2
    return encoding_map[labels.astype(int)]


def get_uncertainty(y_pred_alpha):
    S = np.sum(y_pred_alpha, axis=1, keepdims=True)
    probs = y_pred_alpha / S
    uncertainty = 2 / S
    return probs, uncertainty


def get_label_with_uncertainty(y_pred_alpha, threshold=1.0):
    _, uncertainty = get_uncertainty(y_pred_alpha)
    labels = np.select(y_pred_alpha[:, 0] > y_pred_alpha[:, 1], [0], default=2)
    uncredible_mask = uncertainty > threshold
    labels[uncredible_mask] = 1
    return labels


def calculate_pnl_average(df: pd.DataFrame, pred_labels: NDArray, time_delay: int):
    returns = df[f"return_after_{time_delay}"].values
    non_one_mask = pred_labels != 1
    pred_labels = pred_labels - 1
    # 方法1：使用 np.nansum 忽略 NaN（推荐）
    selected_returns = returns[(len(returns) - len(pred_labels)) :]

    acc_return = np.nansum(pred_labels * selected_returns)
    if sum(non_one_mask) > 0:
        average_return = acc_return / sum(non_one_mask)
    else:
        print("Why no 0 or 2? Fuck you sonuvbitch!")
        average_return = 114514

    return average_return
