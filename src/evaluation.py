import numpy as np
import pandas as pd


def get_label(y, X, time_delay):  # y可为y_test或y_pred，Day为一个数[5,10,20,40,60]
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
