import numpy as np


# TODO: Deal with both the continnuous and balanced
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


# TODO: Split into several functions to make better use
def calculate_f_beta_multiclass(true_labels, pred_labels, beta=0.5):
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


# TODO: Insight into data, including especially numbers of different labels
def get_data_count():
    pass
