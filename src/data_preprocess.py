from numpy.typing import NDArray
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import joblib
import pandas as pd
import numpy as np

selected_features = [
    # 1. 核心价格特征（2个）
    "n_close",  # 标准化后的收盘价
    "n_midprice",  # 标准化后的中间价
    # 2. 市场微观结构（5个）
    "bid_ask_spread",  # 买卖价差
    "size_imbalance_1",  # 一档买卖量不平衡
    "microprice",  # 微观价格（考虑深度的加权价格）
    "size_imbalance_5",
    # "order_flow_imbalance",  # 订单流不平衡
    "total_depth",  # 总市场深度
    # 3. 动量与趋势（4个）
    "midprice_momentum_20",  # 20期动量
    "macd",  # MACD线
    "ma_cross_5_20",  # 移动平均线交叉信号
    "price_acceleration",  # 价格加速度
    # 4. 波动率特征（3个）
    "volatility_20",  # 20期波动率
    "bollinger_width",  # 布林带宽度
    "parkinson_vol_20",  # Parkinson波动率（更准确的高低价估计）
    # 5. 技术指标（3个）
    "rsi_14",  # 14期RSI
    "stochastic_k",  # 随机指标K值
    "bias_20",  # 20期乖离率
    # 6. 成交量与流动性（2个）
    "amount_delta",  # 成交额变化
    "volume_momentum",  # 成交量动量
    # 7. 时间特征（1个）
    "time_sin",  # 时间正弦编码
    "sym",
]


def create_all_features(df: pd.DataFrame):
    """创建所有特征"""
    print("开始特征工程...")

    df["n_midprice"] = df["n_midprice"] + 1
    df["n_close"] = df["n_close"] + 1
    df["n_bid1"] = df["n_bid1"] + 1
    df["n_bid2"] = df["n_bid2"] + 1
    df["n_bid3"] = df["n_bid3"] + 1
    df["n_bid4"] = df["n_bid4"] + 1
    df["n_bid5"] = df["n_bid5"] + 1
    df["n_ask1"] = df["n_ask1"] + 1
    df["n_ask2"] = df["n_ask2"] + 1
    df["n_ask3"] = df["n_ask3"] + 1
    df["n_ask4"] = df["n_ask4"] + 1
    df["n_ask5"] = df["n_ask5"] + 1
    time_series = pd.to_datetime(df["time"], format="%H:%M:%S")

    df["time_sin"] = np.sin(
        2
        * np.pi
        * (
            time_series.dt.hour * 3600
            + time_series.dt.minute * 60
            + time_series.dt.second
        )
        / 86400
    )

    df["bid_ask_spread"] = df["n_ask1"] - df["n_bid1"]

    df["bid_depth"] = sum([df[f"n_bsize{i}"] for i in range(1, 6)])
    df["ask_depth"] = sum([df[f"n_asize{i}"] for i in range(1, 6)])
    df["total_depth"] = df["bid_depth"] + df["ask_depth"]
    df["depth_imbalance"] = (df["bid_depth"] - df["ask_depth"]) / df["total_depth"]
    for depth in [1, 5]:
        df[f"size_imbalance_{depth}"] = (
            df[f"n_bsize{depth}"] - df[f"n_asize{depth}"]
        ) / (df[f"n_bsize{depth}"] + df[f"n_asize{depth}"])
    total_bid_size = sum([df[f"n_bsize{i}"] for i in range(1, 6)])
    total_ask_size = sum([df[f"n_asize{i}"] for i in range(1, 6)])
    df["microprice"] = (
        df["n_bid1"] * total_ask_size + df["n_ask1"] * total_bid_size
    ) / (total_bid_size + total_ask_size)

    df[f"midprice_momentum_20"] = df["n_midprice"] - df["n_midprice"].shift(20)

    exp1 = df["n_midprice"].ewm(span=12, adjust=False).mean()
    exp2 = df["n_midprice"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2

    df["price_acceleration"] = (df["n_midprice"] - df["n_midprice"].shift(1)) - (
        df["n_midprice"] - df["n_midprice"].shift(1)
    ).shift(1)

    ma_periods = [5, 20]
    for period in ma_periods:
        df[f"ma_{period}"] = df["n_midprice"].rolling(window=period).mean()
        df[f"price_vs_ma_{period}"] = df["n_midprice"] / df[f"ma_{period}"] - 1

    # 4. 移动平均线交叉
    df["ma_cross_5_20"] = df["ma_5"] - df["ma_20"]
    df["bias_20"] = (df["n_midprice"] - df["ma_20"]) / df["ma_20"] * 100

    rolling_window = 50
    df["midprice_ma"] = df["n_midprice"].rolling(window=rolling_window).mean()
    df["midprice_std"] = df["n_midprice"].rolling(window=rolling_window).std()
    df["bollinger_upper"] = df["midprice_ma"] + 2 * df["midprice_std"]
    df["bollinger_lower"] = df["midprice_ma"] - 2 * df["midprice_std"]
    df["bollinger_width"] = (df["bollinger_upper"] - df["bollinger_lower"]) / df[
        "midprice_ma"
    ]

    delta = df["n_midprice"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df["rsi_14"] = np.where((gain + loss) != 0, 1 - (loss / (gain + loss)), 0.5)

    low_min = df["n_midprice"].rolling(window=14).min()
    high_max = df["n_midprice"].rolling(window=14).max()
    price_diff = df["n_midprice"] - low_min
    range_diff = high_max - low_min
    df["stochastic_k"] = np.where(range_diff != 0, price_diff / range_diff, 0.5)

    df["volume_momentum"] = df["amount_delta"] - df["amount_delta"].shift(1)

    df["midprice_return"] = df["n_midprice"].pct_change()
    df["volatility_20"] = df["midprice_return"].rolling(window=20).std() * np.sqrt(20)
    high_low_ratio = (
        np.log(
            df["n_midprice"].rolling(window=2).max()
            / df["n_midprice"].rolling(window=2).min()
        )
        ** 2
    )
    df["parkinson_vol_20"] = np.sqrt(
        (1 / (4 * 20 * np.log(2))) * high_low_ratio.rolling(window=20).sum()
    )
    df["amount_delta"] = df["amount_delta"].apply(
        lambda x: np.sign(x) * np.log1p(np.abs(x))
    )
    df["volume_momentum"] = df["volume_momentum"].apply(
        lambda x: np.sign(x) * np.log1p(np.abs(x))
    )

    print(f"特征工程完成")

    return df


def split(X, y, test_size=0.2):
    n_samples = X.shape[0]
    split_idx = int(n_samples * (1 - test_size))
    X_train = X[:split_idx, :, :]
    y_train = y[:split_idx]
    X_test = X[split_idx:, :, :]
    y_test = y[split_idx:]

    return X_train, X_test, y_train, y_test


def add_midprice_label(df: pd.DataFrame, time_delay: int):
    df[f"midprice_after_{time_delay}"] = df["n_midprice"].shift(-time_delay)
    return df


def sequentialize_certain_features(
    df: pd.DataFrame, feature_columns: list[str], label_column: str, seq_length: int
):
    """创建序列特征（用于LSTM）"""
    print(f"Sequentializing features, sequence length: {seq_length}")

    features_np = df[feature_columns].values
    labels_np = df[label_column].values

    # 创建视图
    X_view = sliding_window_view(features_np, (seq_length, features_np.shape[1]))
    X = X_view[:, 0, :, :]

    y = labels_np[seq_length - 1 :]

    # X, y = [], []
    # features = df[feature_columns]
    # for i in range(len(features) - seq_length):
    #     X.append(features[i : i + seq_length])
    #     y.append(df.iloc[i + seq_length - 1][label_column])
    print(f"Sequentializing features... done.")
    return X, y


def display_detail(df: pd.DataFrame, feature: str):
    print(df[feature].head())


def scale_train(scaler, X_train: NDArray):
    original_shape_train = X_train.shape

    # 重塑为2D用于缩放 (samples*timesteps, features)
    n_features = X_train.shape[2]

    X_train_2d = X_train.reshape(-1, n_features)
    X_train_scaled_2d = scaler.fit_transform(X_train_2d)

    # 重塑回3D (samples, timesteps, features)
    X_train_scaled = X_train_scaled_2d.reshape(original_shape_train)

    return X_train_scaled


def scale_test(scaler, X_test: NDArray):
    original_shape_test = X_test.shape
    n_features = X_test.shape[2]

    # 重塑为2D用于缩放 (samples*timesteps, features)
    X_test_2d = X_test.reshape(-1, n_features)
    X_test_scaled_2d = scaler.transform(X_test_2d)

    # 重塑回3D (samples, timesteps, features)
    X_test_scaled = X_test_scaled_2d.reshape(original_shape_test)

    return X_test_scaled


def scale(X_train: NDArray, X_test: NDArray):
    print("Scaling data...")

    balance_scaler = MinMaxScaler(feature_range=(-1, 1))

    volume_scaler = RobustScaler()
    X_train_scaled = np.concatenate(
        [
            scale_train(balance_scaler, X_train[:, :, 0:17]),
            scale_train(volume_scaler, X_train[:, :, 17:19]),
            X_train[:, :, 19:],
        ],
        axis=2,
    )
    # print(f"After Train Scaling: {price_scaler.center_,price_scaler.scale_}")
    X_test_scaled = np.concatenate(
        [
            scale_test(balance_scaler, X_test[:, :, 0:17]),
            scale_test(volume_scaler, X_test[:, :, 17:19]),
            X_test[:, :, 19:],
        ],
        axis=2,
    )
    # print(f"After Test Scaling: {price_scaler.center_,price_scaler.scale_}")

    joblib.dump(balance_scaler, "./balance.joblib")
    joblib.dump(volume_scaler, "./volume.joblib")
    print("The scalers saved to . ")

    print("Scaling data... Done.")
    return X_train_scaled, X_test_scaled


def inverse_scale(scaler, X_scaled: np.ndarray):
    original_shape = X_scaled.shape
    n_features = X_scaled.shape[-1]

    # 展平成2D进行逆变换（与缩放时保持一致）
    X_2d = X_scaled.reshape(-1, n_features)
    X_original_2d = scaler.inverse_transform(X_2d)

    # 重塑回原始形状
    X_original = X_original_2d.reshape(original_shape)

    return X_original


def comprehensive_scaler_selection(
    df, numerical_cols=None, skew_threshold=1.0, outlier_threshold=1.5
):
    """
    综合检查并推荐Scaler的完整流程
    """
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns

    print("=" * 60)
    print("数据预处理Scaler选择分析报告")
    print("=" * 60)

    # 1. 检查偏度
    print("\n📊 1. 数据分布检查（偏度分析）")
    print("-" * 40)
    skew_df = check_skewness(df, numerical_cols, skew_threshold)

    # 2. 检查异常值
    print("\n📊 2. 异常值检查（IQR方法）")
    print("-" * 40)
    outliers_df = detect_outliers_iqr(df, numerical_cols, outlier_threshold)

    # 3. 检查值范围
    print("\n📊 3. 值范围检查")
    print("-" * 40)
    range_df = check_value_ranges(df, numerical_cols)

    # 4. 综合推荐
    print("\n🎯 4. 综合Scaler推荐")
    print("-" * 40)

    recommendations = {}
    for col in numerical_cols:
        # 获取该特征的各项检查结果
        skew_info = skew_df[skew_df["feature"] == col].iloc[0]
        outlier_info = outliers_df[outliers_df["feature"] == col].iloc[0]
        range_info = range_df[range_df["feature"] == col].iloc[0]

        # 决策逻辑
        if skew_info["is_highly_skewed"]:
            recommendations[col] = {
                "scaler": "PowerTransformer + StandardScaler",
                "reason": f"严重偏态（偏度={skew_info['skewness']:.2f}）",
            }
        elif outlier_info["is_high_outlier"]:
            recommendations[col] = {
                "scaler": "RobustScaler",
                "reason": f"异常值较多（{outlier_info['outlier_percentage']:.1f}%）",
            }
        elif range_info["has_clear_bounds"]:
            recommendations[col] = {
                "scaler": "MinMaxScaler",
                "reason": f"有明确边界（{range_info['bound_type']}）",
            }
        else:
            recommendations[col] = {
                "scaler": "StandardScaler",
                "reason": "分布相对正常，无明显异常值",
            }

    # 打印推荐结果
    rec_df = pd.DataFrame.from_dict(recommendations, orient="index")
    rec_df.index.name = "feature"
    rec_df.reset_index(inplace=True)

    print("\n推荐方案汇总:")
    print(rec_df.to_string(index=False))

    # 统计各Scaler使用频率
    scaler_counts = rec_df["scaler"].value_counts()
    print(f"\n📈 Scaler使用统计:")
    for scaler, count in scaler_counts.items():
        print(f"  {scaler}: {count}个特征")

    # 给出最终建议
    if len(scaler_counts) == 1:
        print(f"\n✅ 建议所有特征使用: {scaler_counts.index[0]}")
    else:
        print(f"\n⚠️ 建议使用混合Scaler（不同特征使用不同Scaler）")
        print("可以使用ColumnTransformer:")
        print(
            """
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('power', PowerTransformer(), [偏态特征列表]),
        ('robust', RobustScaler(), [异常值多的特征列表]),
        ('minmax', MinMaxScaler(), [有边界特征列表]),
        ('standard', StandardScaler(), [其他特征])
    ])
        """
        )

    return {
        "skew_df": skew_df,
        "outliers_df": outliers_df,
        "range_df": range_df,
        "recommendations": rec_df,
    }


def check_value_ranges(df, numerical_cols=None):
    """
    检查数值范围，判断是否有明确物理边界

    常见有明确边界的特征：
    - 百分比：0-100
    - 概率：0-1
    - 年龄：0-150
    - 评分：1-5, 1-10
    - 二值特征：0/1
    """
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns

    range_results = []

    # 常见边界条件
    common_bounds = {
        "percentage": (0, 100),
        "probability": (0, 1),
        "rating_5": (1, 5),
        "rating_10": (1, 10),
        "binary": (0, 1),
        "age": (0, 150),
    }

    for col in numerical_cols:
        data = df[col].dropna()
        min_val = data.min()
        max_val = data.max()
        range_val = max_val - min_val

        # 检查是否符合常见边界
        has_clear_bounds = False
        bound_type = None

        for bound_name, (lower, upper) in common_bounds.items():
            if min_val >= lower and max_val <= upper:
                has_clear_bounds = True
                bound_type = bound_name
                break

        # 自定义边界检查（根据业务知识）
        # 例如：如果数据在[0, 255]之间，可能是图像像素

        range_results.append(
            {
                "feature": col,
                "min": min_val,
                "max": max_val,
                "range": range_val,
                "has_clear_bounds": has_clear_bounds,
                "bound_type": bound_type,
                "recommendation": (
                    "MinMaxScaler" if has_clear_bounds else "根据分布选择"
                ),
            }
        )

    range_df = pd.DataFrame(range_results)

    # 打印有明确边界的特征
    bounded_features = range_df[range_df["has_clear_bounds"]]
    if len(bounded_features) > 0:
        print(f"✅ 发现 {len(bounded_features)} 个有明确边界的特征:")
        print(
            bounded_features[["feature", "min", "max", "bound_type", "recommendation"]]
        )
        print("\n这些特征适合使用MinMaxScaler")

    return range_df


def detect_outliers_iqr(df, numerical_cols=None, threshold=1.5):
    """
    使用IQR方法检测异常值

    threshold通常取1.5（中度异常）或3（极端异常）
    """
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns

    outliers_results = []

    for col in numerical_cols:
        data = df[col].dropna()

        # 计算Q1, Q3, IQR
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        # 异常值边界
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # 检测异常值
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        n_outliers = len(outliers)
        outlier_percentage = n_outliers / len(data) * 100

        outliers_results.append(
            {
                "feature": col,
                "q1": Q1,
                "q3": Q3,
                "iqr": IQR,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "n_outliers": n_outliers,
                "outlier_percentage": outlier_percentage,
                "is_high_outlier": outlier_percentage > 5,  # 超过5%视为有大量异常值
                "recommendation": (
                    "RobustScaler" if outlier_percentage > 5 else "StandardScaler"
                ),
            }
        )

    outliers_df = pd.DataFrame(outliers_results)

    # 打印有大量异常值的特征
    high_outlier_features = outliers_df[outliers_df["is_high_outlier"]]
    if len(high_outlier_features) > 0:
        print(f"⚠️ 发现 {len(high_outlier_features)} 个特征有大量异常值（>5%）:")
        print(
            high_outlier_features[["feature", "outlier_percentage", "recommendation"]]
        )
        print("\n推荐使用RobustScaler处理这些特征")
    else:
        print(f"✅ 异常值比例正常，可以考虑使用StandardScaler")

    return outliers_df


# 方法3：Z-score方法（适合近似正态分布）
def detect_outliers_zscore(df, numerical_cols=None, threshold=3):
    """使用Z-score方法检测异常值"""
    outliers_results = []

    for col in numerical_cols:
        data = df[col].dropna()
        z_scores = np.abs(stats.zscore(data))
        outliers = data[z_scores > threshold]
        outlier_percentage = len(outliers) / len(data) * 100

        outliers_results.append(
            {
                "feature": col,
                "outlier_percentage": outlier_percentage,
                "is_high_outlier": outlier_percentage > 5,
            }
        )

    return pd.DataFrame(outliers_results)


def check_skewness(df, numerical_cols=None, threshold=1.0):
    """
    检查偏度，判断是否需要PowerTransformer

    偏度判断标准：
    |Skewness| < 0.5: 近似对称
    0.5 ≤ |Skewness| < 1: 中等偏态
    |Skewness| ≥ 1: 严重偏态（需要处理）
    """
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns

    skewness_results = []
    for col in numerical_cols:
        skew = df[col].skew()
        is_highly_skewed = abs(skew) >= threshold
        skewness_results.append(
            {
                "feature": col,
                "skewness": skew,
                "abs_skewness": abs(skew),
                "is_highly_skewed": is_highly_skewed,
                "recommendation": (
                    "PowerTransformer" if is_highly_skewed else "StandardScaler"
                ),
            }
        )

    skew_df = pd.DataFrame(skewness_results)

    # 打印严重偏态的特征
    highly_skewed = skew_df[skew_df["is_highly_skewed"]]
    if len(highly_skewed) > 0:
        print(f"⚠️ 发现 {len(highly_skewed)} 个严重偏态特征（|偏度|≥{threshold}）:")
        print(highly_skewed[["feature", "skewness", "recommendation"]])
        print("\n推荐先对这些特征使用PowerTransformer，然后再用StandardScaler")
    else:
        print(f"✅ 没有严重偏态特征，可以直接使用StandardScaler")

    return skew_df


# 使用示例
# skew_df = check_skewness(df, threshold=1.0)
if __name__ == "__main__":
    df = pd.read_csv("./merged_data/merged_0.csv")
    df = create_all_features(df)
    results = comprehensive_scaler_selection(
        df, skew_threshold=1.0, outlier_threshold=1.5
    )
