from os import ST_NOATIME
from numpy.typing import NDArray
from sklearn.preprocessing import RobustScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
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
    # "sym",
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

    print(f"特征工程完成")

    return df


def add_midprice_label(df: pd.DataFrame, time_delay: int):
    df[f"midprice_after_{time_delay}"] = df["n_midprice"].shift(-time_delay)
    return df


def sequentialize_certain_features(
    df: pd.DataFrame, feature_columns: list[str], label_column: str, seq_length: int
):
    """创建序列特征（用于LSTM）"""
    print(f"Sequentializing features, sequence length: {seq_length}")

    X, y = [], []
    features = df[feature_columns]
    for i in range(len(features) - seq_length):
        X.append(features[i : i + seq_length])
        y.append(df.iloc[i + seq_length - 1][label_column])

    print(f"Sequentializing features... done.")
    return np.array(X), np.array(y)


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


def split_and_scale(X: NDArray, y: NDArray, test_size=0.2):
    print(f"Splitting Data... Test size: {test_size}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    print("Splitting Data... Done.")

    print("Scaling data...")

    # Different scalers
    price_scaler = RobustScaler()
    # print(f"Initial: {price_scaler.center_,price_scaler.scale_}")
    microstructure_scaler = Pipeline(
        [
            ("log", FunctionTransformer(lambda x: np.sign(x) * np.log1p(np.abs(x)))),
            ("scale", RobustScaler()),
        ]
    )
    momentum_scaler = RobustScaler(quantile_range=(10, 90))
    volatility_scaler = Pipeline(
        [
            ("log", FunctionTransformer(lambda x: np.sign(x) * np.log1p(np.abs(x)))),
            ("scale", RobustScaler()),
        ]
    )
    technical_scaler = RobustScaler()
    volume_scaler = Pipeline(
        [
            (
                "sign_log",
                FunctionTransformer(lambda x: np.sign(x) * np.log1p(np.abs(x))),
            ),
            ("scale", RobustScaler()),
        ]
    )
    X_train_scaled = np.concatenate(
        [
            scale_train(price_scaler, X_train[:, :, 0:3]),
            # X_train[:, :, 0:3],
            scale_train(microstructure_scaler, X_train[:, :, 3:7]),
            scale_train(momentum_scaler, X_train[:, :, 7:11]),
            scale_train(volatility_scaler, X_train[:, :, 11:14]),
            scale_train(technical_scaler, X_train[:, :, 14:17]),
            scale_train(volume_scaler, X_train[:, :, 17:19]),
            X_train[:, :, 19].reshape(X_train.shape[0], X_train.shape[1], 1),
        ],
        axis=2,
    )
    # print(f"After Train Scaling: {price_scaler.center_,price_scaler.scale_}")
    X_test_scaled = np.concatenate(
        [
            scale_test(price_scaler, X_test[:, :, 0:3]),
            # X_test[:, :, 0:3],
            scale_test(microstructure_scaler, X_test[:, :, 3:7]),
            scale_test(momentum_scaler, X_test[:, :, 7:11]),
            scale_test(volatility_scaler, X_test[:, :, 11:14]),
            scale_test(technical_scaler, X_test[:, :, 14:17]),
            scale_test(volume_scaler, X_test[:, :, 17:19]),
            X_test[:, :, 19].reshape(X_test.shape[0], X_test.shape[1], 1),
        ],
        axis=2,
    )
    # print(f"After Test Scaling: {price_scaler.center_,price_scaler.scale_}")

    print("Scaling data... Done.")
    return X_train_scaled, X_test_scaled, y_train, y_test


def inverse_scale(scaler, X_scaled: np.ndarray):
    original_shape = X_scaled.shape
    n_features = X_scaled.shape[-1]

    # 展平成2D进行逆变换（与缩放时保持一致）
    X_2d = X_scaled.reshape(-1, n_features)
    X_original_2d = scaler.inverse_transform(X_2d)

    # 重塑回原始形状
    X_original = X_original_2d.reshape(original_shape)

    return X_original
