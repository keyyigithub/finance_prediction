from json import load
import os
import joblib
from typing import List
from numpy.typing import NDArray
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Conv1D,
    LayerNormalization,
    Concatenate,
    MultiHeadAttention,
    Flatten,
    Bidirectional,
)
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import pandas as pd
import numpy as np


def log_transform(df: pd.DataFrame, features):
    df[features] = df[features].apply(lambda x: np.sign(x) * np.log1p(np.abs(x)))
    return df


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

    return labels


class Predictor:
    def __init__(self):
        self.selected_features = [
            # features already balanced, using MinMaxScaler:
            "n_close",  # 标准化后的收盘价
            "n_midprice",  # 标准化后的中间价
            "n_bid5",
            "n_ask5",
            "bid_ask_spread",  # 买卖价差
            "size_imbalance_1",  # 一档买卖量不平衡
            "microprice",  # 微观价格（考虑深度的加权价格）
            "size_imbalance_5",
            "total_depth",  # 总市场深度
            "midprice_momentum_20",  # 20期动量
            "macd",  # MACD线
            "ma_cross_5_20",  # 移动平均线交叉信号
            "price_acceleration",  # 价格加速度
            "volatility_20",  # 20期波动率
            "bollinger_width",  # 布林带宽度
            "parkinson_vol_20",  # Parkinson波动率（更准确的高低价估计）
            "rsi_14",  # 14期RSI
            "stochastic_k",  # 随机指标K值
            "bias_20",  # 20期乖离率
            # features need to transform using log1p
            "amount_delta",  # 成交额变化
            "volume_momentum",  # 成交量动量
            "n_asize5",
            "n_bsize5",
            # features that don't need a scaler
            "time_sin",  # 时间正弦编码
            "time_cos",  # 时间余弦编码
        ]

        self.input_shape = (80, 25)
        self.num_classes = 2
        self._build_model_architecture()
        self.balance_scaler = joblib.load(
            os.path.join(os.path.dirname(__file__), "balance.joblib")
        )
        self.volume_scaler = joblib.load(
            os.path.join(os.path.dirname(__file__), "volume.joblib")
        )
        # Used when testing
        # self.balance_scaler = joblib.load("./balance.joblib")
        # self.volume_scaler = joblib.load("./volume.joblib")
        # self._load_weights("./model.weights.h5")

    def predict(self, data: List[pd.DataFrame]) -> List[List[int]]:
        results = []
        for df in data:
            X = self.preprocess(df)
            pred_labels = []
            for td in [40]:
                self._load_weights(
                    os.path.join(
                        os.path.dirname(__file__), f"continue_model_{td}.weights.h5"
                    )
                )
                y = self.model.predict(X)
                y = get_label(y, np.zeros(len(y)), td, 0.0019 * 200, 0.0019 * 200)
                pred_labels.append(y[0])

            results.append(pred_labels)

        return results

    def _build_model_architecture(self):
        """构建模型架构（与训练时相同）"""

        # 构建基础模型
        def build_conv_residual_block(input_shape):
            inputs = keras.Input(input_shape)
            shortcut = inputs
            feat_1 = Conv1D(
                filters=32, kernel_size=3, padding="same", activation="tanh"
            )(inputs)
            feat_2 = Conv1D(
                filters=32, kernel_size=5, padding="same", activation="tanh"
            )(inputs)
            feat_3 = Conv1D(
                filters=32, kernel_size=7, padding="same", activation="tanh"
            )(inputs)
            outputs = Concatenate(axis=2)([feat_1, feat_2, feat_3, shortcut])
            return keras.Model(inputs, outputs)

        def build_lstm_residual_block(input_shape, units=256):
            inputs = keras.Input(input_shape)
            shortcut = inputs
            x = Bidirectional(
                LSTM(units, return_sequences=True, kernel_regularizer=l2(0.01))
            )(inputs)
            x = Bidirectional(
                LSTM(units, return_sequences=True, kernel_regularizer=l2(0.01))
            )(inputs)
            shortcut_reshaped = Dense(units * 2)(shortcut)
            x = shortcut_reshaped + x
            outputs = Dropout(0.3)(x)
            return keras.Model(inputs, outputs)

        def build_base_model(input_shape):
            inputs = keras.Input(input_shape)
            x = build_conv_residual_block(input_shape)(inputs)
            x = LayerNormalization()(x)
            # x = Dropout(0.3)(x)
            # x = build_conv_residual_block(input_shape)(inputs)
            # x = LayerNormalization()(x)
            # x = Dropout(0.3)(x)

            x = build_lstm_residual_block((x.shape[1], x.shape[2]), units=128)(x)
            x = LayerNormalization()(x)
            # x = build_lstm_residual_block((x.shape[1], x.shape[2]))(x)
            # x = LayerNormalization()(x)

            x = LSTM(128, return_sequences=True)(x)
            short_cut = x
            attention_output_1 = MultiHeadAttention(num_heads=4, key_dim=128)(x, x)
            x = LayerNormalization()(x + attention_output_1)
            # attention_output_2 = MultiHeadAttention(num_heads=4, key_dim=256)(x, x)
            # x = LayerNormalization()(x + attention_output_2)

            x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(x)
            x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
            short_cut = Dense(64)(short_cut)
            x = short_cut + x
            # x = Dropout(0.3)(x)

            outputs = Flatten()(x)

            model = keras.Model(inputs, outputs)
            return model

        # 构建完整模型
        inputs = keras.Input(shape=self.input_shape)
        x = build_base_model(self.input_shape)(inputs)
        x = Dense(64, activation="tanh", kernel_regularizer=l2(0.01))(x)
        # x = Dropout(0.3)(x)
        x = Dense(32, activation="tanh", kernel_regularizer=l2(0.01))(x)
        # x = Dropout(0.3)(x)
        outputs = Dense(1, activation="tanh")(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer="adam",
            loss="mse",
        )

    def _load_weights(self, weights_path):
        """加载权重"""
        try:
            self.model.load_weights(weights_path)
            print(f"权重加载成功: {weights_path}")
            return True
        except Exception as e:
            print(f"权重加载失败: {e}")
            print("使用随机初始化权重")
            return False

    def create_all_features(self, df: pd.DataFrame):
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
        print(isinstance(time_series, pd.Series))
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
        df["time_cos"] = np.cos(
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

        rolling_window = 20
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
        df["volatility_20"] = df["midprice_return"].rolling(window=20).std() * np.sqrt(
            20
        )
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

        df = log_transform(df, self.selected_features[19:21])
        print(f"特征工程完成")

        return df

    def preprocess(self, df: pd.DataFrame):
        df = self.create_all_features(df)
        X = df[self.selected_features].values
        X[:, 0:19] = self.balance_scaler.transform(X[:, 0:19])
        X[:, 19:21] = self.volume_scaler.transform(X[:, 19:21])

        X = X.reshape(1, X.shape[0], X.shape[1])
        X = X[:, 20:, :]
        return X


def test():
    pred = Predictor()
    df = pd.read_csv("./merged_data/merged_0.csv")
    data = []
    for i in range(100):
        data.append(df.iloc[i : 100 + i])

    result = pred.predict(data)
    print(result)
    print(np.any(np.isnan(np.asarray(result))))

    pass


# test()
