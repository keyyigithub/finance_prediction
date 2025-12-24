from json import load
import os
import joblib
from typing import List
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from tensorflow import keras
import pandas as pd
import numpy as np


class Predictor:
    def __init__(self):
        self.selected_features = [
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
        self.balance_scaler = joblib.load("./balance.joblib")
        self.volume_scaler = joblib.load("./volume.joblib")
        self.load_model("./lstm_price_prediction_model.keras")

    def predict(self, data: List[pd.DataFrame]) -> List[List[int]]:
        results = []
        for df in data:
            X = self.preprocess(df)
            y = self.model.predict(X)
            results.append(y)

        return results

    def load_model(self, model_path: str):
        self.model = keras.saving.load_model(model_path)

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

        print(f"特征工程完成")

        return df

    def preprocess(self, df: pd.DataFrame):
        df = self.create_all_features(df)
        X = df[self.selected_features].values
        X = X.reshape(1, X.shape[0], X.shape[1])
        X[:, :, 0:17] = self.balance_scaler.transform(X[:, :, 0:17])
        X[:, :, 17:19] = self.volume_scaler.transform(X[:, :, 17:19])

        return X


def test():
    pred = Predictor()


if __name__ == "__main__":
    test()
