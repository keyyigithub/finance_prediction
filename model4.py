import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.arraylike import dispatch_ufunc_with_out
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
class MarketDataFeatureEngineer:
    """金融市场数据特征工程类"""

    def __init__(self, seq_length=100, target_label="label5"):
        self.seq_length = seq_length
        self.target_label = target_label
        self.scaler = StandardScaler()

        # 基础特征列
        self.base_price_columns = ["n_close", "n_midprice"]
        self.bid_columns = [f"n_bid{i}" for i in range(1, 6)]
        self.ask_columns = [f"n_ask{i}" for i in range(1, 6)]
        self.bid_size_columns = [f"n_bsize{i}" for i in range(1, 6)]
        self.ask_size_columns = [f"n_asize{i}" for i in range(1, 6)]

    def create_all_features(self, df):
        """创建所有特征"""
        print("开始特征工程...")

        # 1. 基础价格特征
        df = self.create_basic_price_features(df)

        # 2. 订单簿特征
        df = self.create_order_book_features(df)

        # 3. 技术指标特征
        df = self.create_technical_indicators(df)

        # 4. 统计特征
        df = self.create_statistical_features(df)

        # 5. 波动率特征
        df = self.create_volatility_features(df)

        # 6. 量价关系特征
        df = self.create_volume_price_features(df)

        # 7. 时间特征
        df = self.create_time_features(df)

        # 8. 衍生特征
        df = self.create_derived_features(df)

        print(
            f"特征工程完成，原始特征数: {len(self.get_original_columns(df))}, "
            f"新特征数: {len(self.get_new_feature_columns(df))}"
        )

        return df

    def create_basic_price_features(self, df):
        """基础价格特征"""
        print("  创建基础价格特征...")

        # 1. 价格变化率
        df["midprice_return"] = df["n_midprice"].pct_change()
        df["close_return"] = df["n_close"].pct_change()

        # 2. 对数收益率（更稳定）
        df["midprice_log_return"] = np.log(df["n_midprice"] / df["n_midprice"].shift(1))

        # 3. 价格动量
        for window in [1, 5, 20, 50]:
            df[f"midprice_momentum_{window}"] = df["n_midprice"] - df[
                "n_midprice"
            ].shift(window)

        # 4. 价格加速度（动量的变化）
        df["price_acceleration"] = df["midprice_momentum_1"] - df[
            "midprice_momentum_1"
        ].shift(1)

        # 5. 价格位置特征
        rolling_window = 50
        df["midprice_ma"] = df["n_midprice"].rolling(window=rolling_window).mean()
        df["midprice_std"] = df["n_midprice"].rolling(window=rolling_window).std()
        df["midprice_zscore"] = (df["n_midprice"] - df["midprice_ma"]) / df[
            "midprice_std"
        ]

        # 6. 布林带特征
        df["bollinger_upper"] = df["midprice_ma"] + 2 * df["midprice_std"]
        df["bollinger_lower"] = df["midprice_ma"] - 2 * df["midprice_std"]
        df["bollinger_width"] = (df["bollinger_upper"] - df["bollinger_lower"]) / df[
            "midprice_ma"
        ]
        df["bollinger_position"] = (df["n_midprice"] - df["bollinger_lower"]) / (
            df["bollinger_upper"] - df["bollinger_lower"]
        )

        return df

    def create_order_book_features(self, df):
        """订单簿特征(只使用单行数据)"""
        print("  创建订单簿特征...")

        # 1. 买卖价差（Spread）
        df["bid_ask_spread"] = df["n_ask1"] - df["n_bid1"]
        df["relative_spread"] = df["bid_ask_spread"] / df["n_midprice"]

        # 2. 订单簿不平衡（Order Book Imbalance）
        for depth in range(1, 6):
            df[f"price_imbalance_{depth}"] = (
                df[f"n_ask{depth}"] - df[f"n_bid{depth}"]
            ) / (df[f"n_ask{depth}"] + df[f"n_bid{depth}"])
            df[f"size_imbalance_{depth}"] = (
                df[f"n_bsize{depth}"] - df[f"n_asize{depth}"]
            ) / (df[f"n_bsize{depth}"] + df[f"n_asize{depth}"])

        # 3. 加权平均价格
        total_bid_size = sum([df[f"n_bsize{i}"] for i in range(1, 6)])
        total_ask_size = sum([df[f"n_asize{i}"] for i in range(1, 6)])

        weighted_bid_price = (
            sum([df[f"n_bid{i}"] * df[f"n_bsize{i}"] for i in range(1, 6)])
            / total_bid_size
        )
        weighted_ask_price = (
            sum([df[f"n_ask{i}"] * df[f"n_asize{i}"] for i in range(1, 6)])
            / total_ask_size
        )

        df["weighted_midprice"] = (weighted_bid_price + weighted_ask_price) / 2
        df["microprice"] = (
            df["n_bid1"] * total_ask_size + df["n_ask1"] * total_bid_size
        ) / (total_bid_size + total_ask_size)

        # 4. 订单簿深度
        df["bid_depth"] = sum([df[f"n_bsize{i}"] for i in range(1, 6)])
        df["ask_depth"] = sum([df[f"n_asize{i}"] for i in range(1, 6)])
        df["total_depth"] = df["bid_depth"] + df["ask_depth"]
        df["depth_imbalance"] = (df["bid_depth"] - df["ask_depth"]) / df["total_depth"]

        # 5. 订单簿斜率
        for depth in [2, 3, 4, 5]:
            if depth > 1:
                bid_slope = (df[f"n_bid1"] - df[f"n_bid{depth}"]) / (depth - 1)
                ask_slope = (df[f"n_ask{depth}"] - df[f"n_ask1"]) / (depth - 1)
                df[f"bid_slope_{depth}"] = bid_slope
                df[f"ask_slope_{depth}"] = ask_slope

        # 6. 订单簿压力
        df["buy_pressure"] = df["bid_depth"] / (df["bid_depth"] + df["ask_depth"])
        df["sell_pressure"] = 1 - df["buy_pressure"]

        return df

    def create_technical_indicators(self, df):
        """技术指标特征"""
        print("  创建技术指标特征...")

        # 使用价格序列计算技术指标
        price_series = df["n_midprice"].values

        # 1. RSI（相对强弱指数）
        periods = [6, 14, 24]
        for period in periods:
            # 手动计算RSI
            delta = df["n_midprice"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # 2. MACD
        exp1 = df["n_midprice"].ewm(span=12, adjust=False).mean()
        exp2 = df["n_midprice"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # 3. 移动平均线
        ma_periods = [5, 10, 20, 30, 50]
        for period in ma_periods:
            df[f"ma_{period}"] = df["n_midprice"].rolling(window=period).mean()
            # 价格相对于移动平均的位置
            df[f"price_vs_ma_{period}"] = df["n_midprice"] / df[f"ma_{period}"] - 1

        # 4. 移动平均线交叉
        df["ma_cross_5_20"] = df["ma_5"] - df["ma_20"]
        df["ma_cross_10_30"] = df["ma_10"] - df["ma_30"]

        # 5. 随机指标（Stochastic Oscillator）
        low_min = df["n_midprice"].rolling(window=14).min()
        high_max = df["n_midprice"].rolling(window=14).max()
        df["stochastic_k"] = 100 * (df["n_midprice"] - low_min) / (high_max - low_min)
        df["stochastic_d"] = df["stochastic_k"].rolling(window=3).mean()

        # 6. 乖离率（BIAS）
        for period in [5, 10, 20, 30, 50]:
            df[f"bias_{period}"] = (
                (df["n_midprice"] - df[f"ma_{period}"]) / df[f"ma_{period}"] * 100
            )

        return df

    def create_statistical_features(self, df):
        """统计特征"""
        print("  创建统计特征...")

        windows = [5, 10, 20, 30, 50]

        for window in windows:
            # 滚动统计量
            df[f"return_mean_{window}"] = (
                df["midprice_return"].rolling(window=window).mean()
            )
            df[f"return_std_{window}"] = (
                df["midprice_return"].rolling(window=window).std()
            )
            df[f"return_skew_{window}"] = (
                df["midprice_return"].rolling(window=window).skew()
            )
            df[f"return_kurtosis_{window}"] = (
                df["midprice_return"].rolling(window=window).kurt()
            )

            # 分位数
            df[f"return_q25_{window}"] = (
                df["midprice_return"].rolling(window=window).quantile(0.25)
            )
            df[f"return_q75_{window}"] = (
                df["midprice_return"].rolling(window=window).quantile(0.75)
            )
            df[f"return_iqr_{window}"] = (
                df[f"return_q75_{window}"] - df[f"return_q25_{window}"]
            )

            # 极值统计
            df[f"price_max_{window}"] = df["n_midprice"].rolling(window=window).max()
            df[f"price_min_{window}"] = df["n_midprice"].rolling(window=window).min()
            df[f"price_range_{window}"] = (
                df[f"price_max_{window}"] - df[f"price_min_{window}"]
            )
            df[f"price_range_pct_{window}"] = (
                df[f"price_range_{window}"] / df[f"price_min_{window}"]
            )

        # 自相关性特征
        for lag in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            df[f"return_autocorr_lag{lag}"] = (
                df["midprice_return"]
                .rolling(window=50)
                .apply(lambda x: x.autocorr(lag=lag), raw=False)
            )

        return df

    def create_volatility_features(self, df):
        """波动率特征"""
        print("  创建波动率特征...")

        windows = [5, 10, 20, 30, 50]

        for window in windows:
            # 历史波动率（基于收益率）
            df[f"volatility_{window}"] = df["midprice_return"].rolling(
                window=window
            ).std() * np.sqrt(window)

            # Parkinson波动率（使用价格范围）
            high_low_ratio = (
                np.log(
                    df["n_midprice"].rolling(window=2).max()
                    / df["n_midprice"].rolling(window=2).min()
                )
                ** 2
            )
            df[f"parkinson_vol_{window}"] = np.sqrt(
                (1 / (4 * window * np.log(2)))
                * high_low_ratio.rolling(window=window).sum()
            )

        # 波动率比率
        df["volatility_ratio_5_20"] = df["volatility_5"] / df["volatility_20"]
        df["volatility_ratio_10_30"] = df["volatility_10"] / df["volatility_30"]

        # 波动率变化
        df["volatility_change"] = df["volatility_5"].pct_change()

        # 已实现波动率
        df["realized_volatility"] = np.sqrt(
            (df["midprice_return"] ** 2).rolling(window=20).sum()
        )

        return df

    def create_volume_price_features(self, df):
        """量价关系特征"""
        print("  创建量价关系特征...")

        # 1. 量价相关性
        windows = [5, 10, 20, 30, 50]
        for window in windows:
            df[f"volume_price_corr_{window}"] = (
                df["amount_delta"].rolling(window=window).corr(df["n_midprice"])
            )

        # 2. 成交量加权价格
        df["vwap"] = (df["n_midprice"] * df["amount_delta"]).cumsum() / df[
            "amount_delta"
        ].cumsum()
        df["price_vwap_ratio"] = df["n_midprice"] / df["vwap"]

        # 3. 成交量动量
        df["volume_momentum"] = df["amount_delta"] - df["amount_delta"].shift(1)
        df["volume_acceleration"] = df["volume_momentum"] - df["volume_momentum"].shift(
            1
        )

        # 4. 成交量比率
        for window in [5, 10, 20, 30, 50]:
            df[f"volume_ma_{window}"] = df["amount_delta"].rolling(window=window).mean()
            df[f"volume_ratio_{window}"] = (
                df["amount_delta"] / df[f"volume_ma_{window}"]
            )

        # 5. 量价背离
        df["price_up_volume_down"] = (
            (df["midprice_return"] > 0) & (df["volume_momentum"] < 0)
        ).astype(int)
        df["price_down_volume_up"] = (
            (df["midprice_return"] < 0) & (df["volume_momentum"] > 0)
        ).astype(int)

        # 6. 订单流不平衡
        df["order_flow_imbalance"] = (df["bid_depth"] - df["ask_depth"]) * df[
            "n_midprice"
        ]

        return df

    def create_time_features(self, df):
        """时间特征"""
        print("  创建时间特征...")

        if "time" in df.columns:
            # 解析时间
            time_series = pd.to_datetime(df["time"], format="%H:%M:%S")

            # 1. 日内时间特征
            df["hour"] = time_series.dt.hour
            df["minute"] = time_series.dt.minute
            df["second"] = time_series.dt.second

            # 2. 时间周期（上午/下午）
            df["is_morning"] = (df["hour"] < 12).astype(int)
            df["is_afternoon"] = ((df["hour"] >= 13) & (df["hour"] < 15)).astype(int)

            # 3. 交易时段特征（开盘、收盘）
            df["is_opening"] = ((df["hour"] == 9) & (df["minute"] < 30)).astype(int)
            df["is_closing"] = ((df["hour"] == 14) & (df["minute"] > 45)).astype(int)

            # 4. 时间sin/cos编码（处理周期性）
            df["time_sin"] = np.sin(
                2
                * np.pi
                * (df["hour"] * 3600 + df["minute"] * 60 + df["second"])
                / 86400
            )
            df["time_cos"] = np.cos(
                2
                * np.pi
                * (df["hour"] * 3600 + df["minute"] * 60 + df["second"])
                / 86400
            )

        # 5. 时间间隔
        if "time" in df.columns:
            time_diff = pd.to_datetime(df["time"]).diff().dt.total_seconds().fillna(3)
            df["time_interval"] = time_diff
            df["is_irregular_time"] = (df["time_interval"] != 3).astype(int)

        return df

    def create_derived_features(self, df):
        """衍生特征"""
        print("  创建衍生特征...")

        # 1. 特征交互
        df["spread_depth_interaction"] = df["bid_ask_spread"] * df["total_depth"]
        df["volatility_momentum_interaction"] = (
            df["volatility_5"] * df["midprice_momentum_5"]
        )

        # 2. 多项式特征
        df["midprice_squared"] = df["n_midprice"] ** 2
        df["midprice_cubed"] = df["n_midprice"] ** 3
        df["volume_squared"] = df["amount_delta"] ** 2

        # 3. 比率特征
        df["price_volume_ratio"] = df["n_midprice"] / (df["amount_delta"] + 1e-10)
        df["spread_depth_ratio"] = df["bid_ask_spread"] / (df["total_depth"] + 1e-10)

        # 4. 累积特征
        df["cumulative_return"] = (1 + df["midprice_return"]).cumprod() - 1
        df["cumulative_volume"] = df["amount_delta"].cumsum()

        # 5. 差异特征
        df["price_vs_vwap_diff"] = df["n_midprice"] - df["vwap"]
        df["price_vs_microprice_diff"] = df["n_midprice"] - df["microprice"]

        # 6. 归一化特征
        df["normalized_volume"] = (df["amount_delta"] - df["amount_delta"].mean()) / df[
            "amount_delta"
        ].std()
        df["normalized_spread"] = (
            df["bid_ask_spread"] - df["bid_ask_spread"].mean()
        ) / df["bid_ask_spread"].std()

        return df

    def feature_selection(self, df, target_column, method="importance", n_features=50):
        """特征选择"""
        print("  特征选择...")

        if method == "correlation":
            # 基于相关性的特征选择
            correlation_matrix = (
                df.corr()[target_column].abs().sort_values(ascending=False)
            )
            selected_features = correlation_matrix.head(n_features).index.tolist()

        elif method == "variance":
            # 基于方差的特征选择
            variances = df.var().sort_values(ascending=False)
            selected_features = variances.head(n_features).index.tolist()

        print(f"  选择了 {len(selected_features)} 个特征")
        return selected_features

    def get_original_columns(self, df):
        """获取原始列名"""
        original_cols = [
            "date",
            "time",
            "sym",
            "close",
            "amount_delta",
            "n_midprice",
            "n_bid1",
            "n_bsize1",
            "n_bid2",
            "n_bsize2",
            "n_bid3",
            "n_bsize3",
            "n_bid4",
            "n_bsize4",
            "n_bid5",
            "n_bsize5",
            "n_ask1",
            "n_asize1",
            "n_ask2",
            "n_asize2",
            "n_ask3",
            "n_asize3",
            "n_ask4",
            "n_asize4",
            "n_ask5",
            "n_asize5",
            "label_5",
            "label_10",
            "label_20",
            "label_40",
            "label_60",
        ]
        return [col for col in original_cols if col in df.columns]

    def get_new_feature_columns(self, df):
        """获取新创建的特征列"""
        original_cols = set(self.get_original_columns(df))
        all_cols = set(df.columns)
        new_feature_cols = list(all_cols - original_cols)
        return sorted(new_feature_cols)

    def prepare_for_training(self, df, target_label="label5"):
        """准备训练数据"""
        print("准备训练数据...")

        # 1. 分离特征和标签
        original_cols = self.get_original_columns(df)
        feature_cols = [
            col for col in df.columns if col not in original_cols or col == "n_midprice"
        ]

        # 移除标签列和其他不需要的特征
        label_cols = [f"label{N}" for N in [5, 10, 20, 40, 60]]
        feature_cols = [col for col in feature_cols if col not in label_cols]

        # 2. 处理缺失值
        # 前向填充，然后后向填充
        df[feature_cols] = (
            df[feature_cols].fillna(method="ffill").fillna(method="bfill")
        )

        # 3. 去除仍然有缺失值的行
        df = df.dropna(subset=feature_cols + [target_label])

        # 4. 获取特征和标签
        X = df[feature_cols].values
        y = df[target_label].values

        print(f"训练数据形状: X={X.shape}, y={y.shape}")
        print(f"特征数量: {len(feature_cols)}")

        return X, y, feature_cols

    def create_sequence_features(self, X, seq_length):
        """创建序列特征（用于LSTM）"""
        print(f"创建序列特征，序列长度: {seq_length}")

        X_sequences = []
        n_samples = len(X)

        for i in range(seq_length, n_samples):
            sequence = X[i - seq_length : i]
            X_sequences.append(sequence)

        return np.array(X_sequences)


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
            Dropout(0.2),
            LSTM(256, activation="tanh", return_sequences=False),
            Dropout(0.2),
            Dense(128, activation="tanh"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2),
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


########################################################################################################################################
# 4. 主函数
def main():
    # file_path = "./merged_data/merged_0.csv"

    # # 加载数据
    # df = load_data(file_path)
    # print(f"数据形状: {df.shape}")
    # print(f"列名: {df.columns.tolist()}")
    # print(df.head())
    #
    # # Visualize the data
    # # Plot the closing price over time
    # plt.figure(figsize=(14, 7))
    # plt.plot(df["n_midprice"], label="Midprice")
    # plt.title("Midprice Over Time")
    # plt.xlabel("Date")
    # plt.ylabel("Price")
    # plt.legend()
    # plt.grid(True)

    # plt.show()

    # # 预处理数据
    # print("Preprocessing Data...")
    # feature_engineer = MarketDataFeatureEngineer(seq_length=100, target_label="label5")
    # df_with_features = feature_engineer.create_all_features(df).fillna(method="ffill")
    # print("Preprocessing Data...Done")
    # print("Saving...")
    # df_with_features.to_csv("df_with_features.csv")
    feature_columns = [
        # "date",
        # "time",
        # "sym",
        "n_close",
        "amount_delta",
        "n_midprice",
        "n_bid1",
        "n_bsize1",
        "n_bid2",
        "n_bsize2",
        "n_bid3",
        "n_bsize3",
        "n_bid4",
        "n_bsize4",
        "n_bid5",
        "n_bsize5",
        "n_ask1",
        "n_asize1",
        "n_ask2",
        "n_asize2",
        "n_ask3",
        "n_asize3",
        "n_ask4",
        "n_asize4",
        "n_ask5",
        "n_asize5",
        "file_sym",
        "file_date",
        "file_session",
        "source_file",
        "midprice_return",
        "close_return",
        "midprice_log_return",
        "midprice_momentum_1",
        "midprice_momentum_5",
        "midprice_momentum_20",
        "midprice_momentum_50",
        "price_acceleration",
        "midprice_ma",
        "midprice_std",
        "midprice_zscore",
        "bollinger_upper",
        "bollinger_lower",
        "bollinger_width",
        "bollinger_position",
        "bid_ask_spread",
        "size_imbalance_1",
        "size_imbalance_2",
        "size_imbalance_3",
        "size_imbalance_4",
        "size_imbalance_5",
        "weighted_midprice",
        "microprice",
        "bid_depth",
        "ask_depth",
        "total_depth",
        "depth_imbalance",
        "bid_slope_2",
        "ask_slope_2",
        "bid_slope_3",
        "ask_slope_3",
        "bid_slope_4",
        "ask_slope_4",
        "bid_slope_5",
        "ask_slope_5",
        "buy_pressure",
        "sell_pressure",
        "rsi_6",
        "rsi_14",
        "rsi_24",
        "macd",
        "macd_signal",
        "macd_histogram",
        "ma_5",
        "price_vs_ma_5",
        "ma_10",
        "price_vs_ma_10",
        "ma_20",
        "price_vs_ma_20",
        "ma_30",
        "price_vs_ma_30",
        "ma_50",
        "price_vs_ma_50",
        "ma_cross_5_20",
        "ma_cross_10_30",
        "stochastic_k",
        "stochastic_d",
        "bias_5",
        "bias_10",
        "bias_20",
        "bias_30",
        "bias_50",
        "return_mean_5",
        "return_std_5",
        "return_skew_5",
        "return_kurtosis_5",
        "return_q25_5",
        "return_q75_5",
        "return_iqr_5",
        "price_max_5",
        "price_min_5",
        "price_range_5",
        "price_range_pct_5",
        "return_mean_10",
        "return_std_10",
        "return_skew_10",
        "return_kurtosis_10",
        "return_q25_10",
        "return_q75_10",
        "return_iqr_10",
        "price_max_10",
        "price_min_10",
        "price_range_10",
        "price_range_pct_10",
        "return_mean_20",
        "return_std_20",
        "return_skew_20",
        "return_kurtosis_20",
        "return_q25_20",
        "return_q75_20",
        "return_iqr_20",
        "price_max_20",
        "price_min_20",
        "price_range_20",
        "price_range_pct_20",
        "return_mean_30",
        "return_std_30",
        "return_skew_30",
        "return_kurtosis_30",
        "return_q25_30",
        "return_q75_30",
        "return_iqr_30",
        "price_max_30",
        "price_min_30",
        "price_range_30",
        "price_range_pct_30",
        "return_mean_50",
        "return_std_50",
        "return_skew_50",
        "return_kurtosis_50",
        "return_q25_50",
        "return_q75_50",
        "return_iqr_50",
        "price_max_50",
        "price_min_50",
        "price_range_50",
        "price_range_pct_50",
        "return_autocorr_lag1",
        "return_autocorr_lag2",
        "return_autocorr_lag3",
        "return_autocorr_lag4",
        "return_autocorr_lag5",
        "return_autocorr_lag6",
        "return_autocorr_lag7",
        "return_autocorr_lag8",
        "return_autocorr_lag9",
        "return_autocorr_lag10",
        "volatility_5",
        "parkinson_vol_5",
        "volatility_10",
        "parkinson_vol_10",
        "volatility_20",
        "parkinson_vol_20",
        "volatility_30",
        "parkinson_vol_30",
        "volatility_50",
        "parkinson_vol_50",
        "volatility_ratio_5_20",
        "volatility_ratio_10_30",
        "volatility_change",
        "realized_volatility",
        "volume_price_corr_5",
        "volume_price_corr_10",
        "volume_price_corr_20",
        "volume_price_corr_30",
        "volume_price_corr_50",
        "vwap",
        "price_vwap_ratio",
        "volume_momentum",
        "volume_acceleration",
        "volume_ma_5",
        "volume_ratio_5",
        "volume_ma_10",
        "volume_ratio_10",
        "volume_ma_20",
        "volume_ratio_20",
        "volume_ma_30",
        "volume_ratio_30",
        "volume_ma_50",
        "volume_ratio_50",
        "price_up_volume_down",
        "price_down_volume_up",
        "order_flow_imbalance",
        "hour",
        "minute",
        "second",
        "is_morning",
        "is_afternoon",
        "is_opening",
        "is_closing",
        "time_sin",
        "time_cos",
        "time_interval",
        "is_irregular_time",
        "spread_depth_interaction",
        "volatility_momentum_interaction",
        "midprice_squared",
        "midprice_cubed",
        "volume_squared",
        "price_volume_ratio",
        "spread_depth_ratio",
        "cumulative_return",
        "cumulative_volume",
        "price_vs_vwap_diff",
        "price_vs_microprice_diff",
        "normalized_volume",
        "normalized_spread",
    ]
    df_with_features = load_data("./df_with_features.csv")
    df_with_features = df_with_features.tail(len(df_with_features) - 51)
    features = df_with_features[feature_columns]
    # print(features.head())
    # return
    sequence_length = 100
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i : i + sequence_length])
        y.append(df_with_features.iloc[i + sequence_length]["label_5"])
    print("Preprocessing Data...Done")

    X = np.array(X)
    y = np.array(y)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # 时间序列不随机打乱
    )

    print(f"训练集形状: {X_train.shape}, {y_train.shape}")
    print(f"测试集形状: {X_test.shape}, {y_test.shape}")

    X_train, y_train = apply_undersampling(X_train, y_train)
    # 构建模型
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    model.summary()

    # 训练模型
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        verbose=1,
    )

    # 预测示例
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    test_score = calculate_f_beta_multiclass(y_test, y_pred)
    print(f"The f beta score on test: {test_score}")

    draw_loss_curve(history)
    # 保存模型
    # model.save('lstm_price_prediction_model.h5')
    # print("模型已保存为 'lstm_price_prediction_model.h5'")


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


if __name__ == "__main__":
    # 运行主函数
    main()

    # 如果需要处理多个文件
    # files = ['snapshot_sym1_date1_am.csv', 'snapshot_sym1_date1_pm.csv']
    # X_all, y_all = process_multiple_files(files)
