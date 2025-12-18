from numpy.typing import NDArray
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


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


def sequentialize_certain_features(
    df: pd.DataFrame, feature_columns: list[str], label_column: str, seq_length: int
):
    """创建序列特征（用于LSTM）"""
    print(f"Sequentializing features, sequence length: {seq_length}")

    X, y = [], []
    features = df[feature_columns]
    for i in range(len(features) - seq_length):
        X.append(features[i : i + seq_length])
        y.append(df.iloc[i + seq_length][label_column])

    print(f"Sequentializing features... done.")
    return np.array(X), np.array(y)


def split_and_scale(X: NDArray, y: NDArray, test_size=0.2):
    print(f"Splitting Data... Test size: {test_size}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    print("Splitting Data... Done.")

    print("Scaling data...")
    scaler = RobustScaler()
    original_shape_train = X_train.shape
    original_shape_test = X_test.shape

    # 重塑为2D用于缩放 (samples*timesteps, features)
    n_samples_train, n_timesteps, n_features = X_train.shape
    n_samples_test = X_test.shape[0]

    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)
    X_train_scaled_2d = scaler.fit_transform(X_train_2d)
    X_test_scaled_2d = scaler.transform(X_test_2d)

    # 重塑回3D (samples, timesteps, features)
    X_train_scaled = X_train_scaled_2d.reshape(original_shape_train)
    X_test_scaled = X_test_scaled_2d.reshape(original_shape_test)

    print("Scaling data... Done.")
    return X_train_scaled, X_test_scaled, y_train, y_test
