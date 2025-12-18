from sklearn.utils import class_weight
import plotter as pt
import model as md
import data_preprocess as dp
import evaluation as eval
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

# ----------
# Next step:
# these features may be divided into several parts, including
# price, volume, technical, and volatility
# But it might take a fucking lot of time, for god's sake!

feature_columns = [
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
    "bid_ask_spread",
    # "size_imbalance_1",
    # "size_imbalance_2",
    # "size_imbalance_3",
    # "size_imbalance_4",
    # "size_imbalance_5",
    # "weighted_midprice",
    # "microprice",
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
    "time_sin",
    "time_cos",
]


def main():
    df_with_features = pd.read_csv("./df_with_features.csv")
    df_with_features = df_with_features.tail(len(df_with_features) - 51)
    sequence_length = 100
    X, y = dp.sequentialize_certain_features(
        df_with_features, feature_columns, "label_5", sequence_length
    )
    X_train, X_test, y_train, y_test = dp.split_and_scale(X, y)
    # 划分训练集和测试集
    # weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y)
    # weights_dict = dict(zip(np.array([0, 1, 2]), weights))
    #
    # print(f"Weights for each class: {weights_dict}")

    print(f"训练集形状: {X_train.shape}, {y_train.shape}")
    print(f"测试集形状: {X_test.shape}, {y_test.shape}")

    # X_train, y_train = apply_undersampling(X_train, y_train)
    # 构建模型
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = md.build_classification_model(input_shape)
    model.summary()

    # 训练模型
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=2,
        batch_size=128,
        verbose=1,
        class_weight={0: 4, 1: 1, 2: 4},
    )

    # 预测示例
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    test_score = eval.calculate_f_beta_multiclass(y_test, y_pred)
    print(f"The f beta score on test: {test_score}")

    y_train_pred = model.predict(X_train)
    y_train_pred = np.argmax(y_train_pred, axis=1)
    train_score = eval.calculate_f_beta_multiclass(y_train, y_train_pred)
    print(f"The f beta score on train: {train_score}")

    # pt.draw_loss_curve(history)
    # pt.draw_accuracy_curve(history)
    # 保存模型
    # model.save('lstm_price_prediction_model.h5')
    # print("模型已保存为 'lstm_price_prediction_model.h5'")


if __name__ == "__main__":
    main()
