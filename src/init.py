from sklearn.utils import class_weight
import plotter as pt
import model as md
import data_preprocess as dp
import evaluation as eval
import pandas as pd
import numpy as np

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
    print("-" * 50)
    df_with_features = pd.read_csv("./df_with_features.csv")
    df_with_features = dp.add_midprice_label(df_with_features, 5)
    df_with_features = df_with_features.tail(len(df_with_features) - 51)
    df_with_features = df_with_features.head(len(df_with_features) - 10)

    sequence_length = 100
    time_delay = 5
    X, y = dp.sequentialize_certain_features(
        df_with_features, dp.selected_features, f"label_{time_delay}", sequence_length
    )
    X_train, X_test, y_train, y_test, price_scaler = dp.split_and_scale(
        X, y, test_size=0.2
    )
    print(f"训练集形状: {X_train.shape}, {y_train.shape}")
    print(f"测试集形状: {X_test.shape}, {y_test.shape}")

    # print(f"From return: {price_scaler.center_,price_scaler.scale_}")
    # print(f"训练集形状: {X_train.shape}, {y_train.shape}")
    # print(f"测试集形状: {X_test.shape}, {y_test.shape}")
    #
    # X_train_original = price_scaler.inverse_transform(
    #     X_train[:, 99, 0:3].reshape(-1, 3)
    # )
    # print(f"After Inverse: {price_scaler.center_,price_scaler.scale_}")
    # print(f"The shape of X_train_original: {X_train_original.shape}")
    # y_train = eval.get_label(y_train, X_train_original[:, 1], 5)
    # print(f"The first 20 (maybe) true labels: {y_train[:20]}")
    # print(f"The first 20 true labels: {df_with_features.iloc[100:120]['label_5']}")

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
        class_weight={0: 5, 1: 1, 2: 5},
    )

    # 预测示例
    y_pred = model.predict(X_test)
    # pt.plot_predict_curve(y_test, y_pred)
    y_pred = np.argmax(y_pred, axis=1)

    # X_test_original = price_scaler.inverse_transform(X_test[:, 99, 0:3].reshape(-1, 3))

    # y_pred = eval.get_label(y_pred, X_test_original[:, 1], 5)
    # y_test = eval.get_label(y_test, X_test_original[:, 1], 5)
    # print(f"The first 20 pred labels: {y_pred[:20]}")
    # print(f"The first 20 true labels: {y_test[:20]}")

    test_score = eval.calculate_f_beta_multiclass(y_test, y_pred)
    print(f"The f beta score on test: {test_score}")

    y_train_pred = model.predict(X_train)
    # pt.plot_predict_curve(y_train, y_train_pred)
    y_train_pred = np.argmax(y_train_pred, axis=1)
    # X_train_original = price_scaler.inverse_transform(
    #     X_train[:, 99, 0:3].reshape(-1, 3)
    # )

    # y_train_pred = eval.get_label(y_train_pred, X_train_original[:, 1], 5)
    # y_train = eval.get_label(y_train, X_train_original[:, 1], 5)

    train_score = eval.calculate_f_beta_multiclass(y_train, y_train_pred)
    print(f"The f beta score on train: {train_score}")

    # pt.draw_loss_curve(history)
    # pt.draw_accuracy_curve(history)
    # 保存模型
    # model.save('lstm_price_prediction_model.h5')
    # print("模型已保存为 'lstm_price_prediction_model.h5'")


if __name__ == "__main__":
    main()
