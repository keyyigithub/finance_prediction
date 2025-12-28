import plotter as pt
import model as md
import data_preprocess as dp
import evaluation as eval
import pandas as pd
import numpy as np


def print_memory_usage(label=""):
    """Print current memory usage for monitoring"""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"{label} - Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")


# ----------
# Next step:
# these features may be divided into several parts, including
# price, volume, technical, and volatility
# But it might take a fucking lot of time, for god's sake!


def main(time_delay = 5):
    print("=" * 100)
    print("~" * 100)
    print("=" * 100)
    print("Data Preprocessing ...")
    sequence_length = 80

    print_memory_usage("Initial")

    # Initialize empty arrays for incremental concatenation
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    for i in range(1):

        print("-" * 50)
        print(f"Stock Index: {i}")
        print_memory_usage(f"Before processing stock {i}")

        # Process one stock at a time to reduce memory footprint
        df_raw = pd.read_csv(f"./merged_data/merged_{i}.csv")
        # print_memory_usage(f"After loading stock {i}")

        df_with_features = dp.create_all_features(df_raw)
        # print_memory_usage(f"After feature engineering stock {i}")

        # Calculate the midprice return
        # df_with_features[f"return_after_{time_delay}"] = (
        #     df_with_features["n_midprice"].shift(-time_delay)
        #     - df_with_features["n_midprice"]
        # ) / (df_with_features["n_midprice"])
        df_with_features[f"midprice_after_{time_delay}"] = df_with_features[
            "n_midprice"
        ].shift(-time_delay)

        df_with_features[f"relabel_continue_{time_delay}"] = ( df_with_features[f"midprice_after_{time_delay}"] / df_with_features[f"n_midprice"] - 1 ) * 200

        df_with_features = df_with_features.tail(len(df_with_features) - 20)
        df_with_features = df_with_features.head(len(df_with_features) - time_delay)

        # print(eval.check_feature_distributions(df_with_features, dp.selected_features))

        X_single, y_single = dp.sequentialize_certain_features(
            df_with_features,
            dp.selected_features,
            f"relabel_continue_{time_delay}",
            sequence_length,
        )
        # print_memory_usage(f"After sequentializing stock {i}")

        # (X_train_single, X_test_single, y_train_single, y_test_single) = dp.split(
        #     X_single, y_single, test_size=0.2
        # )
        X_train_single = X_single
        y_train_single = y_single
        # print_memory_usage(f"After splitting data of stock {i}")

        # Incremental concatenation
        if X_train is None:
            X_train = X_train_single
            # X_test = X_test_single
            y_train = y_train_single
            # y_test = y_test_single
        else:
            X_train = np.concatenate([X_train, X_train_single], axis=0)
            # X_test = np.concatenate([X_test, X_test_single], axis=0)
            y_train = np.concatenate([y_train, y_train_single], axis=0)
            # y_test = np.concatenate([y_test, y_test_single], axis=0)

    print("=" * 100)
    df_raw_9 = pd.read_csv("./merged_data/merged_9.csv")
    df_with_features_9 = dp.create_all_features(df_raw_9)
    df_with_features_9[f"return_after_{time_delay}"] = (
        df_with_features_9["n_midprice"].shift(-time_delay)
        - df_with_features_9["n_midprice"]
    ) / df_with_features_9["n_midprice"]

    df_with_features_9 = df_with_features_9.tail(len(df_with_features_9) - 20)
    df_with_features_9 = df_with_features_9.head(len(df_with_features_9) - time_delay)
    X_test, y_test = dp.sequentialize_certain_features(
        df_with_features_9,
        dp.selected_features,
        f"label_{time_delay}",
        sequence_length,
    )
    print("-" * 50)

    X_train, X_test = dp.scale(X_train, X_test)
    print_memory_usage("Final")
    print(f"训练集形状: {X_train.shape}, {y_train.shape}")
    print(f"测试集形状: {X_test.shape}, {y_test.shape}")

    print("=" * 100)
    print("Data Preprocessing ... Done.")

    # 构建模型
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = md.build_continuous_model(input_shape)
    model.summary()

    # 训练模型
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=0,
        batch_size=1024,
                                               )

    # 预测示例
    y_pred = model.predict(X_test)
    
    for i in range(15):
        alpha_1 = 0.0005 +0.0001 * i
        alpha_2 = 0.001 +0.0001 * i
        y_pred_custom = eval.get_label(
                y_pred,
                np.zeros(shape=(len(y_pred))),
                time_delay,
                alpha1=alpha_1 * 200,
                alpha2=alpha_2 * 200,
        )
        y_pred_custom = np.asarray(y_pred_custom)
        # pt.plot_predict_curve(y_test, y_pred)
        # y_pred = np.argmax(y_pred, axis=1)

        # X_test_original = price_scaler.inverse_transform(X_test[:, 99, 0:3].reshape(-1, 3))

        # y_pred = eval.get_label(y_pred, X_test_original[:, 1], 5)
        # y_test = eval.get_label(y_test, X_test_original[:, 1], 5)
        # print(f"The first 20 pred labels: {y_pred[:20]}")
        # print(f"The first 20 true labels: {y_test[:20]}")

        test_score = eval.calculate_f_beta_multiclass(y_test, y_pred_custom)
        #test_score_custom = eval.calculate_f_beta_multiclass(y_test, y_pred_custom)
        #test_pnl_average = eval.calculate_pnl_average(
        #    df_with_features_9, y_pred, time_delay
        #)
        test_pnl_average_custom = eval.calculate_pnl_average(
            df_with_features_9, y_pred_custom, time_delay
        )
        print(f"The f beta score on test(default): {test_score}")
        # print(f"The f beta score on test(custom): {test_score_custom}")
        # print(f"The pnl average on test(default): {test_pnl_average}")
        print(f"The pnl average on test(custom): {test_pnl_average_custom}")
        if time_delay == 5 or time_delay == 10:
            print(f"alpha: {alpha_1}")
        else:
            print(f"alpha: {alpha_2}") 

    # y_train_pred = model.predict(X_train)
    # pt.plot_predict_curve(y_train, y_trai n_pred)
    # y_train_pred = np.argmax(y_train_pred, axis=1)
    # X_train_original = price_scaler.inverse_transform(
    #     X_train[:, 99, 0:3].reshape(-1, 3)
    # )
 # y_train_pred = eval.get_label(y_train_pred, X_train_original[:, 1], 5)
    # y_train = eval.get_label(y_train, X_train_original[:, 1], 5)

    # train_score = eval.calculate_f_beta_multiclass(y_train, y_train_pred)
    # print(f"The f beta score on train: {train_score}")

    # pt.draw_loss_curve(history)
    # pt.draw_accuracy_curve(history)
    # 保存模型
    model.save_weights(f"continue_model_{time_delay}.weights.h5")
    print(f"模型已保存为 'continue_model_{time_delay}.weights.h5'")


if __name__ == "__main__":
    for td in [5, 10, 20, 40, 60]:
        main(td)
