import plotter as pt
import model as md
import data_preprocess as dp
import evaluation as eval
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping


def print_memory_usage(label=""):
    """Print current memory usage for monitoring"""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"{label} - Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")


def main(time_delay=5):
    print("=" * 101)
    print("~" * 40 + f" Time delay = {time_delay} " + "~" * 40)
    print("=" * 101)

    print("Data Preprocessing ...")
    sequence_length = 80

    print_memory_usage("Initial")

    # Initialize empty arrays for incremental concatenation
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    for i in range(9):

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
        df_with_features = df_with_features.head(len(df_with_features) - time_delay)
        df_with_features = df_with_features.tail(len(df_with_features) - 20)

        # print(eval.check_feature_distributions(df_with_features, dp.selected_features))

        X_single, y_single = dp.sequentialize_certain_features(
            df_with_features,
            dp.selected_features,
            f"label_{time_delay}",
            sequence_length,
        )
        y_single_code = eval.label_to_double_one_hot(y_single)
        # print_memory_usage(f"After sequentializing stock {i}")

        # (X_train_single, X_test_single, y_train_single, y_test_single) = dp.split(
        #     X_single, y_single, test_size=0.2
        # )
        X_train_single = X_single
        y_train_single = y_single_code
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
    y_test_code = eval.label_to_double_one_hot(y_test)
    print("-" * 50)

    X_train, X_test = dp.scale(X_train, X_test)
    print_memory_usage("Final")
    print(f"训练集形状: {X_train.shape}, {y_train.shape}")
    print(f"测试集形状: {X_test.shape}, {y_test.shape}")

    print("=" * 100)
    print("Data Preprocessing ... Done.")

    # 构建模型
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = md.build_classification_model(input_shape, 2)
    model.summary()
    early_stopping = EarlyStopping(
        monitor="val_loss",  # 监控验证集损失
        patience=2,  # 容忍多少个epoch没有改善
        restore_best_weights=True,  # 恢复最佳权重
        mode="min",  # 最小化指标
        verbose=1,
    )
    # 训练模型
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test_code),
        epochs=10,
        batch_size=1024,
        callbacks=[early_stopping],
    )

    # 预测示例
    y_pred = model.predict(X_test)
    # test_evidiential_model(y_test, y_pred, df_with_features_9, time_delay)
    test_classification_model(y_test, y_pred, df_with_features_9, time_delay)
    # 保存模型
    model.save_weights(f"onehot_model_{time_delay}.weights.h5")
    print(f"模型已保存为 'onehot_model_{time_delay}.weights.h5'")


def test_evidiential_model(y_true, y_pred, df, time_delay):
    print(y_pred[:, :10])
    probs, uncertainty = eval.get_uncertainty(y_pred_alpha=y_pred)
    print(f"Probs: {probs[:10]}")
    print(f"uncertainty: {uncertainty[:10]}")
    y_pred_l = eval.get_label_with_uncertainty(y_pred, 0.1)
    test_score = eval.calculate_f_beta_multiclass(y_true, y_pred_l)
    # test_score_custom = eval.calculate_f_beta_multiclass(y_test, y_pred_custom)
    test_pnl_average = eval.calculate_pnl_average(df, y_pred_l, time_delay)
    print(f"The f beta score on test(of Threshold {0.1}): {test_score}")
    # print(f"The f beta score on test(custom): {test_score_custom}")
    print(f"The pnl average on test(of Threshold {0.1}): {test_pnl_average}")


def test_classification_model(y_true, y_pred, df, time_delay):
    for i in range(10):
        print("-" * 50)
        thres = 0.3 + 0.04 * i
        print(f"Threshold = {thres}")
        y_pred_l = eval.double_one_hot_to_label(y_pred, threshold=thres)
        # print(y_pred_l.shape)

        test_score = eval.calculate_f_beta_multiclass(y_true, y_pred_l)
        # test_score_custom = eval.calculate_f_beta_multiclass(y_test, y_pred_custom)
        test_pnl_average = eval.calculate_pnl_average(df, y_pred_l, time_delay)
        print(f"The f beta score on test(of Threshold {thres}): {test_score}")
        # print(f"The f beta score on test(custom): {test_score_custom}")
        print(f"The pnl average on test(of Threshold {thres}): {test_pnl_average}")


if __name__ == "__main__":
    for td in [20]:
        main(td)
