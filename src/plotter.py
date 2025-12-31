import numpy as np
import matplotlib.pyplot as plt


def plot_predict_curve(y_true, y_pred, label: str = "pnl"):
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label="True Curve")
    plt.plot(y_pred, label="Prediction Curve")
    plt.title("Prediction v.s. Truth")
    plt.xlabel("Ticks")
    plt.ylabel(f"{label}")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"predict_curve_of_{label}.svg")
    print(f"The figure saved to predict_curve_of_{label}.svg")
    plt.show()


def draw_loss_curve(history):
    history_dict = history.history
    # 创建图表
    plt.figure(figsize=(10, 10))

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


def draw_accuracy_curve(history):
    history_dict = history.history
    # 创建图表
    plt.figure(figsize=(10, 10))

    # 1. 绘制损失曲线
    plt.plot(history_dict["accuracy"], label="Training Accuracy")
    if "val_loss" in history_dict:
        plt.plot(history_dict["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_data(df):
    # Visualize the data
    # Plot the closing price over time
    plt.figure(figsize=(14, 7))
    plt.plot(df["n_midprice"], label="Midprice")
    plt.title("Midprice Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    plt.show()
