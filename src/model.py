import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Conv1D,
    LayerNormalization,
    Concatenate,
    MaxPooling1D,
    MultiHeadAttention,
    Flatten,
    Bidirectional,
)
from tensorflow.keras.regularizers import l2


def build_conv_residual_block(input_shape):
    inputs = keras.Input(input_shape)
    shortcut = inputs
    feat_1 = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(
        inputs
    )
    feat_2 = Conv1D(filters=32, kernel_size=5, padding="same", activation="relu")(
        inputs
    )
    feat_3 = Conv1D(filters=32, kernel_size=7, padding="same", activation="relu")(
        inputs
    )

    outputs = Concatenate(axis=2)([feat_1, feat_2, feat_3, shortcut])

    model = keras.Model(inputs, outputs)
    return model


def build_lstm_residual_block(input_shape, units=256):
    inputs = keras.Input(input_shape)

    shortcut = inputs

    x = Bidirectional(LSTM(units, return_sequences=True, kernel_regularizer=l2(0.01)))(
        inputs
    )
    x = Bidirectional(LSTM(units, return_sequences=True, kernel_regularizer=l2(0.01)))(
        inputs
    )
    shortcut_reshaped = Dense(units * 2)(shortcut)
    x = shortcut_reshaped + x
    outputs = Dropout(0.3)(x)

    model = keras.Model(inputs, outputs)
    return model


# Returns basic model without compiling
def build_base_model(input_shape):
    inputs = keras.Input(input_shape)
    x = build_conv_residual_block(input_shape)(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(pool_size=3)(x)
    # x = build_conv_residual_block(input_shape)(inputs)
    # x = LayerNormalization()(x)
    # x = Dropout(0.3)(x)
    # x = MaxPooling1D(pool_size=2)(x)

    x = build_lstm_residual_block((x.shape[1], x.shape[2]), units=128)(x)
    x = LayerNormalization()(x)
    # x = build_lstm_residual_block((x.shape[1], x.shape[2]))(x)
    # x = LayerNormalization()(x)

    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    short_cut = x
    attention_output_1 = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = LayerNormalization()(x + attention_output_1)
    # attention_output_2 = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    # x = LayerNormalization()(x + attention_output_2)

    x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
    short_cut = Dense(64)(short_cut)
    x = short_cut + x
    x = Dropout(0.3)(x)

    outputs = Flatten()(x)

    model = keras.Model(inputs, outputs)
    return model


# Get classification model with compiling
def build_classification_model(input_shape, num_classes=3):

    inputs = keras.Input(input_shape)
    x = build_base_model(input_shape)(inputs)
    x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        clipnorm=1.0,  # 梯度裁剪，防止梯度爆炸
    )
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        # metrics=[eval.FBetaScore()],
    )

    return model


# Get model that process continuous data like n_midprice
def build_continuous_model(input_shape):

    inputs = keras.Input(input_shape)
    x = build_base_model(input_shape)(inputs)
    x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="linear")(x)
    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        clipnorm=1.0,  # 梯度裁剪，防止梯度爆炸
    )
    model.compile(
        optimizer=optimizer,
        loss="mse",
    )

    return model


def new_mse(y_true, y_pred):
    # 使用TensorFlow的操作
    error = tf.sign(y_pred) * tf.math.log1p(tf.abs(y_pred)) - tf.sign(
        y_true
    ) * tf.math.log1p(tf.abs(y_true))
    loss_value = tf.square(error)
    return tf.reduce_mean(loss_value)


def build_evidential_model(input_shape):
    inputs = keras.Input(input_shape)

    # 特征提取（复用您现有的base_model）
    x = build_base_model(input_shape)(inputs)

    # 证据头
    x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)

    # 证据输出（2个正数）
    evidence = Dense(2, activation="softplus", name="evidence")(x)

    # Dirichlet参数 α = evidence + 1
    alpha = keras.layers.Lambda(lambda e: e + 1.0, name="alpha")(evidence)

    model = keras.Model(inputs=inputs, outputs=alpha)

    # 编译使用Dirichlet损失
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=dirichlet_loss,
    )
    return model


# 在损失函数中体现
def dirichlet_loss(y_true, y_pred_alpha):
    """
    y_true: 真实标签的one-hot编码
    y_pred_alpha: 预测的Dirichlet参数
    """
    # 防止数值不稳定
    y_pred_alpha = tf.clip_by_value(y_pred_alpha, 1e-8, 1e8)

    # 计算总强度S
    S = tf.reduce_sum(y_pred_alpha, axis=1, keepdims=True)

    # Dirichlet损失包含两部分：
    # 1. 拟合误差（类似交叉熵）
    error_loss = tf.reduce_sum((y_true - y_pred_alpha / S) ** 2, axis=1)

    # 2. 正则化项（惩罚过度自信）
    # 当S很大但预测错误时，这个惩罚会很大
    reg_loss = tf.reduce_sum(
        y_pred_alpha * (S - y_pred_alpha) / (S**2 * (S + 1)), axis=1
    )

    total_loss = error_loss + reg_loss
    return tf.reduce_mean(total_loss)
