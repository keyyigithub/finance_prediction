import evaluation as eval
import numpy as np
import tensorflow as tf
from tensorflow import keras
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
from tensorflow.keras import backend as K


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
    feat_4 = Conv1D(filters=32, kernel_size=11, padding="same", activation="relu")(
        inputs
    )
    outputs = Concatenate(axis=2)([feat_1, feat_2, feat_3, feat_4, shortcut])

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
    # x = build_conv_residual_block(input_shape)(inputs)
    # x = LayerNormalization()(x)
    # x = Dropout(0.3)(x)

    x = build_lstm_residual_block((x.shape[1], x.shape[2]), units=128)(x)
    x = LayerNormalization()(x)
    # x = build_lstm_residual_block((x.shape[1], x.shape[2]))(x)
    # x = LayerNormalization()(x)

    x = LSTM(128, return_sequences=True, dropout=0.3)(x)
    short_cut = x
    attention_output_1 = MultiHeadAttention(num_heads=4, key_dim=128)(x, x)
    x = LayerNormalization()(x + attention_output_1)
    # attention_output_2 = MultiHeadAttention(num_heads=4, key_dim=256)(x, x)
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
    model = Sequential(
        [
            build_base_model(input_shape),
            Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),  # 3个类别：0,1,2
        ]
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        clipnorm=1.0,  # 梯度裁剪，防止梯度爆炸
    )
    model.compile(
        optimizer=optimizer,
        loss="categorical_focal_crossentropy",
        # metrics=[eval.FBetaScore()],
    )

    return model


# Get model that process continuous data like n_midprice
def build_continuous_model(input_shape):
    model = Sequential(
        [
            build_base_model(input_shape),
            Dense(64, activation="tanh", kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(32, activation="tanh", kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(1, activation="tanh"),  # 3个类别：0,1,2
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse",
    )

    return model
