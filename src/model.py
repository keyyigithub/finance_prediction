import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Conv1D,
    Layer,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


# Define the Attention layer
@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = K.sum(context, axis=1)
        return context


# Returns basic model without compiling
def build_base_model(input_shape):
    model = Sequential(
        [
            # Conv1D(filters=128, kernel_size=3, padding="same", activation="tanh"),
            # Dropout(0.3),
            # Conv1D(filters=128, kernel_size=3, padding="same", activation="tanh"),
            # Dropout(0.3),
            LSTM(
                256,
                activation="tanh",
                return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=l2(0.01),
            ),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(
                256,
                activation="tanh",
                return_sequences=False,
                kernel_regularizer=l2(0.01),
            ),
            BatchNormalization(),
            Dropout(0.3),
            # Attention(),
            # BatchNormalization(),
        ]
    )
    return model


# Get classification model with compiling
def build_classification_model(input_shape, num_classes=3):
    model = Sequential(
        [
            build_base_model(input_shape),
            # Dense(256, activation="relu"),
            # Dropout(0.3),
            Dense(128, activation="tanh", kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),  # 3个类别：0,1,2
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
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
