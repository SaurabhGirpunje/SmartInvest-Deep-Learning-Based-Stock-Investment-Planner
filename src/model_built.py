# src/model_built.py

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Input, Dropout
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# --------------------------
# Model builders
# --------------------------
# def build_lstm(window):
#     inp = Input(shape=(window, 1))
#     x = LSTM(64, return_sequences=True)(inp)
#     x = Dropout(0.2)(x)
#     x = LSTM(32)(x)
#     x = Dense(16, activation='relu')(x)
#     out = Dense(1)(x)
#     model = Model(inputs=inp, outputs=out)
#     model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
#                   loss=tf.keras.losses.Huber(delta=1))
#     return model


# def build_cnn(window):
#     inp = Input(shape=(window, 1))
#     x = Conv1D(32, 2, activation='relu', padding='same')(inp)
#     x = Dropout(0.2)(x)
#     x = MaxPooling1D(2, padding='same')(x)
#     x = Flatten()(x)
#     x = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(x)
#     out = Dense(1)(x)
#     model = Model(inputs=inp, outputs=out)
#     model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
#                   loss=tf.keras.losses.Huber(delta=1))
#     return model


# def build_hybrid(window):
#     inp = Input(shape=(window, 1))
#     x = Conv1D(32, 2, activation='relu', padding='same')(inp)
#     x = Dropout(0.2)(x)
#     x = LSTM(32, kernel_regularizer=l2(0.001))(x)
#     out = Dense(1)(x)
#     model = Model(inputs=inp, outputs=out)
#     model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
#                   loss=tf.keras.losses.Huber(delta=1))
#     return model


def build_lstm(window):
    inp = Input(shape=(window, 1))
    x = LSTM(128, return_sequences=True)(inp)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(1)(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.Huber(delta=5)
    )
    return model

def build_cnn(window):
    inp = Input(shape=(window, 1))
    x = Conv1D(64, 3, activation='relu', padding='same')(inp)
    x = Dropout(0.3)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(1)(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
        loss=tf.keras.losses.Huber(delta=5)
    )
    return model


def build_hybrid(window):
    inp = Input(shape=(window, 1))
    x = Conv1D(64, 3, activation='relu', padding='same')(inp)
    x = Dropout(0.3)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(1)(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.Huber(delta=5)
    )
    return model
