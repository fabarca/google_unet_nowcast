import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


# U-Net architecture inspired by "Machine Learning for Precipitation Nowcasting from Radar Images".
# https://ai.googleblog.com/2020/01/using-machine-learning-to-nowcast.html
# https://arxiv.org/abs/1912.12132


def basic_block(input_layer, n_channels):
    out_layer = layers.Conv2D(n_channels, 3, padding="same")(input_layer)
    out_layer = layers.BatchNormalization()(out_layer)
    out_layer = layers.LeakyReLU(alpha=0.1)(out_layer)
    out_layer = layers.Conv2D(n_channels, 3, padding="same")(out_layer)

    residual = layers.Conv2D(n_channels, 1, padding="same")(input_layer)  # short_skip
    out_layer = layers.add([out_layer, residual])  # Add back residual

    return out_layer


def down_block(input_layer, n_channels):
    residual = layers.Conv2D(n_channels, 1, strides=2, padding="same")(input_layer)  # short_skip

    out_layer = layers.BatchNormalization()(input_layer)
    out_layer = layers.LeakyReLU(alpha=0.1)(out_layer)
    out_layer = layers.MaxPooling2D(3, strides=2, padding="same")(out_layer)

    out_layer = layers.BatchNormalization()(out_layer)
    out_layer = layers.LeakyReLU(alpha=0.1)(out_layer)
    out_layer = layers.Conv2D(n_channels, 3, padding="same")(out_layer)

    long_skip = out_layer
    out_layer = layers.add([out_layer, residual])  # Add back residual

    return out_layer, long_skip


def up_block(input_layer, long_skip, n_channels):

    out_layer = layers.concatenate([input_layer, long_skip], axis=-1)

    residual = layers.Conv2DTranspose(n_channels, (2, 2), strides=(2, 2), padding='same')(out_layer)  # short_skip

    out_layer = layers.UpSampling2D(2)(out_layer)
    out_layer = layers.BatchNormalization()(out_layer)
    out_layer = layers.LeakyReLU(alpha=0.1)(out_layer)
    out_layer = layers.Conv2D(n_channels, 3, padding="same")(out_layer)
    out_layer = layers.BatchNormalization()(out_layer)
    out_layer = layers.LeakyReLU(alpha=0.1)(out_layer)
    out_layer = layers.Conv2D(n_channels, 3, padding="same")(out_layer)

    out_layer = layers.add([out_layer, residual])  # Add back residual

    return out_layer


def get_model(height, width, channels, out_channels, is_classification=True):
    inputs = layers.Input(shape=(height, width, channels))

    basicx1 = basic_block(inputs, 32)

    downx2, downx2_skip = down_block(basicx1, 32)
    downx4, downx4_skip = down_block(downx2, 64)
    downx8, downx8_skip = down_block(downx4, 128)
    downx16, downx16_skip = down_block(downx8, 256)
    downx32, downx32_skip = down_block(downx16, 512)
    downx64, downx64_skip = down_block(downx32, 512)
    downx128, downx128_skip = down_block(downx64, 1024)

    centerx128 = basic_block(downx128, 1024)

    upx64 = up_block(centerx128, downx128_skip, 1024)
    upx32 = up_block(upx64, downx64_skip, 512)
    upx16 = up_block(upx32, downx32_skip, 512)
    upx8 = up_block(upx16, downx16_skip, 256)
    upx4 = up_block(upx8, downx8_skip, 128)
    upx2 = up_block(upx4, downx4_skip, 64)
    upx1 = up_block(upx2, downx2_skip, 32)

    if is_classification:
        outputs = layers.Conv2D(out_channels, 3, activation="softmax", padding="same")(upx1)
    else:
        outputs = layers.Conv2D(out_channels, 3, activation="linear", padding="same")(upx1)

    model = models.Model(inputs, outputs)

    return model

# Free up RAM in case the model definition cells were run multiple times
tf.keras.backend.clear_session()

# Build model
model = get_model(256, 256, 16, 4, is_classification=True)
model.summary(line_length=120)
