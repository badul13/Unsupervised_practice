from keras import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU, Add
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow version:", tf.__version__)
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print("Is GPU available: ", tf.test.is_gpu_available())
    except RuntimeError as e:
        print(e)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1)
x_train = x_train / 127.5 - 1

encoder_input = Input(shape=(28, 28, 1))

# Encoder with residual connections
def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = LeakyReLU()(x)
    return x

x = Conv2D(32, 3, padding='same')(encoder_input)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = residual_block(x, 32)

x = Conv2D(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = residual_block(x, 64)

x = Conv2D(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = residual_block(x, 64)

x = Flatten()(x)
encoder_output = Dense(2)(x)

encoder = Model(encoder_input, encoder_output)
encoder.summary()

decoder_input = Input(shape=(2,))
x = Dense(7*7*64)(decoder_input)
x = Reshape((7, 7, 64))(x)

x = Conv2DTranspose(64, 3, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = residual_block(x, 64)

x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = residual_block(x, 64)

x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = residual_block(x, 64)

x = Conv2DTranspose(32, 3, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = residual_block(x, 32)

decoder_output = Conv2DTranspose(1, 3, strides=1, padding='same', activation='tanh')(x)

decoder = Model(decoder_input, decoder_output)
decoder.summary()

LEARNING_RATE = 0.001
BATCH_SIZE = 32

encoder_in = Input(shape=(28, 28, 1))
x = encoder(encoder_in)
decoder_out = decoder(x)

auto_encoder = Model(encoder_in, decoder_out)
auto_encoder.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss=tf.keras.losses.MeanSquaredError())

checkpoint_path = 'imcheckpoint.weights.h5'
checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=True, monitor='loss', verbose=1)

auto_encoder.fit(x_train, x_train, batch_size=BATCH_SIZE, epochs=40, callbacks=[checkpoint])

auto_encoder.load_weights(checkpoint_path)

import matplotlib.pyplot as plt

# MNIST 이미지에 대하여 x, y 좌표로 뽑아냅니다.
xy = encoder.predict(x_train)

plt.figure(figsize=(15, 12))
plt.scatter(x=xy[:, 0], y=xy[:, 1], c=y_train, cmap=plt.get_cmap('Paired'), s=3)
plt.colorbar()
plt.show()

decoded_images = auto_encoder.predict(x_train)
fig, axes = plt.subplots(3, 5)
fig.set_size_inches(12, 6)
for i in range(15):
    axes[i//5, i%5].imshow(x_train[i].reshape(28, 28), cmap='gray')
    axes[i//5, i%5].axis('off')
plt.tight_layout()
plt.title('Original Images')
plt.show()

fig, axes = plt.subplots(3, 5)
fig.set_size_inches(12, 6)
for i in range(15):
    axes[i//5, i%5].imshow(decoded_images[i].reshape(28, 28), cmap='gray')
    axes[i//5, i%5].axis('off')
plt.tight_layout()
plt.title('Auto Encoder Images')
plt.show()
