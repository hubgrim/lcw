import keras
import tensorflow as tf
from sklearn.datasets import fetch_openml

K = tf.keras.backend
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
X_train, X_valid, y_train, y_valid = X[:60000], X[60000:], y[:60000], y[60000:]

class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean
    
class Exp_layer(keras.layers.Layer):
    def call(self, x):
        return K.exp(x)

class Square_layer(keras.layers.Layer):
    def call(self, x):
        return K.square(x)
    
class Sum_layer(keras.layers.Layer):
    def call(self, x, axis):
        return K.sum(x, axis=axis)

class Mean_layer(keras.layers.Layer):
    def call(self, x):
        return K.mean(x)
    
class Vae_loss(keras.layers.Layer):
    def call(self, x, codings_log_var, codings_mean):
        latent_loss = -0.5 * Sum_layer()(1 + codings_log_var - Exp_layer()(codings_log_var) - Square_layer()(codings_mean), axis=-1)
        self.add_loss(Mean_layer()(latent_loss) / 784.)
        return x
    

codings_size = 10
inputs = keras.layers.Input(shape=[28, 28])
z = keras.layers.Flatten()(inputs)
z = keras.layers.Dense(150, activation="selu")(z)
z = keras.layers.Dense(100, activation="selu")(z)
codings_mean = keras.layers.Dense(codings_size)(z) # μ
codings_log_var = keras.layers.Dense(codings_size)(z) # γ
codings = Sampling()([codings_mean, codings_log_var])
variational_encoder = keras.Model(inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])

decoder_inputs = keras.layers.Input(shape=[codings_size])
x = keras.layers.Dense(100, activation="selu")(decoder_inputs)
x = keras.layers.Dense(150, activation="selu")(x)
x = keras.layers.Dense(28 * 28, activation="sigmoid")(x)
x = Vae_loss()(x, codings_log_var, codings_mean)
outputs = keras.layers.Reshape([28, 28])(x)
variational_decoder = keras.Model(inputs=[decoder_inputs], outputs=[outputs])

_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = keras.Model(inputs=[inputs], outputs=reconstructions)

# latent_loss = -0.5 * Sum_layer()(1 + codings_log_var - Exp_layer()(codings_log_var) - Square_layer()(codings_mean), axis=-1)
# variational_ae.add_loss(Mean_layer()(latent_loss) / 784.)
variational_ae.compile(loss="binary_crossentropy", optimizer="rmsprop")

history = variational_ae.fit(X_train, X_train, epochs=50, batch_size=128, validation_data=[X_valid, X_valid])

codings = tf.random.normal(shape=[12, codings_size])
images = variational_decoder(codings).numpy()