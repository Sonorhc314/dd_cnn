from preprocessing import *

class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
          layers.Flatten(),
          layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
          layers.Dense(9*23, activation='sigmoid'),
          layers.Reshape((9, 23))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder(latent_dim)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(X_train, y_train,
                epochs=10,
                shuffle=True,
                validation_data=(X_test, y_test))

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)