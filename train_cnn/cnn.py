import train_dense_nn

class Denoise(Model): #cnn
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(9, 23, 1)),
            
            #lower level features - 
            #middle layers encode features, replace input and output to right sizes
            layers.Conv2D(25, (3, 3), activation='relu', padding='same', strides=1),
            layers.Conv2D(5, (3, 3), activation='relu', padding='same', strides=1),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(5, kernel_size=3, strides=1, activation='sigmoid', padding='same'),
            layers.Conv2DTranspose(25, kernel_size=3, strides=1, activation='sigmoid', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder_conv = Denoise()
autoencoder_conv.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder_conv.fit(X_train, y_train,
                epochs=1000,
                shuffle=True,
                validation_data=(X_test, y_test))

encoded_imgs = autoencoder_conv.encoder(X_test).numpy()
decoded_imgs = autoencoder_conv.decoder(encoded_imgs).numpy()

#print(len(encoded_imgs))
print(X_test)
print(X_test[0].shape)
print(decoded_imgs[0].shape)
#print(decoded_imgs[1])
plt.imshow(decoded_imgs[1])