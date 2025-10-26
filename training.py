import tensorflow as tf
from tensorflow import keras

def create_model():
    model = tf.keras.Sequential([

        # The first input layer extracts local features (edges, shapes)
        keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=(28, 28, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),

        # The second layer extracts more intricate and detailed patterns
        keras.layers.Conv2D(128, (3, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(256, (3, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),

        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.25),

        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.25),

        # The output layer
        keras.layers.Dense(10, activation='softmax')
        ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model

if __name__ == "__main__":

    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

    print("Size of train_images: ", train_images.shape)

    train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

    train_labels = train_labels
    test_labels = test_labels

    # Create a basic model instance
    model = create_model()

    # Display the model's architecture
    model.summary()

    BATCH_SIZE = 64
    EPOCHS = 20

    model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

    # Save the parameters of the model
    model.save('my_model.keras')

    # Evaluate the accuracy of the model
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
