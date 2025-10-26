import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

BATCH_SIZE = 128
EPOCHS = 15

def create_model():
    model = tf.keras.Sequential([

        # First convolution block
        keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),

        # Second convolution block
        keras.layers.Conv2D(64, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),

        # Third convolution block
        keras.layers.Conv2D(128, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),

        # Fourth convolution block
        keras.layers.Conv2D(256, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),

        # Flatten and dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model

def create_gen():

    gen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,  # degrees
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True, # pictures may be mirrored
    )

    return gen

if __name__ == "__main__":

    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

    # Split the train data into train & validation data
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels,
        test_size=0.15,
        random_state=41
    )

    train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
    val_images = val_images.reshape(-1, 28, 28, 1) / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

    # Create a basic model instance
    model = create_model()

    # Display the model's architecture
    model.summary()

    # Create a generator that applies rotation effect, shift effect, etc.
    gen = create_gen()

    checkpoint_filepath = './best_model.keras'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    model.fit(
        gen.flow(train_images, train_labels, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(val_images, val_labels),
        callbacks=[checkpoint_callback]
    )

    # Save the parameters of the model
    model.save('my_model.keras')

    # Evaluate the accuracy of the model
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
