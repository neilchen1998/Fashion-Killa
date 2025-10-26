import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

IMG_HEIGHT = 28
IMG_WIDTH = 28

BATCH_SIZE = 128    # large batch size reduces overfitting
BUFFER_SIZE = 7000  # large buffer size helps shuffle randomly but may cause high memory usage
EPOCHS = 15         # number of iterations that allows the model to refine itself

def create_model():
    model = tf.keras.Sequential([

        # The input layer
        keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),

        # First convolution block
        keras.layers.Conv2D(32, (3, 3), padding='same'),
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

    # Normalize the images
    train_images = train_images.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1) / 255.0
    val_images = val_images.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1) / 255.0
    test_images = test_images.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1) / 255.0

    # Change the label type
    train_labels = train_labels.astype(np.int32)
    val_labels = val_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)

    # Create dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # Create a basic model instance
    model = create_model()

    # # Display the model's architecture
    # model.summary()

    # Create a generator that applies rotation effect, shift effect, etc.
    gen = create_gen()

    # Create checkpoint callback
    checkpoint_filepath = './best_model.keras'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    # Train the model
    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[checkpoint_callback]
    )

    # Save the parameters of the model
    model.save('my_model.keras')

    # Evaluate the accuracy of the model
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
