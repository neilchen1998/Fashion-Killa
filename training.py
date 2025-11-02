import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import math

IMG_HEIGHT = 28
IMG_WIDTH = 28

BATCH_SIZE = 128    # large batch size reduces overfitting
BUFFER_SIZE = 7000  # large buffer size helps shuffle randomly but may cause high memory usage
EPOCHS = 30         # number of iterations that allows the model to refine itself

def create_model():
    """Creates a model"""
    model = tf.keras.Sequential([
        # Input layer
        keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),

        # First convolution block - increased filters
        keras.layers.Conv2D(64, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(64, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        # Second convolution block
        keras.layers.Conv2D(128, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(128, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        # Third convolution block
        keras.layers.Conv2D(256, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(256, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        # Flatten and dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model

def create_pipeline():
    """Creates a pipeline for preprocessing"""
    pipeline = tf.keras.Sequential([

        # The input layer
        keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),

        # The effects
        keras.layers.RandomRotation(factor=0.02),
        keras.layers.RandomZoom(
            height_factor=0.1, # people often zoom out to look taller
            width_factor=-0.1,  # people often zoom in to look slimmer
        ),
        keras.layers.RandomTranslation(
            height_factor=0.1,
            width_factor=0.1,
            fill_mode='nearest' # the input is extended to the nearest pixel
        ),
        keras.layers.RandomFlip("horizontal"),  # pictures may be mirrored
        keras.layers.RandomBrightness(factor=0.15),  # lighting is different from picture to picture
        keras.layers.RandomContrast(0.1),
    ])

    return pipeline

def preprocessing(img, label, pipeline):
    """Preprocesses the images with the given pipeline

    Keyword arguments:
    img -- image
    label -- label
    pipeline -- the pipeline in Keras sequential model format
    """

    img = tf.cast(img, dtype=tf.float32)

    img = pipeline(img)

    img = img / 255.0

    return img, label

def exponetial_decay(epoch, lr):
    """Decreases the learning rate exponentially after 10 epochs

    Keyword arguments:
    epoch -- epoch
    lr -- the learning rate
    """
    if epoch < 10:
        return lr
    else:
        return lr * math.exp(-0.1)

if __name__ == "__main__":

    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

    # Split the train data into train & validation data
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels,
        test_size=0.15,
        random_state=41
    )

    # Reshape the images
    train_images = train_images.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    val_images = val_images.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    test_images = test_images.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)

    # Create a pipeline for preprocessing
    pipeline = create_pipeline()

    # Normalize the images
    val_images = val_images / 255.0
    test_images = test_images / 255.0

    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)) \
        .shuffle(BUFFER_SIZE) \
        .map(
            lambda img, label: preprocessing(img, label, pipeline)  # preprocess using the pipeline specified earlier
            , num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(BATCH_SIZE) \
        .prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # Create a basic model instance
    model = create_model()

    # # Display the model's architecture
    # model.summary()

    # Create checkpoint callback
    checkpoint_filepath = './best_model.keras'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    # Create early stopping callback
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,         # stops when there is no improvement after this number of epochs
        start_from_epoch=5, # number of epochs to wait before monitoring
    )

    # Create reduce learning rate callback
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
        factor=0.1,     # the discount learning rate factor
        patience=10,    # number of epochs with no improvement after which learning rate will be reduced
        min_lr=1e-5
    )

    # Learning rate scheduler
    scheduler = keras.callbacks.LearningRateScheduler(
        # declay exponentially after 10 epochs
        exponetial_decay,
        verbose=1
    )

    # Train the model
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr, scheduler]
    )

    # Evaluate the accuracy of the model
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")

    print(f"Number of epochs ran: {len(history.history['val_loss'])}")

    # Save the parameters of the model
    if accuracy > 0.9:
        model.save('my_model.keras')
        print(f"The accuracy rate is: {accuracy:.4f} and the parameters of the model is saved.\n")
