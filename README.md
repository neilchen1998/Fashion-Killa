# Fashion-MNIST

## Dataset

In this project, Fashion-MNIST is used.

## Models

* 3 Layers of Conv2D w/ Batch Normalization & Max Pooling + 3 Layers of Dense

```python
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
```

Test Accuracy: 0.8844
