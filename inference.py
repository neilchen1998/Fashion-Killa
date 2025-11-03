# Copyright 2025 Neil Chen

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BATCH_SIZE = 64

def pick_and_plot_samples(n):
    """Plot the training history

    Keyword arguments:
    num -- number of samples
    """
    pass


if __name__ == "__main__":

    # Load dataset
    _, (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    num_test_samples = test_images.shape[0]

    print("Number of test_images: ", num_test_samples)

    test_images_scaled = test_images / 255.0

    new_model = tf.keras.models.load_model('best_model.keras')

    # Show the model architecture
    new_model.summary()

    # Evaluate the restored model
    loss, acc = new_model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE, verbose=2)
    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

    # Pick 5 indices randomly
    random_indices = np.random.choice(num_test_samples, size=5, replace=False)

    # Select the random samples
    random_images = test_images[random_indices]
    random_true_labels = test_labels[random_indices]
    random_images_scaled = test_images_scaled[random_indices]

    predictions = new_model.predict(random_images_scaled)
    predicted_labels = np.argmax(predictions, axis=1)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    plt.figure(figsize=(10, 5))

    plt.suptitle("Model Predictions vs. True Labels (5 Samples)")

    for i in range(5):
        # Create the subplot for the current image
        plt.subplot(1, 5, i + 1)

        # The title of each image
        plt.title(f"Img. {i + 1}", fontsize=12)

        # Disable ticks
        plt.xticks([])
        plt.yticks([])

        # Display the image
        plt.imshow(random_images[i], cmap=plt.cm.binary)

        # Get the true and predicted label
        true_label = random_true_labels[i]
        pred_label = predicted_labels[i]

        # Change the colour if the prediction is incorrect
        color = 'black'
        if pred_label != true_label:
            color = 'red'

        # Set the title to show the predicted label and the true label
        plt.xlabel(
            f"Pred: {class_names[pred_label]}\n"
            f"True: {class_names[true_label]}",
            color=color
        )

    plt.tight_layout()

    # Save the plot
    filename = "prediciton-vs-true.png"
    try:
        plt.savefig(filename)
        print(f"Plot successfully saved to {filename}.\n")
    except Exception as e:
        print(f"Error saving plot: {e}.\n")
