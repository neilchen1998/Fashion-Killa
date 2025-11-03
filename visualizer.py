import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_training_data(history):
    """Plot the training history

    Keyword arguments:
    history -- the history of the training
    """

    history_dict = history.history

    # Get the accuracies
    acc = history_dict['sparse_categorical_accuracy']
    val_acc = history_dict['val_sparse_categorical_accuracy']

    # Get the losses
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    # Get the number of epochs
    epochs = range(1, len(acc) + 1)

    # Create a figure
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label="Training Accuracy")
    plt.plot(epochs, val_acc, 'ro-', label="Validation Accuracy")
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label="Training loss")
    plt.plot(epochs, val_loss, 'ro-', label="Validation loss")
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.ylabel('loss')
    plt.legend()

    # Tweak the plot layout
    plt.tight_layout()

    # Construct the filename
    final_val_acc = val_acc[-1]
    final_val_loss = val_loss[-1]
    filename = f"training_history_acc-{final_val_acc:.2f}_loss-{final_val_loss:.2f}.png"

    # Save the plot
    try:
        plt.savefig(filename)
        print(f"Plot successfully saved to {filename}.\n")
    except Exception as e:
        print(f"Error saving plot: {e}.\n")
