import matplotlib.pyplot as plt
import re

output = """
Epoch 1/100
34/34 [==============================] - ETA: 0s - loss: 5.9584 - accuracy: 0.00372023-07-16 22:48:24.837628: I tensorflow/core/common_runtime/executor.cc:1210] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype int32
	 [[{{node Placeholder/_0}}]]
34/34 [==============================] - 17s 485ms/step - loss: 5.9584 - accuracy: 0.0037 - val_loss: 4.8977 - val_accuracy: 0.0075
Epoch 2/100
34/34 [==============================] - 16s 477ms/step - loss: 4.9073 - accuracy: 0.0075 - val_loss: 4.9209 - val_accuracy: 0.0075
Epoch 3/100
34/34 [==============================] - 16s 473ms/step - loss: 4.9260 - accuracy: 0.0084 - val_loss: 4.8557 - val_accuracy: 0.0075
Epoch 4/100
34/34 [==============================] - 16s 473ms/step - loss: 4.7148 - accuracy: 0.0252 - val_loss: 4.6201 - val_accuracy: 0.0746
Epoch 5/100
34/34 [==============================] - 16s 471ms/step - loss: 4.2814 - accuracy: 0.0514 - val_loss: 4.0511 - val_accuracy: 0.0821
Epoch 6/100
34/34 [==============================] - 16s 468ms/step - loss: 3.8870 - accuracy: 0.1120 - val_loss: 3.9771 - val_accuracy: 0.0821
Epoch 7/100
34/34 [==============================] - 16s 476ms/step - loss: 3.6214 - accuracy: 0.1232 - val_loss: 3.2853 - val_accuracy: 0.1754
Epoch 8/100
34/34 [==============================] - 17s 490ms/step - loss: 3.4210 - accuracy: 0.1569 - val_loss: 3.2583 - val_accuracy: 0.2090
Epoch 9/100
34/34 [==============================] - 17s 508ms/step - loss: 3.1260 - accuracy: 0.2035 - val_loss: 2.6613 - val_accuracy: 0.2948
Epoch 10/100
34/34 [==============================] - 18s 515ms/step - loss: 2.9496 - accuracy: 0.2325 - val_loss: 3.0310 - val_accuracy: 0.2575
Epoch 11/100
34/34 [==============================] - 17s 492ms/step - loss: 2.7385 - accuracy: 0.2782 - val_loss: 2.2792 - val_accuracy: 0.3769
Epoch 12/100
34/34 [==============================] - 17s 489ms/step - loss: 2.5155 - accuracy: 0.3091 - val_loss: 2.1915 - val_accuracy: 0.4067
Epoch 13/100
34/34 [==============================] - 17s 486ms/step - loss: 2.2779 - accuracy: 0.3688 - val_loss: 2.4650 - val_accuracy: 0.3246
Epoch 14/100
34/34 [==============================] - 16s 485ms/step - loss: 2.1453 - accuracy: 0.3968 - val_loss: 2.1091 - val_accuracy: 0.4254
Epoch 15/100
34/34 [==============================] - 17s 485ms/step - loss: 2.0098 - accuracy: 0.4155 - val_loss: 2.0408 - val_accuracy: 0.4254
Epoch 16/100
34/34 [==============================] - 16s 484ms/step - loss: 1.9184 - accuracy: 0.4295 - val_loss: 2.0836 - val_accuracy: 0.4515
Epoch 17/100
34/34 [==============================] - 17s 490ms/step - loss: 1.8023 - accuracy: 0.4641 - val_loss: 1.7397 - val_accuracy: 0.4851
Epoch 18/100
34/34 [==============================] - 17s 488ms/step - loss: 1.7188 - accuracy: 0.4799 - val_loss: 1.7400 - val_accuracy: 0.5075
Epoch 19/100
34/34 [==============================] - 17s 491ms/step - loss: 1.6482 - accuracy: 0.5005 - val_loss: 1.6452 - val_accuracy: 0.4925
Epoch 20/100
34/34 [==============================] - 16s 482ms/step - loss: 1.4951 - accuracy: 0.5303 - val_loss: 1.5781 - val_accuracy: 0.5224
Epoch 21/100
34/34 [==============================] - 16s 482ms/step - loss: 1.4598 - accuracy: 0.5472 - val_loss: 1.5291 - val_accuracy: 0.5746
Epoch 22/100
34/34 [==============================] - 16s 481ms/step - loss: 1.4118 - accuracy: 0.5612 - val_loss: 1.5943 - val_accuracy: 0.5261
Epoch 23/100
34/34 [==============================] - 16s 481ms/step - loss: 1.3260 - accuracy: 0.5798 - val_loss: 1.6165 - val_accuracy: 0.5373
Epoch 24/100
34/34 [==============================] - 17s 486ms/step - loss: 1.2907 - accuracy: 0.5929 - val_loss: 1.6564 - val_accuracy: 0.5448
Epoch 25/100
34/34 [==============================] - 17s 486ms/step - loss: 1.2510 - accuracy: 0.6013 - val_loss: 1.5584 - val_accuracy: 0.5261
Epoch 26/100
34/34 [==============================] - 16s 482ms/step - loss: 1.1792 - accuracy: 0.6181 - val_loss: 1.4626 - val_accuracy: 0.5933
Epoch 27/100
34/34 [==============================] - 16s 485ms/step - loss: 1.1593 - accuracy: 0.6321 - val_loss: 1.5857 - val_accuracy: 0.5746
Epoch 28/100
34/34 [==============================] - 16s 482ms/step - loss: 1.1020 - accuracy: 0.6489 - val_loss: 1.6947 - val_accuracy: 0.5410
Epoch 29/100
34/34 [==============================] - 16s 484ms/step - loss: 1.0414 - accuracy: 0.6564 - val_loss: 1.5757 - val_accuracy: 0.5858
Epoch 30/100
34/34 [==============================] - 16s 483ms/step - loss: 1.0286 - accuracy: 0.6667 - val_loss: 1.6007 - val_accuracy: 0.5933
Epoch 31/100
34/34 [==============================] - 16s 481ms/step - loss: 1.0031 - accuracy: 0.6723 - val_loss: 1.6065 - val_accuracy: 0.6194
Epoch 32/100
34/34 [==============================] - 16s 481ms/step - loss: 0.9125 - accuracy: 0.7134 - val_loss: 1.6223 - val_accuracy: 0.6045
Epoch 33/100
34/34 [==============================] - 16s 484ms/step - loss: 0.8922 - accuracy: 0.7171 - val_loss: 1.7797 - val_accuracy: 0.5933
Epoch 34/100
34/34 [==============================] - 16s 483ms/step - loss: 0.8630 - accuracy: 0.7087 - val_loss: 1.5886 - val_accuracy: 0.5858
Epoch 35/100
34/34 [==============================] - 16s 484ms/step - loss: 0.8310 - accuracy: 0.7171 - val_loss: 1.4883 - val_accuracy: 0.6045
Epoch 36/100
34/34 [==============================] - 16s 485ms/step - loss: 0.7998 - accuracy: 0.7255 - val_loss: 1.6075 - val_accuracy: 0.5970
Epoch 37/100
34/34 [==============================] - 16s 483ms/step - loss: 0.7968 - accuracy: 0.7479 - val_loss: 1.4770 - val_accuracy: 0.6157
Epoch 38/100
34/34 [==============================] - 16s 483ms/step - loss: 0.7621 - accuracy: 0.7535 - val_loss: 1.6838 - val_accuracy: 0.6231
Epoch 39/100
34/34 [==============================] - 16s 483ms/step - loss: 0.7244 - accuracy: 0.7488 - val_loss: 1.6293 - val_accuracy: 0.6269
Epoch 40/100
34/34 [==============================] - 16s 484ms/step - loss: 0.7215 - accuracy: 0.7442 - val_loss: 1.7897 - val_accuracy: 0.6194
Epoch 41/100
34/34 [==============================] - 16s 484ms/step - loss: 0.6891 - accuracy: 0.7582 - val_loss: 1.7334 - val_accuracy: 0.6455
Epoch 42/100
34/34 [==============================] - 16s 484ms/step - loss: 0.6651 - accuracy: 0.7778 - val_loss: 1.6968 - val_accuracy: 0.6119
Epoch 43/100
34/34 [==============================] - 16s 483ms/step - loss: 0.6695 - accuracy: 0.7778 - val_loss: 1.9166 - val_accuracy: 0.6082
Epoch 44/100
34/34 [==============================] - 16s 480ms/step - loss: 0.6194 - accuracy: 0.7908 - val_loss: 1.8753 - val_accuracy: 0.5933
Epoch 45/100
34/34 [==============================] - 16s 482ms/step - loss: 0.6091 - accuracy: 0.7899 - val_loss: 1.8078 - val_accuracy: 0.6604
Epoch 46/100
34/34 [==============================] - 16s 492ms/step - loss: 0.6165 - accuracy: 0.7880 - val_loss: 1.9507 - val_accuracy: 0.6306
Epoch 47/100
34/34 [==============================] - 257s 8s/step - loss: 0.6023 - accuracy: 0.7993 - val_loss: 2.4183 - val_accuracy: 0.6082
Epoch 48/100
34/34 [==============================] - 16s 482ms/step - loss: 0.5987 - accuracy: 0.8133 - val_loss: 1.8353 - val_accuracy: 0.6455
Epoch 49/100
34/34 [==============================] - 16s 482ms/step - loss: 0.5485 - accuracy: 0.8067 - val_loss: 1.7566 - val_accuracy: 0.6418
Epoch 50/100
34/34 [==============================] - 17s 487ms/step - loss: 0.5451 - accuracy: 0.8058 - val_loss: 1.7411 - val_accuracy: 0.6530
"""

def extract_metrics(output):
    accuracy = []
    loss = []
    val_accuracy = []
    val_loss = []

    epoch_pattern = r"Epoch \d+/\d+"
    accuracy_pattern = r"accuracy: (\d+\.\d+)"
    loss_pattern = r"loss: (\d+\.\d+)"
    val_accuracy_pattern = r"val_accuracy: (\d+\.\d+)"
    val_loss_pattern = r"val_loss: (\d+\.\d+)"

    epochs = re.findall(epoch_pattern, output)
    accuracies = re.findall(accuracy_pattern, output)
    losses = re.findall(loss_pattern, output)
    val_accuracies = re.findall(val_accuracy_pattern, output)
    val_losses = re.findall(val_loss_pattern, output)

    for i in range(len(epochs)):
        accuracy.append(float(accuracies[i]))
        loss.append(float(losses[i]))
        val_accuracy.append(float(val_accuracies[i]))
        val_loss.append(float(val_losses[i]))

    return accuracy, loss, val_accuracy, val_loss

def plot_metrics(accuracy, loss, val_accuracy, val_loss):
    epochs = range(1, len(accuracy) + 1)

    plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.grid('on')
    plt.tight_layout()

    plt.savefig('wavelet_plot_loss_accuracy.png')
    plt.show()


accuracy, loss, val_accuracy, val_loss = extract_metrics(output)

# print("Accuracy:", accuracy)
# print("Loss:", loss)
# print("Validation Accuracy:", val_accuracy)
# print("Validation Loss:", val_loss)

# Example usage
plot_metrics(accuracy, loss, val_accuracy, val_loss)
