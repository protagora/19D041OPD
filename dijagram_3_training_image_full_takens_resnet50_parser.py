import matplotlib.pyplot as plt
import re

output = """
Epoch 1/100
34/34 [==============================] - ETA: 0s - loss: 6.4394 - accuracy: 0.04482023-07-16 08:16:56.162419: I tensorflow/core/common_runtime/executor.cc:1210] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype int32
	 [[{{node Placeholder/_0}}]]
34/34 [==============================] - 26s 735ms/step - loss: 6.4394 - accuracy: 0.0448 - val_loss: 4.1690 - val_accuracy: 0.1007
Epoch 2/100
34/34 [==============================] - 25s 728ms/step - loss: 3.7132 - accuracy: 0.1401 - val_loss: 3.2167 - val_accuracy: 0.2164
Epoch 3/100
34/34 [==============================] - 25s 727ms/step - loss: 3.0418 - accuracy: 0.2157 - val_loss: 2.5740 - val_accuracy: 0.3246
Epoch 4/100
34/34 [==============================] - 25s 729ms/step - loss: 2.5689 - accuracy: 0.2885 - val_loss: 2.3860 - val_accuracy: 0.3284
Epoch 5/100
34/34 [==============================] - 25s 745ms/step - loss: 2.3358 - accuracy: 0.3119 - val_loss: 2.2932 - val_accuracy: 0.3134
Epoch 6/100
34/34 [==============================] - 25s 742ms/step - loss: 2.1517 - accuracy: 0.3800 - val_loss: 1.9908 - val_accuracy: 0.4216
Epoch 7/100
34/34 [==============================] - 26s 756ms/step - loss: 2.0184 - accuracy: 0.3819 - val_loss: 1.9395 - val_accuracy: 0.4478
Epoch 8/100
34/34 [==============================] - 26s 754ms/step - loss: 1.9378 - accuracy: 0.4034 - val_loss: 1.7651 - val_accuracy: 0.4776
Epoch 9/100
34/34 [==============================] - 25s 744ms/step - loss: 1.7688 - accuracy: 0.4472 - val_loss: 1.8366 - val_accuracy: 0.4366
Epoch 10/100
34/34 [==============================] - 25s 752ms/step - loss: 1.7092 - accuracy: 0.4650 - val_loss: 1.6851 - val_accuracy: 0.4590
Epoch 11/100
34/34 [==============================] - 25s 741ms/step - loss: 1.6361 - accuracy: 0.4799 - val_loss: 1.6547 - val_accuracy: 0.4627
Epoch 12/100
34/34 [==============================] - 25s 746ms/step - loss: 1.5789 - accuracy: 0.5023 - val_loss: 1.6645 - val_accuracy: 0.5149
Epoch 13/100
34/34 [==============================] - 26s 754ms/step - loss: 1.5165 - accuracy: 0.5219 - val_loss: 1.7894 - val_accuracy: 0.4366
Epoch 14/100
34/34 [==============================] - 26s 762ms/step - loss: 1.5067 - accuracy: 0.5089 - val_loss: 1.4504 - val_accuracy: 0.5448
Epoch 15/100
34/34 [==============================] - 27s 793ms/step - loss: 1.4494 - accuracy: 0.5210 - val_loss: 1.5242 - val_accuracy: 0.5149
Epoch 16/100
34/34 [==============================] - 29s 849ms/step - loss: 1.3911 - accuracy: 0.5331 - val_loss: 1.6092 - val_accuracy: 0.4963
Epoch 17/100
34/34 [==============================] - 29s 853ms/step - loss: 1.3443 - accuracy: 0.5584 - val_loss: 1.5104 - val_accuracy: 0.5261
Epoch 18/100
34/34 [==============================] - 29s 854ms/step - loss: 1.3334 - accuracy: 0.5789 - val_loss: 1.4539 - val_accuracy: 0.5187
Epoch 19/100
34/34 [==============================] - 29s 862ms/step - loss: 1.3107 - accuracy: 0.5602 - val_loss: 1.5981 - val_accuracy: 0.5112
Epoch 20/100
34/34 [==============================] - 28s 831ms/step - loss: 1.2398 - accuracy: 0.5780 - val_loss: 1.6657 - val_accuracy: 0.4701
Epoch 21/100
34/34 [==============================] - 28s 822ms/step - loss: 1.2575 - accuracy: 0.5873 - val_loss: 1.4264 - val_accuracy: 0.5448
Epoch 22/100
34/34 [==============================] - 28s 829ms/step - loss: 1.2024 - accuracy: 0.5761 - val_loss: 1.4094 - val_accuracy: 0.5336
Epoch 23/100
34/34 [==============================] - 27s 800ms/step - loss: 1.1980 - accuracy: 0.6013 - val_loss: 1.4511 - val_accuracy: 0.5522
Epoch 24/100
34/34 [==============================] - 27s 808ms/step - loss: 1.1597 - accuracy: 0.6088 - val_loss: 1.7308 - val_accuracy: 0.4813
Epoch 25/100
34/34 [==============================] - 28s 845ms/step - loss: 1.1444 - accuracy: 0.5976 - val_loss: 1.3900 - val_accuracy: 0.5522
Epoch 26/100
34/34 [==============================] - 27s 809ms/step - loss: 1.1340 - accuracy: 0.6190 - val_loss: 1.2815 - val_accuracy: 0.5933
Epoch 27/100
34/34 [==============================] - 29s 866ms/step - loss: 1.0780 - accuracy: 0.6265 - val_loss: 1.4240 - val_accuracy: 0.5149
Epoch 28/100
34/34 [==============================] - 28s 822ms/step - loss: 1.0846 - accuracy: 0.6246 - val_loss: 1.4074 - val_accuracy: 0.5560
Epoch 29/100
34/34 [==============================] - 39s 1s/step - loss: 1.0836 - accuracy: 0.6293 - val_loss: 1.3481 - val_accuracy: 0.5821
Epoch 30/100
34/34 [==============================] - 67s 2s/step - loss: 1.0495 - accuracy: 0.6405 - val_loss: 1.3775 - val_accuracy: 0.5634
Epoch 31/100
34/34 [==============================] - 63s 2s/step - loss: 1.0436 - accuracy: 0.6359 - val_loss: 1.3191 - val_accuracy: 0.5970
Epoch 32/100
34/34 [==============================] - 67s 2s/step - loss: 1.0049 - accuracy: 0.6489 - val_loss: 1.4786 - val_accuracy: 0.5784
Epoch 33/100
34/34 [==============================] - 60s 2s/step - loss: 1.0326 - accuracy: 0.6489 - val_loss: 1.2908 - val_accuracy: 0.6007
Epoch 34/100
34/34 [==============================] - 60s 2s/step - loss: 0.9901 - accuracy: 0.6601 - val_loss: 1.2302 - val_accuracy: 0.6007
Epoch 35/100
34/34 [==============================] - 69s 2s/step - loss: 0.9661 - accuracy: 0.6788 - val_loss: 1.3202 - val_accuracy: 0.6045
Epoch 36/100
34/34 [==============================] - 68s 2s/step - loss: 0.9758 - accuracy: 0.6639 - val_loss: 1.1899 - val_accuracy: 0.6119
Epoch 37/100
34/34 [==============================] - 68s 2s/step - loss: 0.9413 - accuracy: 0.6592 - val_loss: 1.6109 - val_accuracy: 0.5485
Epoch 38/100
34/34 [==============================] - 305s 9s/step - loss: 0.9435 - accuracy: 0.6816 - val_loss: 1.3402 - val_accuracy: 0.5970
Epoch 39/100
34/34 [==============================] - 38s 1s/step - loss: 0.9479 - accuracy: 0.6741 - val_loss: 1.4031 - val_accuracy: 0.5746
Epoch 40/100
34/34 [==============================] - 24s 722ms/step - loss: 0.9427 - accuracy: 0.6685 - val_loss: 1.2218 - val_accuracy: 0.6194
Epoch 41/100
34/34 [==============================] - 24s 713ms/step - loss: 0.8995 - accuracy: 0.6872 - val_loss: 1.2561 - val_accuracy: 0.5933
Epoch 42/100
34/34 [==============================] - 25s 746ms/step - loss: 0.8890 - accuracy: 0.6900 - val_loss: 1.1586 - val_accuracy: 0.6269
Epoch 43/100
34/34 [==============================] - 25s 733ms/step - loss: 0.8736 - accuracy: 0.7003 - val_loss: 1.3652 - val_accuracy: 0.6045
Epoch 44/100
34/34 [==============================] - 25s 737ms/step - loss: 0.8448 - accuracy: 0.7087 - val_loss: 1.1507 - val_accuracy: 0.6604
Epoch 45/100
34/34 [==============================] - 26s 758ms/step - loss: 0.8320 - accuracy: 0.7190 - val_loss: 1.4339 - val_accuracy: 0.5634
Epoch 46/100
34/34 [==============================] - 26s 779ms/step - loss: 0.8767 - accuracy: 0.6937 - val_loss: 1.3024 - val_accuracy: 0.6269
Epoch 47/100
34/34 [==============================] - 27s 787ms/step - loss: 0.8461 - accuracy: 0.7106 - val_loss: 1.3707 - val_accuracy: 0.6082
Epoch 48/100
34/34 [==============================] - 27s 784ms/step - loss: 0.8043 - accuracy: 0.7227 - val_loss: 1.3647 - val_accuracy: 0.6007
Epoch 49/100
34/34 [==============================] - 26s 778ms/step - loss: 0.8460 - accuracy: 0.7059 - val_loss: 1.2878 - val_accuracy: 0.5970
Epoch 50/100
34/34 [==============================] - 26s 761ms/step - loss: 0.8127 - accuracy: 0.7255 - val_loss: 1.2090 - val_accuracy: 0.6343
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

    plt.savefig('dijagram_3_takens_plot_loss_accuracy_resnet50.png')
    plt.show()


accuracy, loss, val_accuracy, val_loss = extract_metrics(output)

# print("Accuracy:", accuracy)
# print("Loss:", loss)
# print("Validation Accuracy:", val_accuracy)
# print("Validation Loss:", val_loss)

# Example usage
plot_metrics(accuracy, loss, val_accuracy, val_loss)
