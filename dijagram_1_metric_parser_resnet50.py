import matplotlib.pyplot as plt
import re

output = """
Epoch 1/50
41/41 [==============================] - ETA: 0s - loss: 6.0624 - accuracy: 0.01782023-06-26 01:11:49.736239: I tensorflow/core/common_runtime/executor.cc:1210] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype int32
	 [[{{node Placeholder/_0}}]]
41/41 [==============================] - 38s 911ms/step - loss: 6.0624 - accuracy: 0.0178 - val_loss: 4.6460 - val_accuracy: 0.0592
Epoch 2/50
41/41 [==============================] - 36s 891ms/step - loss: 4.5929 - accuracy: 0.0626 - val_loss: 4.3360 - val_accuracy: 0.1000
Epoch 3/50
41/41 [==============================] - 37s 911ms/step - loss: 4.1766 - accuracy: 0.0990 - val_loss: 3.8257 - val_accuracy: 0.1449
Epoch 4/50
41/41 [==============================] - 37s 905ms/step - loss: 3.8206 - accuracy: 0.1307 - val_loss: 3.5500 - val_accuracy: 0.1776
Epoch 5/50
41/41 [==============================] - 37s 917ms/step - loss: 3.5526 - accuracy: 0.1539 - val_loss: 3.3913 - val_accuracy: 0.1735
Epoch 6/50
41/41 [==============================] - 37s 900ms/step - loss: 3.3287 - accuracy: 0.1872 - val_loss: 3.0804 - val_accuracy: 0.2245
Epoch 7/50
41/41 [==============================] - 37s 904ms/step - loss: 3.1622 - accuracy: 0.2096 - val_loss: 3.0141 - val_accuracy: 0.2388
Epoch 8/50
41/41 [==============================] - 37s 900ms/step - loss: 3.0367 - accuracy: 0.2196 - val_loss: 2.8707 - val_accuracy: 0.2755
Epoch 9/50
41/41 [==============================] - 37s 906ms/step - loss: 2.9100 - accuracy: 0.2475 - val_loss: 2.8402 - val_accuracy: 0.2551
Epoch 10/50
41/41 [==============================] - 37s 911ms/step - loss: 2.8058 - accuracy: 0.2707 - val_loss: 2.6683 - val_accuracy: 0.2939
Epoch 11/50
41/41 [==============================] - 38s 926ms/step - loss: 2.7457 - accuracy: 0.2753 - val_loss: 2.6119 - val_accuracy: 0.2918
Epoch 12/50
41/41 [==============================] - 37s 915ms/step - loss: 2.6605 - accuracy: 0.2769 - val_loss: 2.5277 - val_accuracy: 0.3163
Epoch 13/50
41/41 [==============================] - 37s 919ms/step - loss: 2.6019 - accuracy: 0.3086 - val_loss: 2.4708 - val_accuracy: 0.3367
Epoch 14/50
41/41 [==============================] - 37s 919ms/step - loss: 2.4923 - accuracy: 0.3233 - val_loss: 2.3504 - val_accuracy: 0.3735
Epoch 15/50
41/41 [==============================] - 39s 947ms/step - loss: 2.4636 - accuracy: 0.3155 - val_loss: 2.3300 - val_accuracy: 0.3694
Epoch 16/50
41/41 [==============================] - 38s 934ms/step - loss: 2.3565 - accuracy: 0.3581 - val_loss: 2.2483 - val_accuracy: 0.3816
Epoch 17/50
41/41 [==============================] - 38s 929ms/step - loss: 2.3576 - accuracy: 0.3349 - val_loss: 2.1807 - val_accuracy: 0.3857
Epoch 18/50
41/41 [==============================] - 38s 943ms/step - loss: 2.3044 - accuracy: 0.3612 - val_loss: 2.1373 - val_accuracy: 0.4041
Epoch 19/50
41/41 [==============================] - 38s 940ms/step - loss: 2.2583 - accuracy: 0.3658 - val_loss: 2.0396 - val_accuracy: 0.4449
Epoch 20/50
41/41 [==============================] - 39s 965ms/step - loss: 2.2102 - accuracy: 0.3805 - val_loss: 2.0389 - val_accuracy: 0.4306
Epoch 21/50
41/41 [==============================] - 40s 986ms/step - loss: 2.1548 - accuracy: 0.3991 - val_loss: 2.0957 - val_accuracy: 0.3857
Epoch 22/50
41/41 [==============================] - 39s 961ms/step - loss: 2.0951 - accuracy: 0.4006 - val_loss: 1.9981 - val_accuracy: 0.4388
Epoch 23/50
41/41 [==============================] - 39s 966ms/step - loss: 2.0415 - accuracy: 0.4246 - val_loss: 1.9568 - val_accuracy: 0.4571
Epoch 24/50
41/41 [==============================] - 40s 986ms/step - loss: 2.0334 - accuracy: 0.4076 - val_loss: 1.7843 - val_accuracy: 0.4816
Epoch 25/50
41/41 [==============================] - 40s 993ms/step - loss: 1.9769 - accuracy: 0.4176 - val_loss: 1.8029 - val_accuracy: 0.5020
Epoch 26/50
41/41 [==============================] - 39s 961ms/step - loss: 1.9447 - accuracy: 0.4323 - val_loss: 1.8198 - val_accuracy: 0.4857
Epoch 27/50
41/41 [==============================] - 40s 982ms/step - loss: 1.8689 - accuracy: 0.4617 - val_loss: 1.9516 - val_accuracy: 0.4490
Epoch 28/50
41/41 [==============================] - 40s 975ms/step - loss: 1.8714 - accuracy: 0.4602 - val_loss: 1.9895 - val_accuracy: 0.4245
Epoch 29/50
41/41 [==============================] - 39s 968ms/step - loss: 1.8242 - accuracy: 0.4764 - val_loss: 1.8019 - val_accuracy: 0.4837
Epoch 30/50
41/41 [==============================] - 39s 957ms/step - loss: 1.7925 - accuracy: 0.4741 - val_loss: 1.8948 - val_accuracy: 0.4633
Epoch 31/50
41/41 [==============================] - 39s 966ms/step - loss: 1.7714 - accuracy: 0.4787 - val_loss: 1.7107 - val_accuracy: 0.5184
Epoch 32/50
41/41 [==============================] - 37s 914ms/step - loss: 1.7215 - accuracy: 0.4919 - val_loss: 1.6627 - val_accuracy: 0.5429
Epoch 33/50
41/41 [==============================] - 38s 922ms/step - loss: 1.7250 - accuracy: 0.4896 - val_loss: 1.6743 - val_accuracy: 0.4857
Epoch 34/50
41/41 [==============================] - 38s 924ms/step - loss: 1.6602 - accuracy: 0.5159 - val_loss: 1.6654 - val_accuracy: 0.5204
Epoch 35/50
41/41 [==============================] - 37s 917ms/step - loss: 1.6585 - accuracy: 0.5120 - val_loss: 1.6634 - val_accuracy: 0.5163
Epoch 36/50
41/41 [==============================] - 38s 921ms/step - loss: 1.6272 - accuracy: 0.5135 - val_loss: 1.5424 - val_accuracy: 0.5531
Epoch 37/50
41/41 [==============================] - 38s 934ms/step - loss: 1.6270 - accuracy: 0.5089 - val_loss: 1.4918 - val_accuracy: 0.5673
Epoch 38/50
41/41 [==============================] - 38s 929ms/step - loss: 1.5787 - accuracy: 0.5275 - val_loss: 1.4175 - val_accuracy: 0.5898
Epoch 39/50
41/41 [==============================] - 38s 930ms/step - loss: 1.5393 - accuracy: 0.5538 - val_loss: 1.5213 - val_accuracy: 0.5531
Epoch 40/50
41/41 [==============================] - 38s 931ms/step - loss: 1.5184 - accuracy: 0.5507 - val_loss: 1.5040 - val_accuracy: 0.5673
Epoch 41/50
41/41 [==============================] - 38s 924ms/step - loss: 1.4921 - accuracy: 0.5592 - val_loss: 1.3821 - val_accuracy: 0.5939
Epoch 42/50
41/41 [==============================] - 38s 931ms/step - loss: 1.4753 - accuracy: 0.5468 - val_loss: 1.3686 - val_accuracy: 0.6102
Epoch 43/50
41/41 [==============================] - 38s 930ms/step - loss: 1.4412 - accuracy: 0.5654 - val_loss: 1.4039 - val_accuracy: 0.5837
Epoch 44/50
41/41 [==============================] - 38s 938ms/step - loss: 1.4077 - accuracy: 0.5855 - val_loss: 1.5955 - val_accuracy: 0.5327
Epoch 45/50
41/41 [==============================] - 38s 942ms/step - loss: 1.4235 - accuracy: 0.5770 - val_loss: 1.4738 - val_accuracy: 0.5959
Epoch 46/50
41/41 [==============================] - 38s 934ms/step - loss: 1.3661 - accuracy: 0.5971 - val_loss: 1.4688 - val_accuracy: 0.5755
Epoch 47/50
41/41 [==============================] - 38s 934ms/step - loss: 1.3624 - accuracy: 0.5777 - val_loss: 1.3297 - val_accuracy: 0.6224
Epoch 48/50
41/41 [==============================] - 38s 942ms/step - loss: 1.3486 - accuracy: 0.5924 - val_loss: 1.3020 - val_accuracy: 0.6265
Epoch 49/50
41/41 [==============================] - 38s 947ms/step - loss: 1.3410 - accuracy: 0.5886 - val_loss: 1.3934 - val_accuracy: 0.6041
Epoch 50/50
41/41 [==============================] - 38s 941ms/step - loss: 1.3164 - accuracy: 0.6056 - val_loss: 1.3549 - val_accuracy: 0.6000
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

    plt.savefig('recurrence_plot_loss_accuracy.png')
    plt.show()


accuracy, loss, val_accuracy, val_loss = extract_metrics(output)

# Example usage
plot_metrics(accuracy, loss, val_accuracy, val_loss)
