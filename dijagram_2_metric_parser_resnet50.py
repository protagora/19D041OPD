import matplotlib.pyplot as plt
import re

output = """
Epoch 1/100
35/35 [==============================] - ETA: 0s - loss: 5.2215 - accuracy: 0.05892023-06-26 09:25:05.127032: I tensorflow/core/common_runtime/executor.cc:1210] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype int32
	 [[{{node Placeholder/_0}}]]
35/35 [==============================] - 30s 835ms/step - loss: 5.2215 - accuracy: 0.0589 - val_loss: 3.8109 - val_accuracy: 0.1356
Epoch 2/100
35/35 [==============================] - 29s 824ms/step - loss: 3.5170 - accuracy: 0.1549 - val_loss: 2.9522 - val_accuracy: 0.2681
Epoch 3/100
35/35 [==============================] - 29s 826ms/step - loss: 2.7852 - accuracy: 0.2899 - val_loss: 2.5424 - val_accuracy: 0.3060
Epoch 4/100
35/35 [==============================] - 29s 831ms/step - loss: 2.3318 - accuracy: 0.3877 - val_loss: 2.2045 - val_accuracy: 0.4164
Epoch 5/100
35/35 [==============================] - 29s 831ms/step - loss: 2.0763 - accuracy: 0.4212 - val_loss: 2.1796 - val_accuracy: 0.4069
Epoch 6/100
35/35 [==============================] - 29s 832ms/step - loss: 1.8832 - accuracy: 0.4583 - val_loss: 1.9821 - val_accuracy: 0.4511
Epoch 7/100
35/35 [==============================] - 29s 834ms/step - loss: 1.7592 - accuracy: 0.4837 - val_loss: 1.8514 - val_accuracy: 0.4953
Epoch 8/100
35/35 [==============================] - 29s 838ms/step - loss: 1.6314 - accuracy: 0.5109 - val_loss: 1.7510 - val_accuracy: 0.5047
Epoch 9/100
35/35 [==============================] - 29s 842ms/step - loss: 1.5256 - accuracy: 0.5399 - val_loss: 1.5205 - val_accuracy: 0.5521
Epoch 10/100
35/35 [==============================] - 29s 843ms/step - loss: 1.4017 - accuracy: 0.5779 - val_loss: 1.6359 - val_accuracy: 0.4795
Epoch 11/100
35/35 [==============================] - 30s 854ms/step - loss: 1.3611 - accuracy: 0.5933 - val_loss: 1.5249 - val_accuracy: 0.5552
Epoch 12/100
35/35 [==============================] - 30s 849ms/step - loss: 1.2910 - accuracy: 0.6060 - val_loss: 1.5596 - val_accuracy: 0.5300
Epoch 13/100
35/35 [==============================] - 30s 855ms/step - loss: 1.2167 - accuracy: 0.6196 - val_loss: 1.7196 - val_accuracy: 0.5142
Epoch 14/100
35/35 [==============================] - 30s 848ms/step - loss: 1.1766 - accuracy: 0.6322 - val_loss: 1.4451 - val_accuracy: 0.5773
Epoch 15/100
35/35 [==============================] - 30s 853ms/step - loss: 1.1442 - accuracy: 0.6576 - val_loss: 1.4784 - val_accuracy: 0.5363
Epoch 16/100
35/35 [==============================] - 30s 854ms/step - loss: 1.1081 - accuracy: 0.6486 - val_loss: 1.4037 - val_accuracy: 0.5804
Epoch 17/100
35/35 [==============================] - 30s 853ms/step - loss: 1.0749 - accuracy: 0.6558 - val_loss: 1.3473 - val_accuracy: 0.6057
Epoch 18/100
35/35 [==============================] - 30s 852ms/step - loss: 1.0312 - accuracy: 0.6766 - val_loss: 1.4513 - val_accuracy: 0.5741
Epoch 19/100
35/35 [==============================] - 30s 859ms/step - loss: 0.9954 - accuracy: 0.6920 - val_loss: 1.3785 - val_accuracy: 0.5931
Epoch 20/100
35/35 [==============================] - 30s 860ms/step - loss: 0.9949 - accuracy: 0.6775 - val_loss: 1.3023 - val_accuracy: 0.5994
Epoch 21/100
35/35 [==============================] - 905s 27s/step - loss: 0.9238 - accuracy: 0.7111 - val_loss: 1.3638 - val_accuracy: 0.6215
Epoch 22/100
35/35 [==============================] - 30s 867ms/step - loss: 0.9426 - accuracy: 0.7011 - val_loss: 1.3788 - val_accuracy: 0.6057
Epoch 23/100
35/35 [==============================] - 30s 866ms/step - loss: 0.9088 - accuracy: 0.7138 - val_loss: 1.2278 - val_accuracy: 0.6215
Epoch 24/100
35/35 [==============================] - 993s 29s/step - loss: 0.8432 - accuracy: 0.7328 - val_loss: 1.5643 - val_accuracy: 0.5552
Epoch 25/100
35/35 [==============================] - 31s 880ms/step - loss: 0.8657 - accuracy: 0.7337 - val_loss: 1.2775 - val_accuracy: 0.6341
Epoch 26/100
35/35 [==============================] - 957s 28s/step - loss: 0.8220 - accuracy: 0.7418 - val_loss: 1.3628 - val_accuracy: 0.6151
Epoch 27/100
35/35 [==============================] - 1001s 29s/step - loss: 0.8297 - accuracy: 0.7255 - val_loss: 1.2181 - val_accuracy: 0.6814
Epoch 28/100
35/35 [==============================] - 30s 885ms/step - loss: 0.8126 - accuracy: 0.7337 - val_loss: 1.3068 - val_accuracy: 0.6372
Epoch 29/100
35/35 [==============================] - 990s 29s/step - loss: 0.7848 - accuracy: 0.7563 - val_loss: 1.2771 - val_accuracy: 0.6530
Epoch 30/100
35/35 [==============================] - 31s 879ms/step - loss: 0.7624 - accuracy: 0.7618 - val_loss: 1.4845 - val_accuracy: 0.5931
Epoch 31/100
35/35 [==============================] - 1017s 30s/step - loss: 0.7605 - accuracy: 0.7554 - val_loss: 1.2473 - val_accuracy: 0.6656
Epoch 32/100
35/35 [==============================] - 981s 29s/step - loss: 0.7536 - accuracy: 0.7554 - val_loss: 1.1839 - val_accuracy: 0.6845
Epoch 33/100
35/35 [==============================] - 31s 887ms/step - loss: 0.7594 - accuracy: 0.7509 - val_loss: 1.2222 - val_accuracy: 0.6530
Epoch 34/100
35/35 [==============================] - 971s 29s/step - loss: 0.7194 - accuracy: 0.7600 - val_loss: 1.2205 - val_accuracy: 0.6562
Epoch 35/100
35/35 [==============================] - 31s 895ms/step - loss: 0.7211 - accuracy: 0.7663 - val_loss: 1.2351 - val_accuracy: 0.6688
Epoch 36/100
35/35 [==============================] - 939s 28s/step - loss: 0.7026 - accuracy: 0.7690 - val_loss: 1.2185 - val_accuracy: 0.6562
Epoch 37/100
35/35 [==============================] - 1040s 31s/step - loss: 0.6969 - accuracy: 0.7754 - val_loss: 1.2974 - val_accuracy: 0.6751
Epoch 38/100
35/35 [==============================] - 31s 898ms/step - loss: 0.7000 - accuracy: 0.7672 - val_loss: 1.2933 - val_accuracy: 0.6562
Epoch 39/100
35/35 [==============================] - 277s 8s/step - loss: 0.6833 - accuracy: 0.7790 - val_loss: 1.2188 - val_accuracy: 0.6845
Epoch 40/100
35/35 [==============================] - 211s 6s/step - loss: 0.6779 - accuracy: 0.7853 - val_loss: 1.3121 - val_accuracy: 0.6593
Epoch 41/100
35/35 [==============================] - 31s 888ms/step - loss: 0.6516 - accuracy: 0.7763 - val_loss: 1.2312 - val_accuracy: 0.6467
Epoch 42/100
35/35 [==============================] - 31s 898ms/step - loss: 0.6417 - accuracy: 0.7862 - val_loss: 1.1896 - val_accuracy: 0.6719
Epoch 43/100
35/35 [==============================] - 997s 29s/step - loss: 0.6584 - accuracy: 0.7826 - val_loss: 1.1749 - val_accuracy: 0.6688
Epoch 44/100
35/35 [==============================] - 6234s 183s/step - loss: 0.6311 - accuracy: 0.7899 - val_loss: 1.2649 - val_accuracy: 0.6751
Epoch 45/100
35/35 [==============================] - 5196s 153s/step - loss: 0.6229 - accuracy: 0.7862 - val_loss: 1.2387 - val_accuracy: 0.6562
Epoch 46/100
35/35 [==============================] - 5965s 175s/step - loss: 0.5963 - accuracy: 0.7962 - val_loss: 1.3090 - val_accuracy: 0.6814
Epoch 47/100
35/35 [==============================] - 1445s 42s/step - loss: 0.6024 - accuracy: 0.8034 - val_loss: 1.2556 - val_accuracy: 0.6909
Epoch 48/100
35/35 [==============================] - 30s 854ms/step - loss: 0.6064 - accuracy: 0.8034 - val_loss: 1.3094 - val_accuracy: 0.6656
Epoch 49/100
35/35 [==============================] - 30s 849ms/step - loss: 0.5991 - accuracy: 0.7944 - val_loss: 1.2529 - val_accuracy: 0.6782
Epoch 50/100
35/35 [==============================] - 31s 890ms/step - loss: 0.5905 - accuracy: 0.8062 - val_loss: 1.2881 - val_accuracy: 0.6940
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
