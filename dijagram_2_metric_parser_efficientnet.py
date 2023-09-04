import matplotlib.pyplot as plt
import re

output = """
Epoch 1/100
41/41 [==============================] - ETA: 0s - loss: 4.3276 - accuracy: 0.08282023-06-27 02:48:55.045910: I tensorflow/core/common_runtime/executor.cc:1210] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype int32
	 [[{{node Placeholder/_0}}]]
41/41 [==============================] - 15s 315ms/step - loss: 4.3276 - accuracy: 0.0828 - val_loss: 3.1662 - val_accuracy: 0.2534
Epoch 2/100
41/41 [==============================] - 13s 318ms/step - loss: 3.0215 - accuracy: 0.2307 - val_loss: 2.4399 - val_accuracy: 0.3512
Epoch 3/100
41/41 [==============================] - 14s 334ms/step - loss: 2.4885 - accuracy: 0.3212 - val_loss: 2.0864 - val_accuracy: 0.4069
Epoch 4/100
41/41 [==============================] - 14s 340ms/step - loss: 2.2361 - accuracy: 0.3638 - val_loss: 1.7842 - val_accuracy: 0.4933
Epoch 5/100
41/41 [==============================] - 14s 331ms/step - loss: 1.9989 - accuracy: 0.4195 - val_loss: 1.7163 - val_accuracy: 0.4894
Epoch 6/100
41/41 [==============================] - 14s 334ms/step - loss: 1.8379 - accuracy: 0.4443 - val_loss: 1.6958 - val_accuracy: 0.4875
Epoch 7/100
41/41 [==============================] - 13s 312ms/step - loss: 1.7634 - accuracy: 0.4799 - val_loss: 1.4771 - val_accuracy: 0.5509
Epoch 8/100
41/41 [==============================] - 13s 312ms/step - loss: 1.6347 - accuracy: 0.5139 - val_loss: 1.4020 - val_accuracy: 0.5739
Epoch 9/100
41/41 [==============================] - 13s 313ms/step - loss: 1.5921 - accuracy: 0.5209 - val_loss: 1.2820 - val_accuracy: 0.5854
Epoch 10/100
41/41 [==============================] - 13s 315ms/step - loss: 1.4746 - accuracy: 0.5519 - val_loss: 1.2281 - val_accuracy: 0.6219
Epoch 11/100
41/41 [==============================] - 13s 311ms/step - loss: 1.4181 - accuracy: 0.5650 - val_loss: 1.1846 - val_accuracy: 0.6430
Epoch 12/100
41/41 [==============================] - 13s 312ms/step - loss: 1.3937 - accuracy: 0.5650 - val_loss: 1.1657 - val_accuracy: 0.6315
Epoch 13/100
41/41 [==============================] - 13s 320ms/step - loss: 1.3273 - accuracy: 0.5743 - val_loss: 1.0806 - val_accuracy: 0.6737
Epoch 14/100
41/41 [==============================] - 13s 315ms/step - loss: 1.2598 - accuracy: 0.6022 - val_loss: 1.0557 - val_accuracy: 0.6871
Epoch 15/100
41/41 [==============================] - 13s 314ms/step - loss: 1.2376 - accuracy: 0.5983 - val_loss: 1.0506 - val_accuracy: 0.6756
Epoch 16/100
41/41 [==============================] - 13s 314ms/step - loss: 1.2036 - accuracy: 0.6169 - val_loss: 0.9871 - val_accuracy: 0.6756
Epoch 17/100
41/41 [==============================] - 13s 314ms/step - loss: 1.1517 - accuracy: 0.6300 - val_loss: 0.9114 - val_accuracy: 0.7140
Epoch 18/100
41/41 [==============================] - 13s 316ms/step - loss: 1.1208 - accuracy: 0.6432 - val_loss: 0.9589 - val_accuracy: 0.6795
Epoch 19/100
41/41 [==============================] - 13s 319ms/step - loss: 1.1030 - accuracy: 0.6463 - val_loss: 0.9039 - val_accuracy: 0.7121
Epoch 20/100
41/41 [==============================] - 13s 317ms/step - loss: 1.0812 - accuracy: 0.6509 - val_loss: 0.8874 - val_accuracy: 0.7217
Epoch 21/100
41/41 [==============================] - 13s 316ms/step - loss: 1.0499 - accuracy: 0.6633 - val_loss: 0.8831 - val_accuracy: 0.7236
Epoch 22/100
41/41 [==============================] - 13s 315ms/step - loss: 1.0600 - accuracy: 0.6556 - val_loss: 0.8632 - val_accuracy: 0.7102
Epoch 23/100
41/41 [==============================] - 13s 317ms/step - loss: 1.0302 - accuracy: 0.6610 - val_loss: 0.8517 - val_accuracy: 0.7006
Epoch 24/100
41/41 [==============================] - 13s 314ms/step - loss: 1.0012 - accuracy: 0.6765 - val_loss: 0.8617 - val_accuracy: 0.7179
Epoch 25/100
41/41 [==============================] - 13s 318ms/step - loss: 0.9755 - accuracy: 0.6850 - val_loss: 0.8434 - val_accuracy: 0.7255
Epoch 26/100
41/41 [==============================] - 13s 318ms/step - loss: 0.9845 - accuracy: 0.6780 - val_loss: 0.8128 - val_accuracy: 0.7236
Epoch 27/100
41/41 [==============================] - 13s 319ms/step - loss: 0.9400 - accuracy: 0.6881 - val_loss: 0.8200 - val_accuracy: 0.7140
Epoch 28/100
41/41 [==============================] - 13s 318ms/step - loss: 0.9370 - accuracy: 0.6889 - val_loss: 0.7890 - val_accuracy: 0.7313
Epoch 29/100
41/41 [==============================] - 13s 326ms/step - loss: 0.9190 - accuracy: 0.6989 - val_loss: 0.7391 - val_accuracy: 0.7505
Epoch 30/100
41/41 [==============================] - 13s 322ms/step - loss: 0.9043 - accuracy: 0.6958 - val_loss: 0.7643 - val_accuracy: 0.7543
Epoch 31/100
41/41 [==============================] - 13s 321ms/step - loss: 0.9063 - accuracy: 0.6935 - val_loss: 0.7562 - val_accuracy: 0.7370
Epoch 32/100
41/41 [==============================] - 13s 321ms/step - loss: 0.8740 - accuracy: 0.7183 - val_loss: 0.8369 - val_accuracy: 0.7313
Epoch 33/100
41/41 [==============================] - 13s 323ms/step - loss: 0.8872 - accuracy: 0.7175 - val_loss: 0.7383 - val_accuracy: 0.7486
Epoch 34/100
41/41 [==============================] - 13s 324ms/step - loss: 0.8620 - accuracy: 0.7214 - val_loss: 0.7168 - val_accuracy: 0.7543
Epoch 35/100
41/41 [==============================] - 13s 321ms/step - loss: 0.8440 - accuracy: 0.7152 - val_loss: 0.7446 - val_accuracy: 0.7562
Epoch 36/100
41/41 [==============================] - 13s 324ms/step - loss: 0.8516 - accuracy: 0.7221 - val_loss: 0.7624 - val_accuracy: 0.7524
Epoch 37/100
41/41 [==============================] - 13s 321ms/step - loss: 0.8156 - accuracy: 0.7283 - val_loss: 0.7524 - val_accuracy: 0.7466
Epoch 38/100
41/41 [==============================] - 13s 316ms/step - loss: 0.8400 - accuracy: 0.7229 - val_loss: 0.6992 - val_accuracy: 0.7678
Epoch 39/100
41/41 [==============================] - 13s 322ms/step - loss: 0.8416 - accuracy: 0.7237 - val_loss: 0.6827 - val_accuracy: 0.7831
Epoch 40/100
41/41 [==============================] - 13s 324ms/step - loss: 0.7836 - accuracy: 0.7353 - val_loss: 0.6902 - val_accuracy: 0.7486
Epoch 41/100
41/41 [==============================] - 13s 321ms/step - loss: 0.7760 - accuracy: 0.7345 - val_loss: 0.7261 - val_accuracy: 0.7639
Epoch 42/100
41/41 [==============================] - 13s 316ms/step - loss: 0.7752 - accuracy: 0.7368 - val_loss: 0.6612 - val_accuracy: 0.7812
Epoch 43/100
41/41 [==============================] - 13s 324ms/step - loss: 0.7767 - accuracy: 0.7485 - val_loss: 0.6713 - val_accuracy: 0.7447
Epoch 44/100
41/41 [==============================] - 13s 325ms/step - loss: 0.7620 - accuracy: 0.7515 - val_loss: 0.6979 - val_accuracy: 0.7620
Epoch 45/100
41/41 [==============================] - 13s 323ms/step - loss: 0.7731 - accuracy: 0.7392 - val_loss: 0.6517 - val_accuracy: 0.7927
Epoch 46/100
41/41 [==============================] - 13s 324ms/step - loss: 0.7805 - accuracy: 0.7384 - val_loss: 0.6498 - val_accuracy: 0.7889
Epoch 47/100
41/41 [==============================] - 13s 325ms/step - loss: 0.7379 - accuracy: 0.7407 - val_loss: 0.6310 - val_accuracy: 0.7927
Epoch 48/100
41/41 [==============================] - 13s 324ms/step - loss: 0.7479 - accuracy: 0.7423 - val_loss: 0.7052 - val_accuracy: 0.7543
Epoch 49/100
41/41 [==============================] - 13s 325ms/step - loss: 0.7600 - accuracy: 0.7415 - val_loss: 0.6495 - val_accuracy: 0.7869
Epoch 50/100
41/41 [==============================] - 13s 325ms/step - loss: 0.6953 - accuracy: 0.7740 - val_loss: 0.7275 - val_accuracy: 0.7716
Epoch 51/100
41/41 [==============================] - 13s 320ms/step - loss: 0.7167 - accuracy: 0.7570 - val_loss: 0.6471 - val_accuracy: 0.8023
Epoch 52/100
41/41 [==============================] - 13s 325ms/step - loss: 0.7356 - accuracy: 0.7469 - val_loss: 0.5998 - val_accuracy: 0.8004
Epoch 53/100
41/41 [==============================] - 13s 328ms/step - loss: 0.7087 - accuracy: 0.7616 - val_loss: 0.6157 - val_accuracy: 0.7850
Epoch 54/100
41/41 [==============================] - 13s 328ms/step - loss: 0.7157 - accuracy: 0.7632 - val_loss: 0.6059 - val_accuracy: 0.7965
Epoch 55/100
41/41 [==============================] - 13s 328ms/step - loss: 0.6973 - accuracy: 0.7686 - val_loss: 0.5871 - val_accuracy: 0.7985
Epoch 56/100
41/41 [==============================] - 13s 327ms/step - loss: 0.7012 - accuracy: 0.7663 - val_loss: 0.5850 - val_accuracy: 0.7946
Epoch 57/100
41/41 [==============================] - 13s 329ms/step - loss: 0.6834 - accuracy: 0.7740 - val_loss: 0.6574 - val_accuracy: 0.7965
Epoch 58/100
41/41 [==============================] - 13s 326ms/step - loss: 0.6869 - accuracy: 0.7647 - val_loss: 0.5728 - val_accuracy: 0.8042
Epoch 59/100
41/41 [==============================] - 13s 326ms/step - loss: 0.6598 - accuracy: 0.7740 - val_loss: 0.5661 - val_accuracy: 0.8004
Epoch 60/100
41/41 [==============================] - 13s 328ms/step - loss: 0.6763 - accuracy: 0.7724 - val_loss: 0.5963 - val_accuracy: 0.8061
Epoch 61/100
41/41 [==============================] - 13s 328ms/step - loss: 0.6800 - accuracy: 0.7786 - val_loss: 0.6581 - val_accuracy: 0.7774
Epoch 62/100
41/41 [==============================] - 14s 331ms/step - loss: 0.6618 - accuracy: 0.7724 - val_loss: 0.6536 - val_accuracy: 0.7889
Epoch 63/100
41/41 [==============================] - 13s 330ms/step - loss: 0.6854 - accuracy: 0.7686 - val_loss: 0.5637 - val_accuracy: 0.8023
Epoch 64/100
41/41 [==============================] - 13s 330ms/step - loss: 0.6442 - accuracy: 0.7856 - val_loss: 0.5664 - val_accuracy: 0.7946
Epoch 65/100
41/41 [==============================] - 14s 331ms/step - loss: 0.6633 - accuracy: 0.7624 - val_loss: 0.5789 - val_accuracy: 0.8138
Epoch 66/100
41/41 [==============================] - 14s 333ms/step - loss: 0.6511 - accuracy: 0.7833 - val_loss: 0.6311 - val_accuracy: 0.7831
Epoch 67/100
41/41 [==============================] - 13s 330ms/step - loss: 0.6241 - accuracy: 0.7887 - val_loss: 0.6760 - val_accuracy: 0.7754
Epoch 68/100
41/41 [==============================] - 14s 333ms/step - loss: 0.6580 - accuracy: 0.7794 - val_loss: 0.5335 - val_accuracy: 0.8349
Epoch 69/100
41/41 [==============================] - 13s 328ms/step - loss: 0.6340 - accuracy: 0.7910 - val_loss: 0.5876 - val_accuracy: 0.8100
Epoch 70/100
41/41 [==============================] - 13s 326ms/step - loss: 0.6520 - accuracy: 0.7748 - val_loss: 0.5683 - val_accuracy: 0.8292
Epoch 71/100
41/41 [==============================] - 13s 327ms/step - loss: 0.6527 - accuracy: 0.7740 - val_loss: 0.6171 - val_accuracy: 0.7927
Epoch 72/100
41/41 [==============================] - 13s 329ms/step - loss: 0.6271 - accuracy: 0.7810 - val_loss: 0.5201 - val_accuracy: 0.8330
Epoch 73/100
41/41 [==============================] - 13s 325ms/step - loss: 0.6013 - accuracy: 0.7941 - val_loss: 0.5719 - val_accuracy: 0.8138
Epoch 74/100
41/41 [==============================] - 13s 328ms/step - loss: 0.6147 - accuracy: 0.7972 - val_loss: 0.6458 - val_accuracy: 0.7812
Epoch 75/100
41/41 [==============================] - 13s 329ms/step - loss: 0.6079 - accuracy: 0.7980 - val_loss: 0.6177 - val_accuracy: 0.7927
Epoch 76/100
41/41 [==============================] - 13s 328ms/step - loss: 0.6017 - accuracy: 0.7964 - val_loss: 0.5494 - val_accuracy: 0.8157
Epoch 77/100
41/41 [==============================] - 14s 331ms/step - loss: 0.6269 - accuracy: 0.7895 - val_loss: 0.5357 - val_accuracy: 0.8253
Epoch 78/100
41/41 [==============================] - 13s 331ms/step - loss: 0.5882 - accuracy: 0.7972 - val_loss: 0.6219 - val_accuracy: 0.7946
Epoch 79/100
41/41 [==============================] - 13s 331ms/step - loss: 0.6230 - accuracy: 0.7910 - val_loss: 0.5687 - val_accuracy: 0.7889
Epoch 80/100
41/41 [==============================] - 13s 330ms/step - loss: 0.6147 - accuracy: 0.7980 - val_loss: 0.5667 - val_accuracy: 0.8081
Epoch 81/100
41/41 [==============================] - 14s 332ms/step - loss: 0.6221 - accuracy: 0.8042 - val_loss: 0.5368 - val_accuracy: 0.8157
Epoch 82/100
41/41 [==============================] - 13s 330ms/step - loss: 0.6050 - accuracy: 0.7980 - val_loss: 0.5422 - val_accuracy: 0.8369
Epoch 83/100
41/41 [==============================] - 14s 334ms/step - loss: 0.5876 - accuracy: 0.7926 - val_loss: 0.5000 - val_accuracy: 0.8369
Epoch 84/100
41/41 [==============================] - 14s 347ms/step - loss: 0.5991 - accuracy: 0.7879 - val_loss: 0.5181 - val_accuracy: 0.8253
Epoch 85/100
41/41 [==============================] - 14s 342ms/step - loss: 0.5590 - accuracy: 0.8088 - val_loss: 0.5432 - val_accuracy: 0.8273
Epoch 86/100
41/41 [==============================] - 14s 349ms/step - loss: 0.5866 - accuracy: 0.8104 - val_loss: 0.5375 - val_accuracy: 0.8273
Epoch 87/100
41/41 [==============================] - 14s 354ms/step - loss: 0.5825 - accuracy: 0.8073 - val_loss: 0.5435 - val_accuracy: 0.8157
Epoch 88/100
41/41 [==============================] - 14s 340ms/step - loss: 0.5565 - accuracy: 0.8127 - val_loss: 0.5342 - val_accuracy: 0.8177
Epoch 89/100
41/41 [==============================] - 14s 344ms/step - loss: 0.5802 - accuracy: 0.8111 - val_loss: 0.5212 - val_accuracy: 0.8215
Epoch 90/100
41/41 [==============================] - 14s 345ms/step - loss: 0.5523 - accuracy: 0.8189 - val_loss: 0.5326 - val_accuracy: 0.8196
Epoch 91/100
41/41 [==============================] - 14s 334ms/step - loss: 0.5662 - accuracy: 0.8065 - val_loss: 0.5780 - val_accuracy: 0.8234
Epoch 92/100
41/41 [==============================] - 14s 331ms/step - loss: 0.5736 - accuracy: 0.8026 - val_loss: 0.5062 - val_accuracy: 0.8215
Epoch 93/100
41/41 [==============================] - 14s 332ms/step - loss: 0.5624 - accuracy: 0.8189 - val_loss: 0.5073 - val_accuracy: 0.8330
Epoch 94/100
41/41 [==============================] - 14s 340ms/step - loss: 0.5573 - accuracy: 0.7972 - val_loss: 0.6916 - val_accuracy: 0.7697
Epoch 95/100
41/41 [==============================] - 14s 342ms/step - loss: 0.5744 - accuracy: 0.7995 - val_loss: 0.5472 - val_accuracy: 0.8196
Epoch 96/100
41/41 [==============================] - 14s 333ms/step - loss: 0.5610 - accuracy: 0.8057 - val_loss: 0.6277 - val_accuracy: 0.7889
Epoch 97/100
41/41 [==============================] - 14s 340ms/step - loss: 0.5731 - accuracy: 0.8050 - val_loss: 0.5413 - val_accuracy: 0.8196
Epoch 98/100
41/41 [==============================] - 14s 343ms/step - loss: 0.5176 - accuracy: 0.8243 - val_loss: 0.4850 - val_accuracy: 0.8426
Epoch 99/100
41/41 [==============================] - 14s 338ms/step - loss: 0.5365 - accuracy: 0.8173 - val_loss: 0.4767 - val_accuracy: 0.8426
Epoch 100/100
41/41 [==============================] - 14s 348ms/step - loss: 0.5672 - accuracy: 0.8181 - val_loss: 0.5055 - val_accuracy: 0.8484
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

# Example usage
plot_metrics(accuracy, loss, val_accuracy, val_loss)
