import matplotlib.pyplot as plt
import re

output = """
Epoch 1/100
34/34 [==============================] - ETA: 0s - loss: 5.0194 - accuracy: 0.01122023-07-16 02:27:02.270313: I tensorflow/core/common_runtime/executor.cc:1210] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype int32
	 [[{{node Placeholder/_0}}]]
34/34 [==============================] - 21s 556ms/step - loss: 5.0194 - accuracy: 0.0112 - val_loss: 4.7156 - val_accuracy: 0.0299
Epoch 2/100
34/34 [==============================] - 18s 520ms/step - loss: 4.6530 - accuracy: 0.0327 - val_loss: 4.4100 - val_accuracy: 0.0672
Epoch 3/100
34/34 [==============================] - 17s 511ms/step - loss: 4.4033 - accuracy: 0.0579 - val_loss: 4.1740 - val_accuracy: 0.0784
Epoch 4/100
34/34 [==============================] - 18s 523ms/step - loss: 4.1983 - accuracy: 0.0738 - val_loss: 3.9881 - val_accuracy: 0.1007
Epoch 5/100
34/34 [==============================] - 17s 517ms/step - loss: 4.0185 - accuracy: 0.0999 - val_loss: 3.8821 - val_accuracy: 0.1119
Epoch 6/100
34/34 [==============================] - 18s 531ms/step - loss: 3.8691 - accuracy: 0.1195 - val_loss: 3.7295 - val_accuracy: 0.1343
Epoch 7/100
34/34 [==============================] - 18s 525ms/step - loss: 3.7550 - accuracy: 0.1345 - val_loss: 3.6121 - val_accuracy: 0.1493
Epoch 8/100
34/34 [==============================] - 18s 518ms/step - loss: 3.6201 - accuracy: 0.1382 - val_loss: 3.5589 - val_accuracy: 0.1381
Epoch 9/100
34/34 [==============================] - 19s 551ms/step - loss: 3.5634 - accuracy: 0.1457 - val_loss: 3.4942 - val_accuracy: 0.1716
Epoch 10/100
34/34 [==============================] - 19s 556ms/step - loss: 3.4902 - accuracy: 0.1578 - val_loss: 3.3924 - val_accuracy: 0.2090
Epoch 11/100
34/34 [==============================] - 19s 553ms/step - loss: 3.3856 - accuracy: 0.1671 - val_loss: 3.4041 - val_accuracy: 0.1604
Epoch 12/100
34/34 [==============================] - 19s 563ms/step - loss: 3.2869 - accuracy: 0.1811 - val_loss: 3.3961 - val_accuracy: 0.1679
Epoch 13/100
34/34 [==============================] - 19s 558ms/step - loss: 3.2696 - accuracy: 0.1867 - val_loss: 3.2313 - val_accuracy: 0.2015
Epoch 14/100
34/34 [==============================] - 19s 558ms/step - loss: 3.1836 - accuracy: 0.2035 - val_loss: 3.2475 - val_accuracy: 0.1791
Epoch 15/100
34/34 [==============================] - 19s 565ms/step - loss: 3.1729 - accuracy: 0.1989 - val_loss: 3.1519 - val_accuracy: 0.1903
Epoch 16/100
34/34 [==============================] - 19s 558ms/step - loss: 3.0907 - accuracy: 0.2073 - val_loss: 3.1566 - val_accuracy: 0.2015
Epoch 17/100
34/34 [==============================] - 19s 573ms/step - loss: 3.0465 - accuracy: 0.2204 - val_loss: 3.1360 - val_accuracy: 0.2090
Epoch 18/100
34/34 [==============================] - 19s 566ms/step - loss: 3.0345 - accuracy: 0.2316 - val_loss: 3.1089 - val_accuracy: 0.1940
Epoch 19/100
34/34 [==============================] - 19s 554ms/step - loss: 2.9469 - accuracy: 0.2362 - val_loss: 3.1464 - val_accuracy: 0.1940
Epoch 20/100
34/34 [==============================] - 19s 565ms/step - loss: 2.9503 - accuracy: 0.2334 - val_loss: 3.0511 - val_accuracy: 0.2127
Epoch 21/100
34/34 [==============================] - 19s 558ms/step - loss: 2.9214 - accuracy: 0.2222 - val_loss: 3.0164 - val_accuracy: 0.2313
Epoch 22/100
34/34 [==============================] - 19s 551ms/step - loss: 2.8556 - accuracy: 0.2502 - val_loss: 3.0269 - val_accuracy: 0.2015
Epoch 23/100
34/34 [==============================] - 19s 563ms/step - loss: 2.8433 - accuracy: 0.2418 - val_loss: 2.9509 - val_accuracy: 0.2201
Epoch 24/100
34/34 [==============================] - 19s 554ms/step - loss: 2.8019 - accuracy: 0.2633 - val_loss: 2.9836 - val_accuracy: 0.2239
Epoch 25/100
34/34 [==============================] - 19s 547ms/step - loss: 2.7890 - accuracy: 0.2568 - val_loss: 2.9572 - val_accuracy: 0.1940
Epoch 26/100
34/34 [==============================] - 19s 562ms/step - loss: 2.7757 - accuracy: 0.2838 - val_loss: 2.9568 - val_accuracy: 0.2351
Epoch 27/100
34/34 [==============================] - 19s 555ms/step - loss: 2.7336 - accuracy: 0.2661 - val_loss: 2.9447 - val_accuracy: 0.2164
Epoch 28/100
34/34 [==============================] - 19s 555ms/step - loss: 2.6833 - accuracy: 0.2810 - val_loss: 2.9481 - val_accuracy: 0.2052
Epoch 29/100
34/34 [==============================] - 19s 551ms/step - loss: 2.6566 - accuracy: 0.2969 - val_loss: 2.9074 - val_accuracy: 0.2313
Epoch 30/100
34/34 [==============================] - 19s 558ms/step - loss: 2.6778 - accuracy: 0.2885 - val_loss: 2.9454 - val_accuracy: 0.2276
Epoch 31/100
34/34 [==============================] - 19s 553ms/step - loss: 2.6758 - accuracy: 0.2885 - val_loss: 2.8671 - val_accuracy: 0.2313
Epoch 32/100
34/34 [==============================] - 19s 555ms/step - loss: 2.6508 - accuracy: 0.2923 - val_loss: 2.9048 - val_accuracy: 0.2276
Epoch 33/100
34/34 [==============================] - 19s 555ms/step - loss: 2.6239 - accuracy: 0.3035 - val_loss: 2.8454 - val_accuracy: 0.2425
Epoch 34/100
34/34 [==============================] - 19s 553ms/step - loss: 2.5747 - accuracy: 0.3007 - val_loss: 2.8329 - val_accuracy: 0.2425
Epoch 35/100
34/34 [==============================] - 19s 555ms/step - loss: 2.5431 - accuracy: 0.3119 - val_loss: 2.8590 - val_accuracy: 0.2239
Epoch 36/100
34/34 [==============================] - 19s 546ms/step - loss: 2.5211 - accuracy: 0.3007 - val_loss: 2.8066 - val_accuracy: 0.2724
Epoch 37/100
34/34 [==============================] - 19s 550ms/step - loss: 2.5613 - accuracy: 0.3277 - val_loss: 2.8315 - val_accuracy: 0.2500
Epoch 38/100
34/34 [==============================] - 19s 549ms/step - loss: 2.5495 - accuracy: 0.3128 - val_loss: 2.7943 - val_accuracy: 0.2239
Epoch 39/100
34/34 [==============================] - 18s 538ms/step - loss: 2.4982 - accuracy: 0.3091 - val_loss: 3.0332 - val_accuracy: 0.1940
Epoch 40/100
34/34 [==============================] - 19s 555ms/step - loss: 2.4678 - accuracy: 0.3324 - val_loss: 2.8639 - val_accuracy: 0.2239
Epoch 41/100
34/34 [==============================] - 19s 561ms/step - loss: 2.4492 - accuracy: 0.3259 - val_loss: 2.8323 - val_accuracy: 0.2313
Epoch 42/100
34/34 [==============================] - 19s 561ms/step - loss: 2.4529 - accuracy: 0.3315 - val_loss: 2.8332 - val_accuracy: 0.2090
Epoch 43/100
34/34 [==============================] - 19s 559ms/step - loss: 2.4141 - accuracy: 0.3557 - val_loss: 2.9071 - val_accuracy: 0.2239
Epoch 44/100
34/34 [==============================] - 19s 556ms/step - loss: 2.3919 - accuracy: 0.3408 - val_loss: 2.8827 - val_accuracy: 0.2351
Epoch 45/100
34/34 [==============================] - 19s 562ms/step - loss: 2.4025 - accuracy: 0.3501 - val_loss: 2.7909 - val_accuracy: 0.2575
Epoch 46/100
34/34 [==============================] - 19s 556ms/step - loss: 2.3731 - accuracy: 0.3473 - val_loss: 2.8608 - val_accuracy: 0.2388
Epoch 47/100
34/34 [==============================] - 19s 559ms/step - loss: 2.3691 - accuracy: 0.3529 - val_loss: 2.7553 - val_accuracy: 0.2724
Epoch 48/100
34/34 [==============================] - 18s 544ms/step - loss: 2.3469 - accuracy: 0.3707 - val_loss: 2.8046 - val_accuracy: 0.2463
Epoch 49/100
34/34 [==============================] - 18s 538ms/step - loss: 2.3660 - accuracy: 0.3408 - val_loss: 3.0464 - val_accuracy: 0.2313
Epoch 50/100
34/34 [==============================] - 18s 546ms/step - loss: 2.3260 - accuracy: 0.3688 - val_loss: 2.7712 - val_accuracy: 0.2351
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

    plt.savefig('dijagram_1_image_full_recurrence_loss_accuracy_plot.png')
    plt.show()


accuracy, loss, val_accuracy, val_loss = extract_metrics(output)

# Example usage
plot_metrics(accuracy, loss, val_accuracy, val_loss)
