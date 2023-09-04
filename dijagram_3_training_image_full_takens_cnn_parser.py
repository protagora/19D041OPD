import matplotlib.pyplot as plt
import re

output = """
Epoch 1/100
42/42 [==============================] - ETA: 0s - loss: 6.3342 - accuracy: 0.00372023-07-16 11:12:23.106233: I tensorflow/core/common_runtime/executor.cc:1210] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype int32
	 [[{{node Placeholder/_0}}]]
42/42 [==============================] - 29s 688ms/step - loss: 6.3342 - accuracy: 0.0037 - val_loss: 4.9040 - val_accuracy: 0.0063
Epoch 2/100
42/42 [==============================] - 29s 686ms/step - loss: 4.9165 - accuracy: 0.0112 - val_loss: 4.8961 - val_accuracy: 0.0076
Epoch 3/100
42/42 [==============================] - 29s 690ms/step - loss: 4.8912 - accuracy: 0.0105 - val_loss: 4.8081 - val_accuracy: 0.0114
Epoch 4/100
42/42 [==============================] - 29s 688ms/step - loss: 4.6477 - accuracy: 0.0164 - val_loss: 4.7432 - val_accuracy: 0.0164
Epoch 5/100
42/42 [==============================] - 29s 691ms/step - loss: 4.4282 - accuracy: 0.0262 - val_loss: 4.4603 - val_accuracy: 0.0442
Epoch 6/100
42/42 [==============================] - 29s 694ms/step - loss: 4.1054 - accuracy: 0.0523 - val_loss: 4.2770 - val_accuracy: 0.0834
Epoch 7/100
42/42 [==============================] - 29s 696ms/step - loss: 3.7467 - accuracy: 0.0949 - val_loss: 3.9272 - val_accuracy: 0.1757
Epoch 8/100
42/42 [==============================] - 29s 694ms/step - loss: 3.1080 - accuracy: 0.2018 - val_loss: 3.6595 - val_accuracy: 0.2339
Epoch 9/100
42/42 [==============================] - 29s 698ms/step - loss: 2.7313 - accuracy: 0.2855 - val_loss: 3.5122 - val_accuracy: 0.3198
Epoch 10/100
42/42 [==============================] - 29s 693ms/step - loss: 2.3873 - accuracy: 0.3490 - val_loss: 3.3724 - val_accuracy: 0.3451
Epoch 11/100
42/42 [==============================] - 29s 697ms/step - loss: 2.1045 - accuracy: 0.4036 - val_loss: 3.2175 - val_accuracy: 0.3578
Epoch 12/100
42/42 [==============================] - 29s 693ms/step - loss: 1.8370 - accuracy: 0.4836 - val_loss: 3.2244 - val_accuracy: 0.4020
Epoch 13/100
42/42 [==============================] - 29s 695ms/step - loss: 1.6030 - accuracy: 0.5366 - val_loss: 3.5564 - val_accuracy: 0.4046
Epoch 14/100
42/42 [==============================] - 29s 695ms/step - loss: 1.5023 - accuracy: 0.5516 - val_loss: 3.6163 - val_accuracy: 0.4286
Epoch 15/100
42/42 [==============================] - 29s 693ms/step - loss: 1.3644 - accuracy: 0.5800 - val_loss: 3.8455 - val_accuracy: 0.4362
Epoch 16/100
42/42 [==============================] - 29s 697ms/step - loss: 1.2391 - accuracy: 0.6166 - val_loss: 3.7435 - val_accuracy: 0.4437
Epoch 17/100
42/42 [==============================] - 29s 693ms/step - loss: 1.0773 - accuracy: 0.6689 - val_loss: 3.5358 - val_accuracy: 0.4248
Epoch 18/100
42/42 [==============================] - 29s 694ms/step - loss: 1.0391 - accuracy: 0.6824 - val_loss: 3.9692 - val_accuracy: 0.4501
Epoch 19/100
42/42 [==============================] - 29s 697ms/step - loss: 0.9728 - accuracy: 0.6906 - val_loss: 4.6255 - val_accuracy: 0.4589
Epoch 20/100
42/42 [==============================] - 29s 693ms/step - loss: 0.8824 - accuracy: 0.7227 - val_loss: 5.5376 - val_accuracy: 0.4627
Epoch 21/100
42/42 [==============================] - 29s 694ms/step - loss: 0.8589 - accuracy: 0.7294 - val_loss: 4.1260 - val_accuracy: 0.4855
Epoch 22/100
42/42 [==============================] - 29s 693ms/step - loss: 0.7803 - accuracy: 0.7504 - val_loss: 4.5477 - val_accuracy: 0.4817
Epoch 23/100
42/42 [==============================] - 29s 694ms/step - loss: 0.7785 - accuracy: 0.7504 - val_loss: 4.3745 - val_accuracy: 0.4779
Epoch 24/100
42/42 [==============================] - 29s 699ms/step - loss: 0.7174 - accuracy: 0.7758 - val_loss: 5.4857 - val_accuracy: 0.4753
Epoch 25/100
42/42 [==============================] - 29s 697ms/step - loss: 0.6446 - accuracy: 0.7848 - val_loss: 6.1401 - val_accuracy: 0.4324
Epoch 26/100
42/42 [==============================] - 29s 695ms/step - loss: 0.6340 - accuracy: 0.7930 - val_loss: 5.1992 - val_accuracy: 0.4855
Epoch 27/100
42/42 [==============================] - 29s 695ms/step - loss: 0.6162 - accuracy: 0.8124 - val_loss: 6.5972 - val_accuracy: 0.5171
Epoch 28/100
42/42 [==============================] - 29s 693ms/step - loss: 0.5687 - accuracy: 0.8094 - val_loss: 7.0457 - val_accuracy: 0.5171
Epoch 29/100
42/42 [==============================] - 29s 698ms/step - loss: 0.5690 - accuracy: 0.8102 - val_loss: 5.3697 - val_accuracy: 0.5284
Epoch 30/100
42/42 [==============================] - 29s 695ms/step - loss: 0.5363 - accuracy: 0.8281 - val_loss: 6.7794 - val_accuracy: 0.5234
Epoch 31/100
42/42 [==============================] - 29s 693ms/step - loss: 0.4916 - accuracy: 0.8386 - val_loss: 6.6533 - val_accuracy: 0.5196
Epoch 32/100
42/42 [==============================] - 29s 697ms/step - loss: 0.5055 - accuracy: 0.8416 - val_loss: 6.2635 - val_accuracy: 0.5310
Epoch 33/100
42/42 [==============================] - 29s 695ms/step - loss: 0.4534 - accuracy: 0.8460 - val_loss: 6.6196 - val_accuracy: 0.5373
Epoch 34/100
42/42 [==============================] - 29s 698ms/step - loss: 0.4382 - accuracy: 0.8468 - val_loss: 6.6699 - val_accuracy: 0.4741
Epoch 35/100
42/42 [==============================] - 29s 696ms/step - loss: 0.4222 - accuracy: 0.8610 - val_loss: 15.3787 - val_accuracy: 0.4817
Epoch 36/100
42/42 [==============================] - 29s 696ms/step - loss: 0.4226 - accuracy: 0.8610 - val_loss: 7.2454 - val_accuracy: 0.5297
Epoch 37/100
42/42 [==============================] - 29s 697ms/step - loss: 0.3872 - accuracy: 0.8714 - val_loss: 6.0749 - val_accuracy: 0.5487
Epoch 38/100
42/42 [==============================] - 29s 697ms/step - loss: 0.3734 - accuracy: 0.8789 - val_loss: 7.4771 - val_accuracy: 0.5411
Epoch 39/100
42/42 [==============================] - 29s 695ms/step - loss: 0.3575 - accuracy: 0.8886 - val_loss: 7.2009 - val_accuracy: 0.5525
Epoch 40/100
42/42 [==============================] - 29s 697ms/step - loss: 0.3213 - accuracy: 0.8842 - val_loss: 6.3681 - val_accuracy: 0.5360
Epoch 41/100
42/42 [==============================] - 29s 696ms/step - loss: 0.3448 - accuracy: 0.8827 - val_loss: 9.4124 - val_accuracy: 0.5436
Epoch 42/100
42/42 [==============================] - 30s 707ms/step - loss: 0.3276 - accuracy: 0.8909 - val_loss: 8.4893 - val_accuracy: 0.5537
Epoch 43/100
42/42 [==============================] - 29s 697ms/step - loss: 0.3004 - accuracy: 0.8954 - val_loss: 9.9034 - val_accuracy: 0.5664
Epoch 44/100
42/42 [==============================] - 29s 695ms/step - loss: 0.2568 - accuracy: 0.9081 - val_loss: 7.7900 - val_accuracy: 0.5474
Epoch 45/100
42/42 [==============================] - 29s 696ms/step - loss: 0.2819 - accuracy: 0.9118 - val_loss: 9.9462 - val_accuracy: 0.5575
Epoch 46/100
42/42 [==============================] - 29s 697ms/step - loss: 0.2594 - accuracy: 0.9170 - val_loss: 9.9654 - val_accuracy: 0.5664
Epoch 47/100
42/42 [==============================] - 29s 695ms/step - loss: 0.2864 - accuracy: 0.9066 - val_loss: 9.7251 - val_accuracy: 0.5664
Epoch 48/100
42/42 [==============================] - 29s 702ms/step - loss: 0.2618 - accuracy: 0.9103 - val_loss: 8.3029 - val_accuracy: 0.5613
Epoch 49/100
42/42 [==============================] - 29s 699ms/step - loss: 0.2465 - accuracy: 0.9178 - val_loss: 8.9995 - val_accuracy: 0.5461
Epoch 50/100
42/42 [==============================] - 29s 700ms/step - loss: 0.2434 - accuracy: 0.9238 - val_loss: 9.6033 - val_accuracy: 0.5575
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

    plt.savefig('dijagram_3_takens_plot_loss_accuracy_cnn.png')
    plt.show()


accuracy, loss, val_accuracy, val_loss = extract_metrics(output)

# print("Accuracy:", accuracy)
# print("Loss:", loss)
# print("Validation Accuracy:", val_accuracy)
# print("Validation Loss:", val_loss)

# Example usage
plot_metrics(accuracy, loss, val_accuracy, val_loss)
