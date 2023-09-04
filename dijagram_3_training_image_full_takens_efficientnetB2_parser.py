import matplotlib.pyplot as plt
import re

output = """
Epoch 1/100
42/42 [==============================] - ETA: 0s - loss: 4.5658 - accuracy: 0.04582023-07-16 06:42:33.411167: I tensorflow/core/common_runtime/executor.cc:1210] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype int32
	 [[{{node Placeholder/_0}}]]
42/42 [==============================] - 29s 638ms/step - loss: 4.5658 - accuracy: 0.0458 - val_loss: 4.0439 - val_accuracy: 0.1336
Epoch 2/100
42/42 [==============================] - 26s 633ms/step - loss: 3.7441 - accuracy: 0.1059 - val_loss: 3.7183 - val_accuracy: 0.1601
Epoch 3/100
42/42 [==============================] - 26s 627ms/step - loss: 3.3774 - accuracy: 0.1757 - val_loss: 3.6474 - val_accuracy: 0.2070
Epoch 4/100
42/42 [==============================] - 26s 626ms/step - loss: 3.1634 - accuracy: 0.1862 - val_loss: 3.5719 - val_accuracy: 0.2056
Epoch 5/100
42/42 [==============================] - 26s 629ms/step - loss: 2.9920 - accuracy: 0.2297 - val_loss: 3.5653 - val_accuracy: 0.2100
Epoch 6/100
42/42 [==============================] - 26s 632ms/step - loss: 2.8475 - accuracy: 0.2357 - val_loss: 3.5329 - val_accuracy: 0.2394
Epoch 7/100
42/42 [==============================] - 26s 629ms/step - loss: 2.7112 - accuracy: 0.2492 - val_loss: 3.4209 - val_accuracy: 0.2570
Epoch 8/100
42/42 [==============================] - 26s 626ms/step - loss: 2.7004 - accuracy: 0.2635 - val_loss: 3.3886 - val_accuracy: 0.3142
Epoch 9/100
42/42 [==============================] - 27s 651ms/step - loss: 2.5673 - accuracy: 0.2958 - val_loss: 3.3436 - val_accuracy: 0.3348
Epoch 10/100
42/42 [==============================] - 27s 639ms/step - loss: 2.5307 - accuracy: 0.3153 - val_loss: 3.4051 - val_accuracy: 0.3260
Epoch 11/100
42/42 [==============================] - 28s 668ms/step - loss: 2.4464 - accuracy: 0.3146 - val_loss: 3.3739 - val_accuracy: 0.3465
Epoch 12/100
42/42 [==============================] - 27s 646ms/step - loss: 2.4399 - accuracy: 0.3198 - val_loss: 3.3954 - val_accuracy: 0.3510
Epoch 13/100
42/42 [==============================] - 27s 652ms/step - loss: 2.3023 - accuracy: 0.3311 - val_loss: 3.3390 - val_accuracy: 0.3598
Epoch 14/100
42/42 [==============================] - 27s 653ms/step - loss: 2.3367 - accuracy: 0.3559 - val_loss: 3.2980 - val_accuracy: 0.3627
Epoch 15/100
42/42 [==============================] - 27s 650ms/step - loss: 2.2502 - accuracy: 0.3626 - val_loss: 3.2931 - val_accuracy: 0.3627
Epoch 16/100
42/42 [==============================] - 27s 659ms/step - loss: 2.2404 - accuracy: 0.3679 - val_loss: 3.3359 - val_accuracy: 0.3480
Epoch 17/100
42/42 [==============================] - 27s 649ms/step - loss: 2.2571 - accuracy: 0.3589 - val_loss: 3.3586 - val_accuracy: 0.3789
Epoch 18/100
42/42 [==============================] - 27s 650ms/step - loss: 2.1374 - accuracy: 0.3859 - val_loss: 3.3973 - val_accuracy: 0.4068
Epoch 19/100
42/42 [==============================] - 27s 637ms/step - loss: 2.1452 - accuracy: 0.3701 - val_loss: 3.3835 - val_accuracy: 0.3906
Epoch 20/100
42/42 [==============================] - 27s 645ms/step - loss: 2.0671 - accuracy: 0.3926 - val_loss: 3.3823 - val_accuracy: 0.3744
Epoch 21/100
42/42 [==============================] - 27s 654ms/step - loss: 2.0786 - accuracy: 0.4054 - val_loss: 3.3842 - val_accuracy: 0.3994
Epoch 22/100
42/42 [==============================] - 28s 670ms/step - loss: 2.0671 - accuracy: 0.3994 - val_loss: 3.5027 - val_accuracy: 0.3847
Epoch 23/100
42/42 [==============================] - 30s 717ms/step - loss: 2.0184 - accuracy: 0.4189 - val_loss: 3.3865 - val_accuracy: 0.4156
Epoch 24/100
42/42 [==============================] - 30s 726ms/step - loss: 1.9918 - accuracy: 0.4137 - val_loss: 3.5093 - val_accuracy: 0.4244
Epoch 25/100
42/42 [==============================] - 31s 744ms/step - loss: 2.0048 - accuracy: 0.4242 - val_loss: 3.5048 - val_accuracy: 0.4023
Epoch 26/100
42/42 [==============================] - 31s 746ms/step - loss: 1.9323 - accuracy: 0.4422 - val_loss: 3.6776 - val_accuracy: 0.4244
Epoch 27/100
42/42 [==============================] - 31s 754ms/step - loss: 1.9063 - accuracy: 0.4362 - val_loss: 3.7782 - val_accuracy: 0.4023
Epoch 28/100
42/42 [==============================] - 32s 767ms/step - loss: 1.9057 - accuracy: 0.4114 - val_loss: 3.6828 - val_accuracy: 0.4376
Epoch 29/100
42/42 [==============================] - 33s 787ms/step - loss: 1.9048 - accuracy: 0.4527 - val_loss: 3.7915 - val_accuracy: 0.3965
Epoch 30/100
42/42 [==============================] - 32s 773ms/step - loss: 1.9016 - accuracy: 0.4332 - val_loss: 3.6060 - val_accuracy: 0.4141
Epoch 31/100
42/42 [==============================] - 31s 756ms/step - loss: 1.8312 - accuracy: 0.4520 - val_loss: 3.5053 - val_accuracy: 0.4347
Epoch 32/100
42/42 [==============================] - 33s 791ms/step - loss: 1.8284 - accuracy: 0.4557 - val_loss: 3.5804 - val_accuracy: 0.4581
Epoch 33/100
42/42 [==============================] - 33s 803ms/step - loss: 1.8479 - accuracy: 0.4565 - val_loss: 3.6572 - val_accuracy: 0.4449
Epoch 34/100
42/42 [==============================] - 33s 794ms/step - loss: 1.8410 - accuracy: 0.4497 - val_loss: 3.8336 - val_accuracy: 0.4420
Epoch 35/100
42/42 [==============================] - 29s 696ms/step - loss: 1.8258 - accuracy: 0.4632 - val_loss: 3.6469 - val_accuracy: 0.4552
Epoch 36/100
42/42 [==============================] - 29s 701ms/step - loss: 1.7748 - accuracy: 0.4640 - val_loss: 3.7518 - val_accuracy: 0.4552
Epoch 37/100
42/42 [==============================] - 29s 691ms/step - loss: 1.7944 - accuracy: 0.4655 - val_loss: 3.9652 - val_accuracy: 0.4317
Epoch 38/100
42/42 [==============================] - 29s 698ms/step - loss: 1.8159 - accuracy: 0.4685 - val_loss: 3.9676 - val_accuracy: 0.4258
Epoch 39/100
42/42 [==============================] - 29s 694ms/step - loss: 1.7329 - accuracy: 0.5030 - val_loss: 3.8486 - val_accuracy: 0.4537
Epoch 40/100
42/42 [==============================] - 29s 695ms/step - loss: 1.7220 - accuracy: 0.4857 - val_loss: 3.6776 - val_accuracy: 0.4449
Epoch 41/100
42/42 [==============================] - 788s 19s/step - loss: 1.7093 - accuracy: 0.4977 - val_loss: 3.7970 - val_accuracy: 0.4552
Epoch 42/100
42/42 [==============================] - 31s 736ms/step - loss: 1.7622 - accuracy: 0.4805 - val_loss: 3.8898 - val_accuracy: 0.4699
Epoch 43/100
42/42 [==============================] - 33s 794ms/step - loss: 1.7452 - accuracy: 0.4752 - val_loss: 4.1972 - val_accuracy: 0.4640
Epoch 44/100
42/42 [==============================] - 34s 804ms/step - loss: 1.6819 - accuracy: 0.4970 - val_loss: 4.0281 - val_accuracy: 0.4758
Epoch 45/100
42/42 [==============================] - 34s 814ms/step - loss: 1.6639 - accuracy: 0.5090 - val_loss: 4.0554 - val_accuracy: 0.4802
Epoch 46/100
42/42 [==============================] - 33s 783ms/step - loss: 1.6112 - accuracy: 0.5113 - val_loss: 3.9380 - val_accuracy: 0.4875
Epoch 47/100
42/42 [==============================] - 30s 722ms/step - loss: 1.6661 - accuracy: 0.5008 - val_loss: 3.9172 - val_accuracy: 0.4640
Epoch 48/100
42/42 [==============================] - 30s 722ms/step - loss: 1.7427 - accuracy: 0.4940 - val_loss: 3.9008 - val_accuracy: 0.4655
Epoch 49/100
42/42 [==============================] - 30s 711ms/step - loss: 1.7058 - accuracy: 0.4917 - val_loss: 3.9587 - val_accuracy: 0.4640
Epoch 50/100
42/42 [==============================] - 30s 721ms/step - loss: 1.6701 - accuracy: 0.5023 - val_loss: 3.8679 - val_accuracy: 0.4523
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

    plt.savefig('dijagram_3_takens_plot_loss_accuracy_efficientnet.png')
    plt.show()


accuracy, loss, val_accuracy, val_loss = extract_metrics(output)

# print("Accuracy:", accuracy)
# print("Loss:", loss)
# print("Validation Accuracy:", val_accuracy)
# print("Validation Loss:", val_loss)

# Example usage
plot_metrics(accuracy, loss, val_accuracy, val_loss)
