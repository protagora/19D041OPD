import matplotlib.pyplot as plt
import re

output = """
Epoch 1/100
35/35 [==============================] - ETA: 0s - loss: 14.7440 - accuracy: 0.01182023-07-16 23:57:52.323948: I tensorflow/core/common_runtime/executor.cc:1210] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype int32
	 [[{{node Placeholder/_0}}]]
35/35 [==============================] - 18s 490ms/step - loss: 12.7440 - accuracy: 0.0118 - val_loss: 5.7965 - val_accuracy: 0.0133
Epoch 2/100
35/35 [==============================] - 17s 474ms/step - loss: 7.3363 - accuracy: 0.0082 - val_loss: 4.9035 - val_accuracy: 0.0133
Epoch 3/100
35/35 [==============================] - 17s 484ms/step - loss: 7.6967 - accuracy: 0.0109 - val_loss: 4.9030 - val_accuracy: 0.0133
Epoch 4/100
35/35 [==============================] - 17s 487ms/step - loss: 4.9079 - accuracy: 0.0091 - val_loss: 4.9018 - val_accuracy: 0.0133
Epoch 5/100
35/35 [==============================] - 18s 516ms/step - loss: 5.8194 - accuracy: 0.0091 - val_loss: 15.9807 - val_accuracy: 0.0067
Epoch 6/100
35/35 [==============================] - 23s 651ms/step - loss: 5.9518 - accuracy: 0.0100 - val_loss: 4.7768 - val_accuracy: 0.0800
Epoch 7/100
35/35 [==============================] - 22s 630ms/step - loss: 4.7900 - accuracy: 0.0462 - val_loss: 4.5112 - val_accuracy: 0.0833
Epoch 8/100
35/35 [==============================] - 22s 626ms/step - loss: 4.3410 - accuracy: 0.1088 - val_loss: 4.1577 - val_accuracy: 0.1433
Epoch 9/100
35/35 [==============================] - 22s 637ms/step - loss: 3.3780 - accuracy: 0.2765 - val_loss: 3.3425 - val_accuracy: 0.3467
Epoch 10/100
35/35 [==============================] - 21s 592ms/step - loss: 1.9522 - accuracy: 0.5512 - val_loss: 3.5897 - val_accuracy: 0.2967
Epoch 11/100
35/35 [==============================] - 21s 584ms/step - loss: 1.4894 - accuracy: 0.6537 - val_loss: 3.5063 - val_accuracy: 0.3767
Epoch 12/100
35/35 [==============================] - 25s 717ms/step - loss: 0.8585 - accuracy: 0.7978 - val_loss: 3.0943 - val_accuracy: 0.3667
Epoch 13/100
35/35 [==============================] - 22s 633ms/step - loss: 0.5112 - accuracy: 0.8858 - val_loss: 3.4257 - val_accuracy: 0.3767
Epoch 14/100
35/35 [==============================] - 22s 622ms/step - loss: 0.9169 - accuracy: 0.8268 - val_loss: 4.4087 - val_accuracy: 0.3833
Epoch 15/100
35/35 [==============================] - 23s 663ms/step - loss: 0.1969 - accuracy: 0.9483 - val_loss: 6.1038 - val_accuracy: 0.3333
Epoch 16/100
35/35 [==============================] - 26s 745ms/step - loss: 0.7842 - accuracy: 0.8441 - val_loss: 5.9091 - val_accuracy: 0.3967
Epoch 17/100
35/35 [==============================] - 20s 552ms/step - loss: 1.3497 - accuracy: 0.8404 - val_loss: 7.2965 - val_accuracy: 0.3967
Epoch 18/100
35/35 [==============================] - 17s 501ms/step - loss: 0.3198 - accuracy: 0.9347 - val_loss: 8.1758 - val_accuracy: 0.3933
Epoch 19/100
35/35 [==============================] - 18s 506ms/step - loss: 0.4349 - accuracy: 0.9220 - val_loss: 9.9067 - val_accuracy: 0.3000
Epoch 20/100
35/35 [==============================] - 17s 486ms/step - loss: 0.4386 - accuracy: 0.9202 - val_loss: 8.0011 - val_accuracy: 0.3667
Epoch 21/100
35/35 [==============================] - 17s 486ms/step - loss: 0.4995 - accuracy: 0.9148 - val_loss: 7.0580 - val_accuracy: 0.3867
Epoch 22/100
35/35 [==============================] - 17s 490ms/step - loss: 0.3044 - accuracy: 0.9420 - val_loss: 9.6126 - val_accuracy: 0.3700
Epoch 23/100
35/35 [==============================] - 18s 502ms/step - loss: 0.4128 - accuracy: 0.9275 - val_loss: 9.5271 - val_accuracy: 0.3900
Epoch 24/100
35/35 [==============================] - 17s 487ms/step - loss: 0.4064 - accuracy: 0.9347 - val_loss: 5.5727 - val_accuracy: 0.3833
Epoch 25/100
35/35 [==============================] - 18s 516ms/step - loss: 0.4605 - accuracy: 0.9393 - val_loss: 3.7759 - val_accuracy: 0.3400
Epoch 26/100
35/35 [==============================] - 17s 485ms/step - loss: 0.3066 - accuracy: 0.9492 - val_loss: 10.9506 - val_accuracy: 0.3833
Epoch 27/100
35/35 [==============================] - 17s 494ms/step - loss: 1.1010 - accuracy: 0.8921 - val_loss: 16.4697 - val_accuracy: 0.3867
Epoch 28/100
35/35 [==============================] - 17s 490ms/step - loss: 1.2409 - accuracy: 0.9275 - val_loss: 7.3699 - val_accuracy: 0.3800
Epoch 29/100
35/35 [==============================] - 17s 490ms/step - loss: 0.1831 - accuracy: 0.9755 - val_loss: 14.0240 - val_accuracy: 0.3900
Epoch 30/100
35/35 [==============================] - 17s 488ms/step - loss: 0.5582 - accuracy: 0.9393 - val_loss: 11.8918 - val_accuracy: 0.3800
Epoch 31/100
35/35 [==============================] - 17s 499ms/step - loss: 1.5692 - accuracy: 0.9157 - val_loss: 24.6337 - val_accuracy: 0.3100
Epoch 32/100
35/35 [==============================] - 17s 498ms/step - loss: 0.5739 - accuracy: 0.9474 - val_loss: 15.5221 - val_accuracy: 0.3900
Epoch 33/100
35/35 [==============================] - 17s 493ms/step - loss: 0.4568 - accuracy: 0.9510 - val_loss: 18.4216 - val_accuracy: 0.3833
Epoch 34/100
35/35 [==============================] - 18s 510ms/step - loss: 0.4097 - accuracy: 0.9501 - val_loss: 17.0175 - val_accuracy: 0.3567
Epoch 35/100
35/35 [==============================] - 19s 536ms/step - loss: 0.3367 - accuracy: 0.9610 - val_loss: 18.6435 - val_accuracy: 0.3733
Epoch 36/100
35/35 [==============================] - 17s 490ms/step - loss: 0.4043 - accuracy: 0.9465 - val_loss: 17.9911 - val_accuracy: 0.3733
Epoch 37/100
35/35 [==============================] - 17s 495ms/step - loss: 0.5047 - accuracy: 0.9565 - val_loss: 18.7097 - val_accuracy: 0.3767
Epoch 38/100
35/35 [==============================] - 18s 508ms/step - loss: 0.3781 - accuracy: 0.9619 - val_loss: 19.6841 - val_accuracy: 0.3867
Epoch 39/100
35/35 [==============================] - 19s 543ms/step - loss: 0.6005 - accuracy: 0.9556 - val_loss: 12.0226 - val_accuracy: 0.2267
Epoch 40/100
35/35 [==============================] - 23s 668ms/step - loss: 0.4628 - accuracy: 0.9601 - val_loss: 16.8372 - val_accuracy: 0.3833
Epoch 41/100
35/35 [==============================] - 23s 665ms/step - loss: 0.2740 - accuracy: 0.9701 - val_loss: 16.6647 - val_accuracy: 0.3900
Epoch 42/100
35/35 [==============================] - 22s 616ms/step - loss: 0.5398 - accuracy: 0.9492 - val_loss: 18.3543 - val_accuracy: 0.4033
Epoch 43/100
35/35 [==============================] - 24s 690ms/step - loss: 0.3091 - accuracy: 0.9710 - val_loss: 36.1645 - val_accuracy: 0.0733
Epoch 44/100
35/35 [==============================] - 21s 604ms/step - loss: 1.3629 - accuracy: 0.9411 - val_loss: 15.9506 - val_accuracy: 0.3733
Epoch 45/100
35/35 [==============================] - 21s 594ms/step - loss: 0.2659 - accuracy: 0.9683 - val_loss: 22.0343 - val_accuracy: 0.3700
Epoch 46/100
35/35 [==============================] - 24s 692ms/step - loss: 0.5829 - accuracy: 0.9411 - val_loss: 18.8391 - val_accuracy: 0.3667
Epoch 47/100
35/35 [==============================] - 23s 666ms/step - loss: 0.2625 - accuracy: 0.9719 - val_loss: 17.7261 - val_accuracy: 0.3633
Epoch 48/100
35/35 [==============================] - 24s 692ms/step - loss: 0.6421 - accuracy: 0.9592 - val_loss: 22.8408 - val_accuracy: 0.3700
Epoch 49/100
35/35 [==============================] - 26s 745ms/step - loss: 0.1187 - accuracy: 0.9918 - val_loss: 27.2037 - val_accuracy: 0.3733
Epoch 50/100
35/35 [==============================] - 23s 669ms/step - loss: 0.6531 - accuracy: 0.9592 - val_loss: 19.8559 - val_accuracy: 0.3000
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

    plt.savefig('dijagram_1_recurrence_plot_loss_accurary_cnn.png')
    plt.show()


accuracy, loss, val_accuracy, val_loss = extract_metrics(output)

# Example usage
plot_metrics(accuracy, loss, val_accuracy, val_loss)
