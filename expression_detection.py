import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

# keras imports
import keras
from keras import models
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils import to_categorical

data = pd.read_csv("icml_face_data.csv")

# making testing images for the other dataset.
jai_images_list = os.listdir("boy_face")
list_pixels = []
for i in jai_images_list:
    image = Image.open("boy_face/"+i).convert('L')
    image = image.resize((48, 48))
    list_pixels.append(np.array(image))
df = pd.DataFrame({"pixels": list_pixels})

# method to process images from all other datasets, to create prediction images.


def prepare_training(data_arr):
    array_image = np.zeros(shape=(len(data_arr), 48, 48))
    for i, row in data_arr.iterrows():
        image = np.array(row['pixels'])
        image = np.reshape(row['pixels'], (48, 48))
        array_image[i] = image
    return array_image


def prepare_data(data):
    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, ' pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label


emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
            3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# training image arrays.
train_image_array, train_image_label = prepare_data(
    data[data[' Usage'] == 'Training'])
val_image_array, val_image_label = prepare_data(
    data[data[' Usage'] == 'PrivateTest'])
test_image_array, test_image_label = prepare_data(
    data[data[' Usage'] == 'PublicTest'])
jai_training = prepare_training(df)

# training image arrays reshapedinto (1,1).
train_images = train_image_array.reshape(
    (train_image_array.shape[0], 48, 48, 1))
train_images = train_images.astype('float32')/255
val_images = val_image_array.reshape((val_image_array.shape[0], 48, 48, 1))
val_images = val_images.astype('float32')/255
test_images = test_image_array.reshape((test_image_array.shape[0], 48, 48, 1))
test_images = test_images.astype('float32')/255
jai_images = jai_training.reshape((jai_training.shape[0], 48, 48, 1))

# This is the categorical for the training set.
train_labels = to_categorical(train_image_label)
val_labels = to_categorical(val_image_label)
test_labels = to_categorical(test_image_label)

# imabalanced data, so we have to make sure the model learn equally.
# It equates the minority classes so that the model can learn equally
# Weight is measured by the presence of those labels and then divided by the total number of elements in the training set.
class_weight = dict(zip(range(0, 7), (((data[data[' Usage'] == 'Training']['emotion'].value_counts(
)).sort_index())/len(data[data[' Usage'] == 'Training']['emotion'])).tolist()))

# Neural Network
model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.summary() to get the details about the model.

# fitting function, to change epochs.
history = model.fit(train_images, train_labels,
                    validation_data=(val_images, val_labels),
                    class_weight=class_weight,
                    epochs=10,
                    batch_size=64)


# figure out what the hell this does.
test_loss, test_acc = model.evaluate(test_images, test_labels)

# predicted labels.
pred_test_labels = model.predict(test_images)

# plot function to test the output on fer, includes testing labels.
# def plot_image_and_emotion(test_image_array, test_image_label, pred_test_labels, image_number):
#     """ Function to plot the image and compare the prediction results with the label """

#     fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
#     bar_label = emotions.values()

#     axs[0].imshow(test_image_array[image_number], 'gray')
#     axs[0].set_title(emotions[test_image_label[image_number]])

#     axs[1].bar(bar_label, pred_test_labels[image_number])
#     print(pred_test_labels[image_number]*100)
#     axs[1].grid()

#     plt.savefig("{}_jai.png".format(image_number))

# plot function, does not include testing labels.


def plot_image_and_emotion(test_image_array, pred_test_labels, image_number):
    """ Function to plot the image and compare the prediction results with the label """

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    bar_label = emotions.values()

    axs[0].imshow(test_image_array[image_number], 'gray')

    axs[1].bar(bar_label, pred_test_labels[image_number])
    axs[1].grid()

    plt.savefig("figs/{}_jai.png".format(image_number))


# change value to get the resultant image to see correctness.
for i in range(len(jai_training)):
    plot_image_and_emotion(jai_training, pred_test_labels, i)
