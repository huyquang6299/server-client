from sklearn.metrics import accuracy_score
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, AveragePooling2D
import cv2
from skimage import transform
from sklearn.model_selection import train_test_split

# from kerastuner.tuners import RandomSearch

FAST_RUN = False
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3  # RGB color


def CNN_Classification():
  model = Sequential()
#layer 1
  model.add(Conv2D(16, (11, 11), strides=(4, 4), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(3, 3),strides=(2, 2) ))
  model.add(Dropout(0.25))
#layer 2
  model.add(Conv2D(20, (5, 5), activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
  model.add(Dropout(0.25))
#layer 3
  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
  model.add(Dropout(0.25))
#layer 4
  model.add(Flatten())
  model.add(Dense(48, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(Dense(2, activation='softmax'))

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  #Save Model
  CarsClassification_json =model.to_json()
  with open(f'modelC9.json', "w") as json_file:
      json_file.write(CarsClassification_json)
      json_file.close()
  model.summary()
  print(model.summary())
  return model
#
# model = CNN_Classification()

def Load_Image(pos_dir, neg_dir):
    train_images = []
    Lable_Image = []
    count = 0
    Images_Positive = os.listdir(pos_dir)
    n = len(Images_Positive)
    for image in Images_Positive:
        count += 1
        path = os.path.join(pos_dir, image)
        img = cv2.imread(path)
        train_images.append(transform.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
        l = [1, 0]
        Lable_Image.append(l)
        print(f"Loading {pos_dir}: {count}/{n}")

    count = 0
    Images_Negative = os.listdir(neg_dir)
    n = len(Images_Negative)
    for image in Images_Negative:
        count += 1
        path = os.path.join(neg_dir, image)
        img = cv2.imread(path)
        #       print(path)
        train_images.append(transform.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
        l = [0, 1]
        Lable_Image.append(l)
        print(f"Loading {neg_dir}: {count}/{n}")
    # np.save("FileOld/train_image_hustpark",train_images)
    # np.save("FileOld/train_label_hustpark",Lable_Image)
    return np.array(train_images), np.array(Lable_Image)


def Train_Model(pos_dir, neg_dir):
    Train_Image, Lable_Image = Load_Image(pos_dir, neg_dir)
    print("Train data size: ", len(Train_Image))
    CNN = CNN_Classification()
    print("Train data shape: ", Train_Image.shape)
    # idx = np.random.permutation(Train_Image.shape[0])
    CNN.fit(Train_Image, Lable_Image, batch_size=32, epochs=10,validation_split=0.2)
    # Save weight
    CNN.save_weights(f'FileOld/weightC9_newmodel_databri.h5')


if __name__ == "__main__":
    pos_dir = "DataNew95 Bri/busy"
    neg_dir = "DataNew95 Bri/free"

    Train_Model(pos_dir, neg_dir)


