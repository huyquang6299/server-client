import numpy as np
import cv2
import os
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import model_from_json
from skimage import transform

json_file_noBright = open('FileOld/modelC9.json', 'r')
loaded_model_json_noBright = json_file_noBright.read()
json_file_noBright.close()
Loaded_Model_noBright = model_from_json(loaded_model_json_noBright)
# Load weights into new model
Loaded_Model_noBright.load_weights("FileOld/weightC9f.h5")
print("Loaded_noBright")

path = 'Slot'
Predict_Slot_NoBri = []
Real_Slot = []
filename_array = []
count = 0
for slot in os.listdir(path):
    for status in os.listdir(f"{path}/{slot}"):
        for imgPath in os.listdir(f"{path}/{slot}/{status}"):
            images = []
            img = cv2.imread(f"{path}/{slot}/{status}/{imgPath}")
            imgCopy = img.copy()
            images.append(transform.resize(img, (150, 150, 3)))
            images = np.array(images)
            predictNoBri = Loaded_Model_noBright.predict(images)[0][0]
            #######NoBri
            if predictNoBri > 0.5:
                Predict_Slot_NoBri.append(1)
            else:
                Predict_Slot_NoBri.append(0)
            if "busy" in status:
                Real_Slot.append(1)
            else:
                Real_Slot.append(0)
            count += 1
            print(f"Loading {status} {slot}: {count}")
    break
accuracy = accuracy_score(Real_Slot,Predict_Slot_NoBri)
print("Acc: ",accuracy)

print("Finish !")
