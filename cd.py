import cv2
import os
import random
import pickle
import numpy as np

DATADIR = "kagglecatsanddogs_3367a/PetImages"
CATEGORIES = ["Dog","Cat"]# 0-cat 1-dog

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(100,100))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)

X=[]
Y=[]

for features,labels in training_data:
    X.append(features)
    Y.append(labels)

X = np.array(X).reshape(-1,100,100,1)

pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle","wb")
pickle.dump(Y,pickle_out)
pickle_out.close()

# for picture in training_data[:15]:
#     print(picture[1])
#
#         print(img_array)
#         break
#     break
# print(img_array.shape)
# new_array = cv2.resize(img_array,(50,50))
# print(new_array)
# print(new_array.shape)
