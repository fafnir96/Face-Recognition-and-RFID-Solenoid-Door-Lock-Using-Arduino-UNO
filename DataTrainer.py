# Tulislah code untuk mentraining hasi dari data yang sudah didapatkan di bawah ini
import numpy as np
import cv2
import os
from PIL import Image

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('Cascades\haarcascade_frontalface_default.xml')

def getImageAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') #grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int (os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+h])
            ids.append(id)

    return faceSamples, ids

print("\n Training faces...")

faces,ids = getImageAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer/trainer.yml')

print("\n face trained".format(len(np.unique(ids))))
