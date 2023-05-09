import cv2
import os
import numpy as np


def count_images_in_folder(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_count = 0

    for filename in os.listdir(folder_path):
        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension in image_extensions:
            image_count += 1

    return image_count
class iris_detection():
    def __init__(self, StartPicturePath, image_count):
        self.StartPicture = [None] * image_count
        self.StartPicturePath = StartPicturePath

    def load_image_from_folder(self):
        image_extensions = ['.jpg', '.jpeg', '.png']
        i = 0
        for filename in os.listdir(self.StartPicturePath):
            file_extension = os.path.splitext(filename)[1].lower()

            if file_extension in image_extensions:
                file_path = os.path.join(self.StartPicturePath, filename)
                self.StartPicture[i] = cv2.imread(file_path)

            i += 1


    def ShowImage(self):
        cv2.imshow("StartPicture", self.StartPicture[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def better_picture(self):
        for i in range(len(self.StartPicture)):
            self.StartPicture[i] = cv2.cvtColor(self.StartPicture[i], cv2.COLOR_BGR2GRAY)
            self.StartPicture[i] = cv2.equalizeHist(self.StartPicture[i])
            self.StartPicture[i] = cv2.GaussianBlur(self.StartPicture[i], (5, 5), 0)

    def iris_detection(self):
        for i in range(len(self.StartPicture)):
            self.StartPicture[i] = cv2.Canny(self.StartPicture[i], 100, 200)

image_count = count_images_in_folder("Intput_database")
iris_dec = iris_detection("Intput_database", image_count);
iris_dec.load_image_from_folder()
iris_dec.ShowImage()
iris_dec.ShowImage()
iris_dec.better_picture()
iris_dec.ShowImage()
iris_dec.iris_detection()
iris_dec.ShowImage()