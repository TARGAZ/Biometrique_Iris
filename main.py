import G6_iris_recognition
import cv2
from G6_iris_recognition import feature_vec
from G6_iris_recognition import iris_matching
from G6_iris_recognition import encode_iris_model
from G6_iris_recognition import main
import os
import pickle


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

    def resize_image(self):
        for i in range(len(self.StartPicture)):
            self.StartPicture[i] = cv2.resize(self.StartPicture[i], (320, 240))

    def encode_image(self, output_file):
        encoding_data = {'encodings': [], 'names': []}

        for image_file in self.StartPicturePath:
            image_path = os.path.join(self.StartPicturePath, image_file)

            feature_vec = G6_iris_recognition.iris_image_encoding("Intput_database/001L_1.png")

            encoding_data['encodings'].append(feature_vec)
            encoding_data['names'].append(image_file.split('.')[0])

        with open(output_file, 'wb') as f:
            pickle.dump(encoding_data, f)



output_file = "output_image.pickle"
image_count = count_images_in_folder("Intput_database")
iris_dec = iris_detection("Intput_database", image_count);
iris_dec.load_image_from_folder()
iris_dec.ShowImage()
iris_dec.resize_image()
iris_dec.ShowImage()
iris_dec.encode_image(output_file)