import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


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


    def ShowImage(self, indice):
        cv2.imshow("StartPicture", self.StartPicture[indice])
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

    def find_pattern(self, img_source, img_template):
        contour1, _ = cv2.findContours(self.StartPicture[img_source].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour2, _ = cv2.findContours(self.StartPicture[img_template].copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        matches = []
        for ctn1 in contour1:
            for ctn2 in contour2:
                match = cv2.matchShapes(ctn1, ctn2, cv2.CONTOURS_MATCH_I2, 0)
                if match < 0.1:
                    matches.append((ctn1, ctn2))

        img_fini1 = self.StartPicture[img_source].copy()
        img_fini2 = self.StartPicture[img_template].copy()

        for mt in matches:
            contour1, contour2 = mt
            x, y, w, h = cv2.boundingRect(contour1)
            img_fini1 = cv2.rectangle(img_fini1, (x, y), (x + w, y + h), (255, 255, 255), 2)
            x, y, w, h = cv2.boundingRect(contour2)
            img_fini2 = cv2.rectangle(img_fini2, (x, y), (x + w, y + h), (255, 255, 255), 2)

        cv2.imshow("Image 1", img_fini1)
        cv2.imshow("Image 2", img_fini2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def new_version_find_pattern(self, img_source, img_template):
        countours, hiearchy = cv2.findContours(self.StartPicture[img_template].copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        petit_contour = []
        for ctn in countours:
            x, y, w, h = cv2.boundingRect(ctn)
            if w < 738 and h < 387:
                contour_in_picture = self.StartPicture[img_template][y:y + h, x:x + w]
                petit_contour.append(contour_in_picture)

        for i in range(len(petit_contour)):
            res = cv2.matchTemplate(self.StartPicture[img_source], petit_contour[i], cv2.TM_SQDIFF)
            """
            plt.imshow(res, cmap='gray')
            plt.show()
            plt.imshow(self.StartPicture[img_source], cmap='gray')
            plt.show()
            plt.imshow(petit_contour[i], cmap='gray')
            plt.show()
            """
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = min_loc
            x, y, w, h = cv2.boundingRect(petit_contour[i])
            bottom_right = (top_left[0] + w + 2, top_left[1] + h + 2)
            cv2.rectangle(self.StartPicture[img_source], top_left, bottom_right, 255, 2)
            """
            plt.imshow(self.StartPicture[img_source], cmap='gray')
            plt.show()
            """

        cv2.imshow("Image 1", self.StartPicture[img_source])
        cv2.imshow("Image 2", self.StartPicture[img_template])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.imshow(res, cmap='gray')
        plt.show()
        plt.imshow(self.StartPicture[img_source], cmap='gray')
        plt.show()
        plt.imshow(self.StartPicture[img_template], cmap='gray')
        plt.show()
        pass


image_count = count_images_in_folder("Intput_database")
iris_dec = iris_detection("Intput_database", image_count)
iris_dec.load_image_from_folder()
iris_dec.better_picture()
iris_dec.iris_detection()
user_img_source = int(input("Enter the number of the source picture: "))
user_img_template = int(input("Enter the number of the template picture: "))
#iris_dec.find_pattern(user_img_source, user_img_template)
iris_dec.new_version_find_pattern(user_img_source, user_img_template)