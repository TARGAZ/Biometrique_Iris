import cv2
import os
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
    def __init__(self, DatabaseFolderPath, image_count):
        self.IrisPicture = [None] * image_count
        self.DatabaseFolderPath = DatabaseFolderPath

    def load_image_from_folder(self):
        image_extensions = ['.jpg', '.jpeg', '.png']
        i = 0
        for filename in os.listdir(self.DatabaseFolderPath):
            file_extension = os.path.splitext(filename)[1].lower()

            if file_extension in image_extensions:
                file_path = os.path.join(self.DatabaseFolderPath, filename)
                self.IrisPicture[i] = cv2.imread(file_path)

            i += 1
    def better_picture(self): #use open cv to make the picture more usable for iris detection
        for i in range(len(self.IrisPicture)):
            self.IrisPicture[i] = cv2.cvtColor(self.IrisPicture[i], cv2.COLOR_BGR2GRAY)
            self.IrisPicture[i] = cv2.equalizeHist(self.IrisPicture[i])
            self.IrisPicture[i] = cv2.GaussianBlur(self.IrisPicture[i], (5, 5), 0)
            self.IrisPicture[i] = cv2.Canny(self.IrisPicture[i], 100, 200)

    def find_pattern(self, img_source, img_template):
        img_source_copy = self.IrisPicture[img_source].copy()
        #find the contours of the template
        contours, hierarchy = cv2.findContours(self.IrisPicture[img_template], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #filter the contours to only get the ones that are smaller than the source picture
        filtered_contours = [
            contour for contour in contours
            if cv2.boundingRect(contour)[2] < 40 and cv2.boundingRect(contour)[2] > 9 and cv2.boundingRect(contour)[3] < 40 and cv2.boundingRect(contour)[3] > 10
        ]

        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            contour_in_picture = self.IrisPicture[img_template][y - 2:y + h + 2, x - 2:x + w + 2]
            plt.imshow(contour_in_picture, cmap='gray')
            plt.show()
            res = cv2.matchTemplate(self.IrisPicture[img_source], contour_in_picture, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if min_val < 1:
                top_left = min_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(img_source_copy, top_left, bottom_right, 255, 2)


        cv2.imshow("Img_source", img_source_copy)
        cv2.imshow("Img_template", self.IrisPicture[img_template])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


image_count = count_images_in_folder("Intput_database")
iris_dec = iris_detection("Intput_database", image_count)
iris_dec.load_image_from_folder()
iris_dec.better_picture()
while True:
    user_exit = input("Do you want to exit? (y/n): ")
    if user_exit == "y":
        break
    user_img_source = int(input("Enter the number of the source picture: "))
    user_img_template = int(input("Enter the number of the template picture: "))
    iris_dec.find_pattern(user_img_source, user_img_template)