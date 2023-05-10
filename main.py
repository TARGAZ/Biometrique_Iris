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


class IrisDetection:
    def __init__(self, database_folder_path, image_count):
        self.iris_pictures = [None] * image_count
        self.database_folder_path = database_folder_path

    def load_images_from_folder(self):
        image_extensions = ['.jpg', '.jpeg', '.png']
        i = 0
        for filename in os.listdir(self.database_folder_path):
            file_extension = os.path.splitext(filename)[1].lower()

            if file_extension in image_extensions:
                file_path = os.path.join(self.database_folder_path, filename)
                self.iris_pictures[i] = cv2.imread(file_path)

            i += 1

    def enhance_pictures(self):
        for i in range(len(self.iris_pictures)):
            self.iris_pictures[i] = cv2.cvtColor(self.iris_pictures[i], cv2.COLOR_BGR2GRAY)
            self.iris_pictures[i] = cv2.equalizeHist(self.iris_pictures[i])
            self.iris_pictures[i] = cv2.GaussianBlur(self.iris_pictures[i], (5, 5), 0)
            self.iris_pictures[i] = cv2.Canny(self.iris_pictures[i], 100, 200)

    def find_pattern(self, source_img, template_img):
        img_source_copy = self.iris_pictures[source_img].copy()

        # Find the contours of the template
        contours, hierarchy = cv2.findContours(self.iris_pictures[template_img], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter the contours to only get the ones that are smaller than the source picture
        filtered_contours = [
            contour for contour in contours
            if 40 > cv2.boundingRect(contour)[2] > 6 and
               40 > cv2.boundingRect(contour)[3] > 6
        ]

        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            contour_in_picture = self.iris_pictures[template_img][y - 2:y + h + 2, x - 2:x + w + 2]
            res = cv2.matchTemplate(self.iris_pictures[source_img], contour_in_picture, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if min_val < 0.7:
                top_left = min_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(img_source_copy, top_left, bottom_right, 255, 2)

        cv2.imshow("Img_source", img_source_copy)
        cv2.imshow("Img_template", self.iris_pictures[template_img])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_pattern_2(self, source_img, template_img):
        img_source_copy = self.iris_pictures[source_img].copy()

        # Find the contours of the template
        contours_template, hierarchy = cv2.findContours(self.iris_pictures[template_img], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_source, hierarchy1 = cv2.findContours(self.iris_pictures[source_img], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter the contours to only get the ones that are smaller than the source picture
        filtered_contours_template = [
            contour for contour in contours_template
            if 40 > cv2.boundingRect(contour)[2] > 6 and
               40 > cv2.boundingRect(contour)[3] > 6
        ]

        filtered_contours_source = [
            contour for contour in contours_source
            if 40 > cv2.boundingRect(contour)[2] > 6 and
               40 > cv2.boundingRect(contour)[3] > 6
        ]

        for contour_template in filtered_contours_template:
            x, y, w, h = cv2.boundingRect(contour_template)
            contour_in_picture_template = self.iris_pictures[template_img][y - 2:y + h + 2, x - 2:x + w + 2]

            for contour_source in filtered_contours_source:
                x, y, w, h = cv2.boundingRect(contour_source)
                contour_in_picture_source = self.iris_pictures[source_img][y - 2:y + h + 2, x - 2:x + w + 2]

                if (contour_in_picture_source.shape[0] <= contour_in_picture_template.shape[0] and contour_in_picture_source.shape[1] <= contour_in_picture_template.shape[1]):
                    res = cv2.matchTemplate(contour_in_picture_source, contour_in_picture_template,cv2.TM_SQDIFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    if min_val < 0.7:
                        top_left = (min_loc[0] + x, min_loc[1] + y)
                        bottom_right = (top_left[0] + w, top_left[1] + h)
                        cv2.rectangle(img_source_copy, top_left, bottom_right, 255, 2)

        cv2.imshow("Img_source", img_source_copy)
        cv2.imshow("Img_template", self.iris_pictures[template_img])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


image_count = count_images_in_folder("Input_database")
iris_detection = IrisDetection("Input_database", image_count)
iris_detection.load_images_from_folder()
iris_detection.enhance_pictures()

while True:
    user_exit = input("Do you want to exit? (y/n): ")
    if user_exit == "y":
        break
    user_img_source = int(input("Enter the number of the source picture: "))
    user_img_template = int(input("Enter the number of the template picture: "))
    iris_detection.find_pattern(user_img_source, user_img_template)
    iris_detection.find_pattern_2(user_img_source, user_img_template)