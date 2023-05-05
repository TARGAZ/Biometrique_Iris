import cv2

class iris_detection():
    def __init__(self, StartPicturePath, ComparisonPicturePath):
        self.StartPicture = None
        self.ComparisonPicture = None
        self.pupil = None
        self.StartPicturePath = StartPicturePath
        self.ComparisonPicturePath = ComparisonPicturePath

    def load_image(self):
        self.StartPicture = cv2.imread(self.StartPicturePath)
        self.ComparisonPicture = cv2.imread(self.ComparisonPicturePath)
        if self.StartPicture is None or self.ComparisonPicture is None:
            print("Error: StartPicture not found")
            return False

    def ShowImage(self):
        cv2.imshow("StartPicture", self.StartPicture)
        cv2.imshow("ComparisonPicture", self.ComparisonPicture)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def ConvertToGray(self):
        self.StartPicture = cv2.cvtColor(self.StartPicture, cv2.COLOR_BGR2GRAY)
        self.ComparisonPicture = cv2.cvtColor(self.ComparisonPicture, cv2.COLOR_BGR2GRAY)

iris_dec = iris_detection("Iris_Picture/Iris1.jpg", "Iris_Picture/Iris2.jpg");
iris_dec.load_image()
iris_dec.ShowImage()
iris_dec.ConvertToGray()
iris_dec.ShowImage()