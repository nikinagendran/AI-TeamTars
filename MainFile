import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
from keras.models import load_model  # TensorFlow is required for Keras to work
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Modelimproved.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Set up the main layout
        self.VBL = QVBoxLayout()

        # Create a label to display the video feed
        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        # Create a label to display the class name and confidence score
        self.InfoLabel = QLabel()
        self.VBL.addWidget(self.InfoLabel)

        # Create a button to stop the video feed
        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        # Create the Worker1 object
        self.Worker1 = Worker1()

        # Connect the signals and slots
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.Worker1.InfoUpdate.connect(self.InfoUpdateSlot)

        # Set the layout
        self.setLayout(self.VBL)

    # Slot to update the video feed
    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    # Slot to update the info label with class name and confidence score
    def InfoUpdateSlot(self, class_name, confidence_score):
        # Display the class name and confidence score
        self.InfoLabel.setText(f"Class: {class_name.strip()}, Confidence: {confidence_score * 100:.2f}%")

    # Method to stop the video feed
    def CancelFeed(self):
        self.Worker1.stop()


class Worker1(QThread):
    # Define signals for updating the video feed and info label
    ImageUpdate = pyqtSignal(QImage)
    InfoUpdate = pyqtSignal(str, float)

    def run(self):
        # Initialize video capture
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        
        # Process video frames in a loop
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                # Resize the frame
                picture = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                
                # Convert the frame to float32 and reshape
                picture = np.asarray(picture, dtype=np.float32).reshape(1, 224, 224, 3)

                # Normalize the image
                picture = (picture / 127.5) - 1

                # Make prediction
                prediction = model.predict(picture)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]

                # Emit the info update signal
                self.InfoUpdate.emit(class_name, confidence_score)

                # Convert the frame to RGB and flip
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)

                # Check for keyboard input (27 is the ASCII code for the ESC key)
                keyboard_input = cv2.waitKey(1)
                if keyboard_input == 27:
                    break

        # Release the video capture when stopping the thread
        Capture.release()

    def stop(self):
        # Stop the thread and end the video capture
        self.ThreadActive = False
        self.quit()


if __name__ == "__main__":
    # Start the PyQt application
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())
