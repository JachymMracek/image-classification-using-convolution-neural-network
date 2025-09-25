from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPoint
from PIL import Image
from build_classificator import Model
from build_dataset import Maping
import torch
import os
from collections import defaultdict
import json 
import sys

#######################################################################################################################

# SOURCES

    #  https://stackoverflow.com/questions/6784084/how-to-pass-arguments-to-functions-by-the-click-of-button-in-pyqt
    #  https://stackoverflow.com/questions/75465473/how-to-make-round-edges-for-the-main-window-in-pyqt?utm
    #  https://forum.qt.io/topic/113585/change-qpushbutton-opacity-on-hover-and-pressed-states/
    #  https://doc.qt.io/qt-6/stylesheet-syntax.html#selector-types
    #  https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QWidget.html#PySide6.QtWidgets.QWidget.setStyleSheet
    #  https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QLineEdit.html#PySide6.QtWidgets.PySide6.QtWidgets.QLineEdit.setPlaceholderText
    #  https://www.youtube.com/watch?v=PGBle4B0UyQ
    #  https://groups.google.com/g/python_inside_maya/c/Z-Jh_uzPGe4


#######################################################################################################################


# PROPERTIES

class Button(QPushButton):

    """parent class for each button"""

    def __init__(self, text, function, parent):
        super().__init__(text, parent)
        self.clicked.connect(function)

class StartButton(Button):

    """button for entering page, where user can add url off image"""

    def __init__(self, parent, function, TEXT="Start",WIDTH = 300,HEIGHT = 100):
        super().__init__(TEXT, function, parent)

        self.setGeometry(MainWindow.WIDTH // 2 - WIDTH // 2,(MainWindow.HEIGHT - HEIGHT) // 2 - HEIGHT // 2 - HEIGHT * 0.2,WIDTH,HEIGHT)

        self.setStyleSheet("""QPushButton {background-color:rgb(34, 197, 94);color:black;border-radius: 15px;}QPushButton:hover {background-color:rgba(0, 0, 0, 125);};""")

        self.show()

class EndButton(Button):

    """button for ending UI and closing it"""

    def __init__(self, parent, function, TEXT="End",WIDTH = 300,HEIGHT = 100):
        super().__init__(TEXT, function, parent)

        self.setGeometry(MainWindow.WIDTH // 2 - WIDTH // 2,(MainWindow.HEIGHT + HEIGHT) // 2 - HEIGHT // 2 + HEIGHT * 0.2,WIDTH,HEIGHT)

        self.setStyleSheet("""QPushButton {background-color:rgb(220, 38, 38);color:black;border-radius: 15px;}QPushButton:hover {background-color:rgba(0, 0, 0, 125);}""")

        self.show()

class BackButton(Button):

    """Button which will get user to previous page"""

    def __init__(self, parent, function, TEXT="Back",WIDTH = 300,HEIGHT = 100):
        super().__init__(TEXT, function, parent)

        self.setGeometry(MainWindow.WIDTH // 2 - WIDTH // 2,(MainWindow.HEIGHT + HEIGHT) // 2 - HEIGHT // 2 + HEIGHT * 0.2,WIDTH,HEIGHT)

        self.setStyleSheet("""QPushButton {background-color:rgb(220, 38, 38);color:black;border-radius: 15px;}QPushButton:hover {background-color:rgba(0, 0, 0, 125);}""")

        self.show()

class ClosedButton(Button):

    """Circle button which shot down UI"""

    def __init__(self, parent, function, TEXT="close", RADIUS=40):
        super().__init__(TEXT, function, parent)
        self.setGeometry(0.9 * MainWindow.WIDTH, 0.05 * MainWindow.HEIGHT, RADIUS, RADIUS)

        self.setStyleSheet("""QPushButton {background-color:red;border-radius:20px;}QPushButton:hover {background-color:rgba(0, 0, 0, 125);}""")

        self.show()

class SendButton(Button):

    """Clicking on this button changing page to classification page where is result of classification of image, which has user given url"""

    def __init__(self, parent, function, TEXT="Send",WIDTH = 100,HEIGHT = 100):
        super().__init__(TEXT, function, parent)

        self.setGeometry(MainWindow.WIDTH * 0.87, MainWindow.HEIGHT * 0.25,WIDTH,HEIGHT)

        self.setStyleSheet("""QPushButton {background-color:green;border-radius:20px;}QPushButton:hover {background-color:rgba(0, 0, 0, 125);}""")

        self.show()

class Title(QLabel):

    """Name of projetct"""

    def __init__(self, parent, TEXT="Museum object classification"):
        super().__init__(TEXT, parent)

        self.move(MainWindow.WIDTH * 0.3, MainWindow.HEIGHT // 15)

        self.setStyleSheet("color: white; font-size: 30px;")

        self.show()

class UrlBox(QLineEdit):

    """User can write here image url, if is wrong text will change for massage error"""

    def __init__(self, parent):
        super().__init__(parent)

        self.img = None

        self.setGeometry(MainWindow.WIDTH * 0.15, MainWindow.HEIGHT * 0.3, 700, 50)

        self.setStyleSheet("""QLineEdit {background-color: white;color: black;border: 2px solid gray;border-radius: 10px;padding-left: 10px;font-size: 16px;}""")

        self.setPlaceholderText("Enter image path")

        self.show()
    
    # if given url is correct, then image is loaded in program 
    def load_img(self):

        img_path = self.text().replace('"', '').replace("\\", "/")

        self.img = Image.open(img_path)
    
    # If given url is not correct, then will be displayed message error
    def write_wrong_url_message(self,TEXT = "URL OF IMAGE IS NOT CORRECT!"):

        self.clear()

        self.setStyleSheet("""QLineEdit {background-color: white;color:black;border: 2px solid gray;border-radius: 10px;padding-left: 10px;font-size: 16px;}""")

        self.setPlaceholderText(TEXT)

        self.show()

class Result(QLabel):
    def __init__(self,maping,img,model,parent,PART1_TEXT="Class:",PART2_TEXT = "           ",TWO_DIGITS_CLASS_NAME = 2):

        super().__init__(parent)

        top_classes_probabilities = Model.classify_img(img, model)


        text = ""
        for label,probability in top_classes_probabilities:
            spaces = PART2_TEXT
            class_name = maping[str(label)]

            if len(str(class_name)) == TWO_DIGITS_CLASS_NAME:
                spaces = spaces[:-2]

            report_class_result = f"{PART1_TEXT} {class_name}{spaces}{probability}%" + "\n"
            text += report_class_result
        
        text = text[:-1]

        self.setText(text)
        self.setStyleSheet("color:white; font-size: 30px;")
        self.move(MainWindow.WIDTH // 2.7, MainWindow.HEIGHT // 5.5)
        self.show()

#######################################################################################################################

# WINDOW

class MainWindow(QMainWindow):

    """Main window which user see"""

    WIDTH = 1000
    HEIGHT = 600

    def __init__(self,HELP_FILES = "help_files"):
        super().__init__()
        model_path  = os.path.join("..",HELP_FILES,"museum_model.pth")
        maping_path = os.path.join("..",HELP_FILES,Maping.MAPING_NAME)
        device = None

        if torch.cuda.is_available():
            device = "cuda"
        
        else:
            device = "cpu"

        self.model = torch.load(model_path,map_location=torch.device(device),weights_only=False)
        self.maping = None
        
        self.model.to(device)

        with open(maping_path,"r",encoding="utf-8") as maping_file:
            self.maping = json.load(maping_file)

        self.resize(self.WIDTH, self.HEIGHT)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.round_widget = QWidget(self)
        self.round_widget.resize(self.WIDTH, self.HEIGHT)
        self.round_widget.setStyleSheet("""background-color: rgb(45, 45, 48);border-radius: 50px;""")

        self.go_start_page()

        self.show()
    
    # mousePressEvent and mouseMoveEvent is for moving window, when user wants to
    def mousePressEvent(self, event):
        self.oldPosition = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QPoint(event.globalPos() - self.oldPosition)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPosition = event.globalPos()
    
    # delete items as buttons,boxes... in system for creating new page
    def delete_items(self):
        for item in self.round_widget.findChildren(QWidget):
            item.deleteLater()

    # move to url page, where user can write url of image, which wants to be classified.
    def go_url_page(self):

        self.delete_items()

        self.title = Title(self.round_widget)
        self.back_button = BackButton(self.round_widget, self.go_start_page)
        self.url_box = UrlBox(self.round_widget)
        self.closed_button = ClosedButton(self.round_widget, self.close_window)
        self.send_button = SendButton(self.round_widget,self.choose_img)

    # shot down window
    def close_window(self):
        self.close()
    
    # move to starting page
    def go_start_page(self):

        self.delete_items()

        self.title = Title(self.round_widget)
        self.start_button = StartButton(self.round_widget, self.go_url_page)
        self.end_button = EndButton(self.round_widget, self.close_window)
        self.closed_button = ClosedButton(self.round_widget, self.close_window)
    
    
    # result page, where user can see result of classified image
    def go_classification_page(self):
        
        self.delete_items()
        
        self.title = Title(self.round_widget)
        self.back_button = BackButton(self.round_widget, self.go_url_page)
        self.closed_button = ClosedButton(self.round_widget, self.close_window)
        self.result = Result(self.maping,self.url_box.img,self.model,self.round_widget)
    
    # method for evaluating correctness of users url
    def choose_img(self):

        try:
            self.url_box.load_img()

            self.go_classification_page()

        except Exception:
            self.url_box.write_wrong_url_message()


#######################################################################################################################

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())