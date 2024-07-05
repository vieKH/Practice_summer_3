import cv2
import sys
from enum import Enum
from PyQt6.QtWidgets import QPushButton, QFileDialog, QApplication, QMainWindow, QLabel, QComboBox
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap, QImage
from embed_extract import embed_watermark, extract_watermark, direct_replacement, bitwise_addition,\
    negated_bitwise_addition


class Option(Enum):
    ORIGINAL = 1
    WATERMARK = 2
    EXTRACT = 3


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Embedding and digital watermark extraction")
        self.setStyleSheet("background-color : #FFDEAD")
        self.setMinimumSize(1000, 700)

        self.path_original, self.path_watermark, self.path_extract, self.method = '', '', '', direct_replacement

        button_crt_upfile_ori = self.add_button("Upload original image", 150, 50, 50, 100)
        button_crt_upfile_ori.clicked.connect(lambda: self.upload_file("Original image", Option.ORIGINAL))

        button_crt_upfile_wat = self.add_button("Upload watermark image", 150, 50, 200, 100)
        button_crt_upfile_wat.clicked.connect(lambda: self.upload_file("Digital watermark image", Option.WATERMARK))

        button_crt_upfile_ext = self.add_button("Upload extract image", 150, 50, 50, 175)
        button_crt_upfile_ext.clicked.connect(lambda: self.upload_file("Image need to be extracted", Option.EXTRACT))

        self.method_box = QComboBox(self)
        self.method_box.addItem("Direct replacement")
        self.method_box.addItem("Bitwise addition")
        self.method_box.addItem("Negated bitwise addition")
        self.method_box.setStyleSheet("color: #800000; background-color: #FFDEAD; border: 2px solid #000000;")
        self.method_box.setFixedSize(150, 50)
        self.method_box.move(200, 175)
        self.method_box.currentTextChanged.connect(self.update_method)

        button_crt_embed = self.add_button("Embed 2 image", 300, 50, 50, 250)
        button_crt_embed.clicked.connect(self.interface_embed)

        button_crt_ext = self.add_button("Digital watermark extraction", 300, 50, 50, 325)
        button_crt_ext.clicked.connect(self.interface_extract)

        button_crt_save_embedded = self.add_button("Save image after embedding", 300, 50, 50, 400)
        button_crt_ext.clicked.connect(self.save_file)

        button_crt_save_extracted = self.add_button("Save binary image after extracting", 300, 50, 50, 475)
        button_crt_save_extracted.clicked.connect(self.save_file)

        button_crt_his = self.add_button("Histogram", 300, 50, 50, 550)

        self.image = QLabel("The results will appear here", self)
        self.image.setStyleSheet("color: #800000")
        self.image.resize(550, 500)
        self.image.move(400, 100)
        self.show()

    def add_button(self, name: str, size_x: int, size_y: int, pos_x: int, pos_y: int):
        """
        Add button with a fixed size and position
        :param name: button name
        :param size_x: width
        :param size_y: height
        :param pos_x: position x
        :param pos_y: position y
        :return: button type QPushButton
        """
        button = QPushButton(name, self)
        button.setFixedSize(QSize(size_x, size_y))
        button.move(pos_x, pos_y)
        return button

    def update_method(self, text):
        if text == "Direct replacement":
            self.method = direct_replacement
        elif text == "Bitwise addition":
            self.method = bitwise_addition
        else:
            self.method = negated_bitwise_addition

    def upload_file(self, title: str, option: Option):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, title, "", "Images (*.png *.xpm *.jpg *.bmp *.gif)")
            if file_path:
                match option:
                    case Option.ORIGINAL:
                        self.path_original = file_path
                    case Option.WATERMARK:
                        self.path_watermark = file_path
                    case Option.EXTRACT:
                        self.path_extract = file_path
        except Exception as e:
            print(f"Error selecting file: {e}")

    def print_image(self, image_array):
        try:
            height, width, channel = image_array.shape
            bytes_per_line = 3 * width
            image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(self.image.height(),
                                                     self.image.width(),
                                                     aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
            self.image.setPixmap(pixmap)
            self.resize(pixmap.size())
            self.adjustSize()
        except OSError as err:
            print(f"Something was wrong: {err}")

    def interface_embed(self):
        container = cv2.imread(self.path_original, cv2.IMREAD_UNCHANGED)
        watermark = cv2.imread(self.path_watermark, cv2.IMREAD_GRAYSCALE)
        embed = embed_watermark(container, watermark, method=self.method)
        self.print_image(embed)

    def interface_extract(self):
        watermarked = cv2.imread(self.path_extract, cv2.IMREAD_UNCHANGED)
        container = cv2.imread(self.path_original, cv2.IMREAD_UNCHANGED)
        watermark = cv2.imread(self.path_watermark, cv2.IMREAD_GRAYSCALE)
        watermark_size = (watermark.shape[1], watermark.shape[0])
        extracted = extract_watermark(watermarked, container, watermark_size, method=self.method)
        self.print_image(extracted)

    def save_file(self):
        try:
            pixmap = QPixmap(self.path_original).scaled(self.image.height(),
                                                        self.image.width(),
                                                        aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
            self.image.setPixmap(pixmap)
            self.resize(pixmap.size())
            self.adjustSize()
        except OSError as err:
            print(f"Error in function embed: {err}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()

    app.exec()
