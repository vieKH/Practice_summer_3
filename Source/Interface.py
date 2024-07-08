import cv2
import numpy as np
import sys

from enum import Enum
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QPushButton, QFileDialog, QApplication, QMainWindow, QLabel, QComboBox, QMessageBox
from embed_extract import embed_watermark, extract_watermark, Method
from histogram import plot_histograms


class Option(Enum):
    ORIGINAL = 1
    WATERMARK = 2
    EXTRACT = 3
    EMBED = 4
    EXTRACTED = 5


def show_error(message: str) -> None:
    """
    Show message error in new box
    :param message: message error
    :return: None
    """
    error_dialog = QMessageBox()
    error_dialog.setIcon(QMessageBox.Icon.Critical)
    error_dialog.setWindowTitle("Error")
    error_dialog.setText("An error has occurred.")
    error_dialog.setInformativeText(message)
    error_dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
    error_dialog.exec()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Embed and digital watermark extraction")
        self.setStyleSheet("background-color : #FFDEAD")
        self.setMinimumSize(1000, 700)

        self.path_original, self.path_watermark, self.path_extract = '', '', ''
        self.embedded, self.extracted, self.method = '', '', Method.DIRECT

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
        button_crt_save_embedded.clicked.connect(lambda: self.save_file(Option.EMBED))

        button_crt_save_extracted = self.add_button("Save binary image after extracting", 300, 50, 50, 475)
        button_crt_save_extracted.clicked.connect(lambda: self.save_file(Option.EXTRACTED))

        button_crt_his = self.add_button("Histogram", 300, 50, 50, 550)
        button_crt_his.clicked.connect(self.interface_histogram)

        self.image = QLabel("The results will appear here", self)
        self.image.setStyleSheet("color: #800000")
        self.image.resize(550, 500)
        self.image.move(400, 100)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_oringal_image = QLabel(f'Path to original image: {self.path_original}', self)
        self.label_oringal_image.resize(500, 25)
        self.label_oringal_image.move(50, 0)
        self.label_oringal_image.setStyleSheet("color: red;")

        self.label_watermark_image = QLabel(f'Path to watermark image: {self.path_watermark}', self)
        self.label_watermark_image.resize(500, 25)
        self.label_watermark_image.move(50, 25)
        self.label_watermark_image.setStyleSheet("color: red;")

        self.label_image_extracted = QLabel(f'Path to image need to extracted: {self.path_extract}', self)
        self.label_image_extracted.resize(500, 25)
        self.label_image_extracted.move(50, 50)
        self.label_image_extracted.setStyleSheet("color: red;")

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

    def update_method(self, text: str) -> None:
        """
        Get data from user, set parameter self.method
        :param text: Data from user
        :return: None
        """
        if text == "Direct replacement":
            self.method = Method.DIRECT
        elif text == "Bitwise addition":
            self.method = Method.BITWISE_ADD
        else:
            self.method = Method.NEGATED_BITWISE_ADD

    def upload_file(self, title: str, option: Option) -> None:
        """
        Choose file from computer, this function use for 3 options
        :param title: title for every option
        :param option: option
        :return: None
        """
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, title, "", "Images (*.png *.xpm *.jpg *.bmp *.gif)")
            if file_path:

                match option:
                    case Option.ORIGINAL:
                        self.path_original = file_path
                        self.label_oringal_image.setText(f'Path to original image: {self.path_original}')
                    case Option.WATERMARK:
                        self.path_watermark = file_path
                        self.label_watermark_image.setText(f'Path to watermark image: {self.path_watermark}')
                    case Option.EXTRACT:
                        self.path_extract = file_path
                        self.label_image_extracted.setText(f'Path to image need to extracted: {self.path_extract}')
        except Exception as e:
            show_error(f"Error selecting file: {e} !!!")

    def print_image(self, image_array: np.ndarray):
        """
        Print image at label self.image
        :param image_array: data's image
        :return: None
        """
        try:
            image_array_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            height, width, channel = image_array_rgb.shape
            bytes_per_line = 3 * width
            image = QImage(image_array_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(self.image.height(),
                                                     self.image.width(),
                                                     aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
            self.image.setPixmap(pixmap)
            self.resize(pixmap.size())
            self.adjustSize()
        except Exception as err:
            show_error(f"Something was wrong when we try print image: {err} !!!")

    def interface_embed(self) -> None:
        """
        Embed 2 image by button
        :return: None
        """
        try:
            container = cv2.imread(self.path_original, cv2.IMREAD_UNCHANGED)
            watermark = cv2.imread(self.path_watermark, cv2.IMREAD_GRAYSCALE)
            _, watermark = cv2.threshold(watermark, 127, 255, cv2.THRESH_BINARY)
            self.embedded = embed_watermark(container, watermark, method=self.method)
            self.print_image(self.embedded)
        except Exception as err:
            show_error(f"Error in embedding {err}")

    def interface_extract(self) -> None:
        """
        Extract watermark image from image
        :return: None
        """
        try:
            watermarked = cv2.imread(self.path_extract, cv2.IMREAD_UNCHANGED)
            container = cv2.imread(self.path_original, cv2.IMREAD_UNCHANGED)
            watermark = cv2.imread(self.path_watermark, cv2.IMREAD_GRAYSCALE)
            _, watermark = cv2.threshold(watermark, 127, 255, cv2.THRESH_BINARY)
            watermark_size = (watermark.shape[1], watermark.shape[0])
            self.extracted = extract_watermark(watermarked, container, watermark_size, method=self.method)
            self.print_image(self.extracted)
        except Exception as err:
            show_error(f"Error in extraction {err}")

    def save_file(self, option: Option) -> None:
        """
        Save file in computer
        :param option: option
        :return: None
        """
        try:
            save_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Images (*.png *.jpg *.bmp)")
            if save_path:
                if option == Option.EMBED:
                    cv2.imwrite(save_path, self.embedded)
                if option == Option.EXTRACTED:
                    cv2.imwrite(save_path, self.extracted)
        except Exception as err:
            show_error(f"Something was wrong when we try to save file: {err} !!!")

    def interface_histogram(self) -> None:
        try:
            original_img = cv2.imread(self.path_original, cv2.IMREAD_UNCHANGED)
            plot_histograms(original_img, self.embedded)
        except Exception as err:
            show_error(f"Something was wrong when we try to draw histogram: {err} !!!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec()
