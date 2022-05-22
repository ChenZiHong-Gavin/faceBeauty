# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'face.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from smoothAndWhitening import smooth, whitening
from redLips import redLip
from ARfilters import apply_filter

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(480, 640)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.cameraShow = QtWidgets.QLabel(self.centralwidget)
        self.cameraShow.setGeometry(QtCore.QRect(0, 0, 480, 640))
        self.cameraShow.setObjectName("cameraShow")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 490, 481, 151))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_4 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_4.setMinimumSize(QtCore.QSize(60, 60))
        self.pushButton_4.setMaximumSize(QtCore.QSize(60, 60))
        self.pushButton_4.setStyleSheet("background-color: rgb(170, 170, 255);\n"
"color: rgb(255,255,255);  \n"
"border-radius: 30px;  border: 2px groove gray;\n"
"font: 9pt \"AcadEref\";\n"
"border-style: outset;\n"
"border-image: url('ARfilters/filters/dog-ears.png');")
        self.pushButton_4.setText("")
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout.addWidget(self.pushButton_4)
        self.pushButton_3 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_3.setMinimumSize(QtCore.QSize(60, 60))
        self.pushButton_3.setMaximumSize(QtCore.QSize(60, 60))
        self.pushButton_3.setStyleSheet("background-color: rgb(170, 170, 255);\n"
"color: rgb(255,255,255);  \n"
"border-radius: 30px;  border: 2px groove gray;\n"
"font: 9pt \"AcadEref\";\n"
"border-style: outset;\n"
"border-image: url('ARfilters/filters/green-carnival.png');")
        self.pushButton_3.setText("")
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout.addWidget(self.pushButton_3)
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton.setMinimumSize(QtCore.QSize(60, 60))
        self.pushButton.setMaximumSize(QtCore.QSize(60, 60))
        self.pushButton.setStyleSheet("background-color: rgb(170, 170, 255);\n"
"color: rgb(255,255,255);  \n"
"border-radius: 30px;  border: 2px groove gray;\n"
"font: 9pt \"AcadEref\";\n"
"border-style: outset;\n"
"border-image: url('ARfilters/filters/flower-crown.png');")
        self.pushButton.setText("")
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_2.setMinimumSize(QtCore.QSize(60, 60))
        self.pushButton_2.setMaximumSize(QtCore.QSize(60, 60))
        self.pushButton_2.setStyleSheet("background-color: rgb(170, 170, 255);\n"
"color: rgb(255,255,255);  \n"
"border-radius: 30px;  border: 2px groove gray;\n"
"font: 9pt \"AcadEref\";\n"
"border-style: outset;\n"
"border-image: url('ARfilters/filters/cat-ears.png');")
        self.pushButton_2.setText("")
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.pushButton_5 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_5.setMinimumSize(QtCore.QSize(60, 60))
        self.pushButton_5.setMaximumSize(QtCore.QSize(60, 60))
        self.pushButton_5.setStyleSheet("background-color: rgb(170, 170, 255);\n"
"color: rgb(255,255,255);  \n"
"border-radius: 30px;  border: 2px groove gray;\n"
"font: 9pt \"AcadEref\";\n"
"border-image: url('ARfilters/filters/anime.png');")
        self.pushButton_5.setText("")
        self.pushButton_5.setObjectName("pushButton_5")
        self.horizontalLayout.addWidget(self.pushButton_5)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(0, 0, 191, 31))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.checkBox = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout_4.addWidget(self.checkBox)
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(0, 30, 191, 31))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.checkBox_2 = QtWidgets.QCheckBox(self.horizontalLayoutWidget_3)
        self.checkBox_2.setObjectName("checkBox_2")
        self.horizontalLayout_5.addWidget(self.checkBox_2)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 60, 191, 131))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_7.addWidget(self.label_4)
        self.horizontalSlider_3 = QtWidgets.QSlider(self.verticalLayoutWidget)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.horizontalLayout_7.addWidget(self.horizontalSlider_3)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.horizontalSlider_2 = QtWidgets.QSlider(self.verticalLayoutWidget)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.horizontalLayout_3.addWidget(self.horizontalSlider_2)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.horizontalSlider = QtWidgets.QSlider(self.verticalLayoutWidget)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout_2.addWidget(self.horizontalSlider)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # 自定义变量
        # 定时器
        self.timer = QtCore.QTimer()
        # 编写定时器的启动函数并与button建立信号与槽
        self.timer.timeout.connect(self.show_viedo)
        # 摄像头
        self.cap_video = 0
        # 记录定时器工作状态
        self.flag = 0
        # 存放每一帧读取的图像
        self.img = []
        self.cap_video = cv2.VideoCapture(0)
        self.timer.start(0)
        # 存放效果状态
        self.effect = {"whiten": False, "smooth": False, "lips": [0, 0, 0], "effect": False}

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.cameraShow.setText(_translate("MainWindow", "IMage"))
        self.checkBox.setText(_translate("MainWindow", "美白"))
        self.checkBox_2.setText(_translate("MainWindow", "磨皮"))
        self.label.setText(_translate("MainWindow", "嘴唇"))
        self.label_4.setText(_translate("MainWindow", "B"))
        self.label_3.setText(_translate("MainWindow", "G"))
        self.label_2.setText(_translate("MainWindow", "R"))


    # label上实时显示图像函数
    def show_cv_img(self, img):
        shrink = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        QtImg = QtGui.QImage(shrink.data,
                             shrink.shape[1],
                             shrink.shape[0],
                             shrink.shape[1] * 3,
                             QtGui.QImage.Format_RGB888)
        jpg_out = QtGui.QPixmap(QtImg).scaled(
            self.cameraShow.width(), self.cameraShow.height())

        self.cameraShow.setPixmap(jpg_out)
    def show_viedo(self):
        ret, self.img = self.cap_video.read()
        effectList = self.effect
        effectList["whiten"] = self.checkBox.isChecked()
        effectList["smooth"] = self.checkBox_2.isChecked()
        r = int((self.horizontalSlider.value()+1)/100*255)
        g = int((self.horizontalSlider_2.value()+1)/100*255)
        b = int((self.horizontalSlider_3.value()+1)/100*255)
        if self.pushButton.isDown():
            effectList["effect"] = "flower-crown"
        if self.pushButton_2.isDown():
            effectList["effect"] = "cat"
        if self.pushButton_3.isDown():
            effectList["effect"] = "green-carnival"
        if self.pushButton_4.isDown():
            effectList["effect"] = "dog"
        if self.pushButton_5.isDown():
            effectList["effect"] = "anime"
        whiten = effectList["whiten"]
        isSmooth = effectList["smooth"]

        if ret:
            width, length, _ = self.img.shape
            # 查询effectList进行修改
            if whiten==True:
                self.img = whitening.add_light(self.img)
            if isSmooth==True:
                self.img = smooth.face_smooth_picture(self.img)
            if r!=0 or b!=0 or g!=0:
                self.img = redLip.get_lip_picture(self.img, [b,g,r])
            if effectList["effect"]!=False:
                self.img = apply_filter.get_AR_Picture(self.img, effectList["effect"])
            self.show_cv_img(self.img)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
