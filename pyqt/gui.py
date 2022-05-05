# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(2002, 883)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.input_path = QtWidgets.QLineEdit(self.centralwidget)
        self.input_path.setGeometry(QtCore.QRect(120, 10, 231, 21))
        self.input_path.setObjectName("input_path")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 10, 91, 21))
        self.label.setObjectName("label")
        self.btn_begin = QtWidgets.QPushButton(self.centralwidget)
        self.btn_begin.setGeometry(QtCore.QRect(370, 10, 93, 28))
        self.btn_begin.setObjectName("btn_begin")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 40, 91, 21))
        self.label_2.setObjectName("label_2")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(120, 40, 231, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 70, 91, 21))
        self.label_3.setObjectName("label_3")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setEnabled(True)
        self.checkBox.setGeometry(QtCore.QRect(130, 70, 91, 19))
        self.checkBox.setText("")
        self.checkBox.setCheckable(True)
        self.checkBox.setChecked(False)
        self.checkBox.setObjectName("checkBox")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 100, 1041, 411))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.graphic_res = QtWidgets.QGraphicsView(self.layoutWidget)
        self.graphic_res.setObjectName("graphic_res")
        self.horizontalLayout.addWidget(self.graphic_res)
        self.graphics_derain = QtWidgets.QGraphicsView(self.layoutWidget)
        self.graphics_derain.setObjectName("graphics_derain")
        self.horizontalLayout.addWidget(self.graphics_derain)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(540, 10, 295, 41))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.depth_btn = QtWidgets.QPushButton(self.layoutWidget1)
        self.depth_btn.setObjectName("depth_btn")
        self.horizontalLayout_2.addWidget(self.depth_btn)
        self.pB_next = QtWidgets.QPushButton(self.centralwidget)
        self.pB_next.setGeometry(QtCore.QRect(870, 530, 40, 40))
        self.pB_next.setObjectName("pB_next")
        self.pB_previous = QtWidgets.QPushButton(self.centralwidget)
        self.pB_previous.setGeometry(QtCore.QRect(800, 530, 40, 40))
        self.pB_previous.setObjectName("pB_previous")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(30, 540, 72, 15))
        self.label_5.setObjectName("label_5")
        self.label_class = QtWidgets.QLabel(self.centralwidget)
        self.label_class.setGeometry(QtCore.QRect(80, 530, 101, 31))
        self.label_class.setText("")
        self.label_class.setObjectName("label_class")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(20, 580, 1031, 201))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1029, 199))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 2002, 26))
        self.menubar.setObjectName("menubar")
        self.menuss = QtWidgets.QMenu(self.menubar)
        self.menuss.setObjectName("menuss")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.dockWidget_3 = QtWidgets.QDockWidget(MainWindow)
        self.dockWidget_3.setObjectName("dockWidget_3")
        self.dockWidgetContents_5 = QtWidgets.QWidget()
        self.dockWidgetContents_5.setObjectName("dockWidgetContents_5")
        self.gridLayout_20 = QtWidgets.QGridLayout(self.dockWidgetContents_5)
        self.gridLayout_20.setObjectName("gridLayout_20")
        self.listWidget_image = QtWidgets.QListWidget(self.dockWidgetContents_5)
        self.listWidget_image.setEnabled(True)
        self.listWidget_image.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.listWidget_image.setAutoScroll(True)
        self.listWidget_image.setProperty("showDropIndicator", True)
        self.listWidget_image.setObjectName("listWidget_image")
        self.gridLayout_20.addWidget(self.listWidget_image, 0, 0, 1, 1)
        self.dockWidget_7 = QtWidgets.QDockWidget(self.dockWidgetContents_5)
        self.dockWidget_7.setObjectName("dockWidget_7")
        self.dockWidgetContents_7 = QtWidgets.QWidget()
        self.dockWidgetContents_7.setObjectName("dockWidgetContents_7")
        self.gridLayout_22 = QtWidgets.QGridLayout(self.dockWidgetContents_7)
        self.gridLayout_22.setObjectName("gridLayout_22")
        self.tab_class_num = QtWidgets.QTableWidget(self.dockWidgetContents_7)
        self.tab_class_num.setAlternatingRowColors(True)
        self.tab_class_num.setObjectName("tab_class_num")
        self.tab_class_num.setColumnCount(2)
        self.tab_class_num.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tab_class_num.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tab_class_num.setHorizontalHeaderItem(1, item)
        self.tab_class_num.horizontalHeader().setStretchLastSection(True)
        self.gridLayout_22.addWidget(self.tab_class_num, 0, 0, 1, 1)
        self.dockWidget_7.setWidget(self.dockWidgetContents_7)
        self.gridLayout_20.addWidget(self.dockWidget_7, 1, 0, 1, 1)
        self.dockWidget_3.setWidget(self.dockWidgetContents_5)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget_3)
        self.dockWidget_6 = QtWidgets.QDockWidget(MainWindow)
        self.dockWidget_6.setObjectName("dockWidget_6")
        self.dockWidgetContents_6 = QtWidgets.QWidget()
        self.dockWidgetContents_6.setObjectName("dockWidgetContents_6")
        self.gridLayout_21 = QtWidgets.QGridLayout(self.dockWidgetContents_6)
        self.gridLayout_21.setObjectName("gridLayout_21")
        self.graphics_input = QtWidgets.QGraphicsView(self.dockWidgetContents_6)
        self.graphics_input.setObjectName("graphics_input")
        self.gridLayout_21.addWidget(self.graphics_input, 0, 0, 1, 1)
        self.dockWidget_8 = QtWidgets.QDockWidget(self.dockWidgetContents_6)
        self.dockWidget_8.setObjectName("dockWidget_8")
        self.dockWidgetContents_8 = QtWidgets.QWidget()
        self.dockWidgetContents_8.setObjectName("dockWidgetContents_8")
        self.gridLayout_23 = QtWidgets.QGridLayout(self.dockWidgetContents_8)
        self.gridLayout_23.setObjectName("gridLayout_23")
        self.graphics_depth = QtWidgets.QGraphicsView(self.dockWidgetContents_8)
        self.graphics_depth.setObjectName("graphics_depth")
        self.gridLayout_23.addWidget(self.graphics_depth, 0, 0, 1, 1)
        self.dockWidget_8.setWidget(self.dockWidgetContents_8)
        self.gridLayout_21.addWidget(self.dockWidget_8, 1, 0, 1, 1)
        self.dockWidget_6.setWidget(self.dockWidgetContents_6)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dockWidget_6)
        self.action_rain_kitti = QtWidgets.QAction(MainWindow)
        self.action_rain_kitti.setObjectName("action_rain_kitti")
        self.action_rain_cityscape = QtWidgets.QAction(MainWindow)
        self.action_rain_cityscape.setObjectName("action_rain_cityscape")
        self.actionreal_world = QtWidgets.QAction(MainWindow)
        self.actionreal_world.setObjectName("actionreal_world")
        self.action_model_kitti = QtWidgets.QAction(MainWindow)
        self.action_model_kitti.setObjectName("action_model_kitti")
        self.action_model_cityscapes = QtWidgets.QAction(MainWindow)
        self.action_model_cityscapes.setObjectName("action_model_cityscapes")
        self.action_real_world = QtWidgets.QAction(MainWindow)
        self.action_real_world.setObjectName("action_real_world")
        self.menuss.addAction(self.action_rain_kitti)
        self.menuss.addSeparator()
        self.menuss.addAction(self.action_rain_cityscape)
        self.menuss.addSeparator()
        self.menuss.addAction(self.action_real_world)
        self.menu.addAction(self.action_model_kitti)
        self.menu.addSeparator()
        self.menu.addAction(self.action_model_cityscapes)
        self.menu.addSeparator()
        self.menubar.addAction(self.menuss.menuAction())
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "图片路径:"))
        self.btn_begin.setText(_translate("MainWindow", "启动"))
        self.label_2.setText(_translate("MainWindow", "启动模式:"))
        self.comboBox.setItemText(0, _translate("MainWindow", "DGT_DerainNet_Faster_RCNN"))
        self.comboBox.setItemText(1, _translate("MainWindow", "DerainNet_Faster_RCNN"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Faster_RCNN"))
        self.label_3.setText(_translate("MainWindow", "图像修复输出:"))
        self.label_4.setText(_translate("MainWindow", "monodepth2算法预测深度图:"))
        self.depth_btn.setText(_translate("MainWindow", "深度估计"))
        self.pB_next.setText(_translate("MainWindow", "↓"))
        self.pB_previous.setText(_translate("MainWindow", "↑"))
        self.label_5.setText(_translate("MainWindow", "类别："))
        self.menuss.setTitle(_translate("MainWindow", "测试集"))
        self.menu.setTitle(_translate("MainWindow", "训练模型"))
        self.menu_2.setTitle(_translate("MainWindow", "选项"))
        self.dockWidget_3.setWindowTitle(_translate("MainWindow", "图片"))
        self.dockWidget_7.setWindowTitle(_translate("MainWindow", "检测信息"))
        item = self.tab_class_num.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "类别"))
        item = self.tab_class_num.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "数量"))
        self.dockWidget_6.setWindowTitle(_translate("MainWindow", "原始图片"))
        self.dockWidget_8.setWindowTitle(_translate("MainWindow", "先验深度图"))
        self.action_rain_kitti.setText(_translate("MainWindow", "rain_kitti"))
        self.action_rain_cityscape.setText(_translate("MainWindow", "rain_cityscapes"))
        self.actionreal_world.setText(_translate("MainWindow", "real_world"))
        self.action_model_kitti.setText(_translate("MainWindow", "kitti"))
        self.action_model_cityscapes.setText(_translate("MainWindow", "cityscape"))
        self.action_real_world.setText(_translate("MainWindow", "real_world"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())