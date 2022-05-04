import json
import numpy as np
import cv2
import torch
from torchvision import transforms
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QSize, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
import matplotlib as mpl
import matplotlib.cm as cm
import monodepth2
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_box
from monodepth2.layers import disp_to_depth
from nets import DDGN_Depth_CFT, DDGN_Basic
from network_files import FasterRCNN
from network_files.faster_rcnn_framework import Derain_FasterRCNN
from predict import time_synchronized
from pyqt.gui import Ui_MainWindow
from pyqt.basic_derain_image import Ui_Form as basic_derain_Ui_Form
from pyqt.basic_image import Ui_Form as basic_Ui_Form
from qt_material import apply_stylesheet
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, qApp, QGraphicsScene, QGraphicsPixmapItem, \
    QTableWidgetItem, QWidget, QHBoxLayout, QLabel, QMessageBox
import sys
import os
import pyqt.gui as gui
from PIL import Image
from monodepth2.utils import confidence, BilateralGrid, grid_params, BilateralSolver, bs_params

KITTI_PATH = "../../val/kitti/rain"
CITYSCAPES_PATH = "../../val/cityscapes/rain"
REAL_PATH = "../../val/real/rain"
SAVE_PATH = "../../val/output/"
MODEL_FLAG = "KITTI"
MODE_FLAG = 0
IMAGE_PATH = ""
DEPTH_PATH = ""
scale = 0.5
COMBINE_FLAG = False

class ImageWidget(QWidget):
    signal_page = pyqtSignal(str)  # 页数信号
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.index = 0
        self.hbox = QHBoxLayout(self)
        self.hbox.setContentsMargins(0, 0, 0, 0)
        # self.show_images_list(class_name=self.classes[self.index])  # 初次加载形状图像列表

    def set_img_path_class(self,path,classes,info_data):
        self.path = path
        self.classes = classes
        self.info_data = info_data

    def img2pixmap(self, image):
        height, width, depth = image.shape
        cvimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cvimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)
        return QtGui.QPixmap(cvimg)

    def show_images_list(self,class_name,path=None):  # 加载图像列表
        if path:
            self.cur_path = path
        for i in range(self.hbox.count()):  # 每次加载先清空内容，避免layout里堆积label
            self.hbox.itemAt(i).widget().deleteLater()
        for [left, top, right, bottom] in self.info_data[class_name]:
            img = cv2.imread(self.cur_path)
            cur_img = img[int(top):int(bottom), int(left):int(right)]
            pix = self.img2pixmap(cur_img)
            pix = pix.scaled(152, 76, Qt.KeepAspectRatio,Qt.SmoothTransformation)
            label = QLabel()
            label.setPixmap(pix)  # 加载图片
            self.hbox.addWidget(label)   # 在水平布局中添加自定义label

    def turn_page(self, num):  # 图像列表翻页
        self.index = self.index + num
        if self.index<=-1:
            self.index = 0
        elif self.index>=len(self.classes):
            self.index = len(self.classes)-1
        self.show_images_list(self.classes[self.index])  # 重新加载图像列表
        self.signal_page.emit(self.classes[self.index])


class MainCode(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        gui.Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.init_module()
        self.btn_begin.clicked.connect(self.process)
        self.depth_btn.clicked.connect(self.depth_predit)
        self.action_rain_kitti.triggered.connect(self.update_listWidget_image_1)
        self.action_rain_cityscape.triggered.connect(self.update_listWidget_image_2)
        self.action_real_world.triggered.connect(self.update_listWidget_image_3)
        self.listWidget_image.currentItemChanged.connect(self.update_image)
        self.action_model_kitti.triggered.connect(self.update_model_flag_1)
        self.action_model_cityscapes.triggered.connect(self.update_model_flag_2)
        self.checkBox.stateChanged.connect(self.change_check_flag)
        self.image_widget = ImageWidget(self)
        self.image_widget.move(30, 550)
        self.scrollArea.setWidget(self.image_widget)
        self.pB_previous.clicked.connect(lambda: self.image_widget.turn_page(-1))
        self.pB_next.clicked.connect(lambda: self.image_widget.turn_page(1))
        self.image_widget.signal_page.connect(self.change_page)
        self.comboBox.currentIndexChanged.connect(self.change_combo_index)
        self.thread1 = Mythread()
        self.thread1.signal0.connect(self.show_res)
        self.thread2 = Depth_thread()
        self.thread2.signal0.connect(self.show_depth)

    def change_check_flag(self):
        global COMBINE_FLAG
        COMBINE_FLAG = self.checkBox.checkState()

    def change_page(self, class_name):
        self.label_class.setText(class_name)

    def init_module(self):
        self.listWidget_image.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

    def change_combo_index(self):
        global MODE_FLAG
        MODE_FLAG = self.comboBox.currentIndex()

    def update_model_flag_1(self):
        global MODEL_FLAG
        MODEL_FLAG = "KITTI"
        derain_basic_weights = "../models/kitti_derain_basic.pth"
        derain_weights = "../models/kitti_derain_depth_cft.pth"
        train_weights = "../models/kitti_20211222_39.pth"
        train_fine_tune_weights = "../models/kitti_depth_transformer_fixup_48.pth"
        self.thread1.basic_model = FasterRCNN(backbone=self.thread1.backbone, num_classes=7 + 1, rpn_score_thresh=0.5)
        self.thread1.basic_model.load_state_dict(torch.load(train_weights, map_location=self.thread1.device)["model"])
        self.thread1.fine_tune_model = FasterRCNN(backbone=self.thread1.backbone, num_classes=7 + 1, rpn_score_thresh=0.5)
        self.thread1.fine_tune_model.load_state_dict(torch.load(train_fine_tune_weights, map_location=self.thread1.device)["model"])
        self.thread1.de_rain_basic_model.load_state_dict(torch.load(derain_basic_weights))
        self.thread1.de_rain_depth_trans_model.load_state_dict(torch.load(derain_weights))
        self.thread1.de_rain_faster_rcnn_model = Derain_FasterRCNN(FasterRCNN=self.thread1.basic_model,DerainNet=self.thread1.de_rain_basic_model,device=self.thread1.device)
        self.thread1.de_rain_depth_trans_faster_rcnn_model = Derain_FasterRCNN(FasterRCNN=self.thread1.fine_tune_model,DerainNet=self.thread1.de_rain_depth_trans_model,device=self.thread1.device)
        QMessageBox.information(self, "提示", "切换kitti_model成功",QMessageBox.Yes)

    def update_model_flag_2(self):
        global MODEL_FLAG
        MODEL_FLAG = "CITYSCAPES"
        derain_basic_weights = "../models/cityscapes_derain_basic.pth"
        derain_weights = "../models/cityscapes_derain_depth_cft.pth"
        train_weights = "../models/cityscapes_20220219_39.pth"
        train_fine_tune_weights ="../models/cityscapes_depth_transformer_fixup_41.pth"
        self.thread1.basic_model = FasterRCNN(backbone=self.thread1.backbone, num_classes=8 + 1, rpn_score_thresh=0.5)
        self.thread1.basic_model.load_state_dict(torch.load(train_weights, map_location=self.thread1.device)["model"])
        self.thread1.fine_tune_model = FasterRCNN(backbone=self.thread1.backbone, num_classes=8 + 1,rpn_score_thresh=0.5)
        self.thread1.fine_tune_model.load_state_dict(torch.load(train_fine_tune_weights, map_location=self.thread1.device)["model"])
        self.thread1.de_rain_basic_model.load_state_dict(torch.load(derain_basic_weights))
        self.thread1.de_rain_depth_trans_model.load_state_dict(torch.load(derain_weights))
        self.thread1.de_rain_faster_rcnn_model = Derain_FasterRCNN(FasterRCNN=self.thread1.basic_model,DerainNet=self.thread1.de_rain_basic_model, device=self.thread1.device)
        self.thread1.de_rain_depth_trans_faster_rcnn_model = Derain_FasterRCNN(FasterRCNN=self.thread1.fine_tune_model,DerainNet=self.thread1.de_rain_depth_trans_model,device=self.thread1.device)
        QMessageBox.information(self, "提示", "切换cityscapes_model成功",QMessageBox.Yes)

    def update_listWidget_image_1(self):
        self.update_listWidget_base(self.action_rain_kitti.text())

    def update_listWidget_image_2(self):
        self.update_listWidget_base(self.action_rain_cityscape.text())

    def update_listWidget_image_3(self):
        self.update_listWidget_base(self.action_real_world.text())

    def update_listWidget_base(self,image_data_name):
        self.listWidget_image.currentItemChanged.disconnect(self.update_image)
        self.listWidget_image.clear()
        self.listWidget_image.currentItemChanged.connect(self.update_image)
        if "kitti" in image_data_name:
            for path in os.listdir(KITTI_PATH):
                self.listWidget_image.addItem(KITTI_PATH + "/" + path)
        elif "cityscapes" in image_data_name:
            for path in os.listdir(CITYSCAPES_PATH):
                self.listWidget_image.addItem(CITYSCAPES_PATH + "/" + path)
        else:
            for path in os.listdir(REAL_PATH):
                self.listWidget_image.addItem(REAL_PATH + "/" + path)

    def update_image(self):
        global IMAGE_PATH,DEPTH_PATH
        image_path = self.listWidget_image.currentItem().text()
        show_img(image_path, self.graphics_input)
        self.input_path.setText(image_path)
        IMAGE_PATH = image_path
        if "real" in image_path:
            DEPTH_PATH = ""
            self.graphics_depth.setScene(None)
        elif "kitti" in image_path:
            depth_path = image_path.replace("/rain/", "/depth/")
            DEPTH_PATH = depth_path
            show_img(depth_path, self.graphics_depth)
        elif "cityscapes" in image_path:
            depth_path = image_path.replace("/rain/", "/depth/").split("_leftImg8bit")[0] + "_depth_rain.png"
            DEPTH_PATH = depth_path
            show_img(depth_path,self.graphics_depth)

    def show_res(self,img_path:str,flag:bool):
        show_img(img_path,self.graphic_res)
        if not flag:
            return
        show_img(SAVE_PATH + "result_derain.png", self.graphics_derain)
        if not os.path.exists(SAVE_PATH+"temp_data.json"):
            return
        f2 = open(SAVE_PATH+"temp_data.json", 'r')
        info_data = json.load(f2)
        row = self.tab_class_num.rowCount()
        for i in range(row+1):
            self.tab_class_num.removeRow(i)
        for idx,row_name in enumerate(info_data.keys()):
            if idx >= self.tab_class_num.rowCount():
                self.tab_class_num.insertRow(idx)  # 尾部插入一行新行表格
            self.tab_class_num.setItem(idx, 0, QTableWidgetItem(row_name))
            self.tab_class_num.setItem(idx, 1, QTableWidgetItem(str(len(info_data[row_name]))))
        list_classes = list(info_data.keys())
        self.label_class.setText(list_classes[0])
        self.image_widget.set_img_path_class(img_path,list_classes,info_data)
        self.image_widget.show_images_list(list_classes[0],self.listWidget_image.currentItem().text())

    def show_depth(self,depth_path):
        global DEPTH_PATH
        DEPTH_PATH = depth_path
        show_img(depth_path, self.graphics_depth)

    def process(self):
        self.thread1.start()

    def depth_predit(self):
        self.thread2.start()


class Depth_thread(QThread):
    signal0 = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_path = os.path.join("../models", "mono+stereo_640x192")
        encoder_path = os.path.join(self.model_path, "encoder.pth")
        depth_decoder_path = os.path.join(self.model_path, "depth.pth")

        # LOADING PRETRAINED MODEL
        self.encoder = monodepth2.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)

        # extract the height and width of image that this models was trained with
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()
        self.depth_decoder = monodepth2.DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        loaded_dict = torch.load(depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict)
        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()
    def run(self):
        with torch.no_grad():
            input_image = Image.open(IMAGE_PATH).convert('RGB')
            input_image = input_image.resize((self.feed_width, self.feed_height), Image.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(self.device)
            features = self.encoder(input_image)
            outputs = self.depth_decoder(features)

            disp = outputs[("disp", 0)]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            scaled_disp = scaled_disp[:, 0].cpu().numpy()
            scaled_disp = np.concatenate(scaled_disp)
            np.save('temp_disp.npy', scaled_disp)
            image_rgb = cv2.imread(IMAGE_PATH)
            reference = image_rgb  # uint8
            im_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            con = confidence(im_gray)  # you shall try with *or* without confidence
            im_shape = reference.shape[:2]
            dis = np.load('temp_disp.npy')
            disp = cv2.resize(dis, (im_shape[1], im_shape[0]), cv2.INTER_AREA)
            focal_length, baseline = 2262, 0.22
            depth = (baseline * focal_length) / (disp * 2048)
            depth_enc = np.minimum(depth * 256., 2 ** 16 - 1).astype(np.uint16)
            cv2.imwrite('test_image_disp.jpg', depth_enc)
            image_depth = cv2.imread('test_image_disp.jpg', 0)
            target = image_depth  # uint8
            grid = BilateralGrid(reference, **grid_params)
            t = target.reshape(-1, 1).astype(np.double) / (pow(2, 16) - 1)
            c = con.reshape(-1, 1).astype(np.double) / (pow(2, 16) - 1)
            output_solver = BilateralSolver(grid, bs_params).solve(t, c).reshape(im_shape)
            depth_filtersolver = np.uint16(output_solver * (pow(2, 16) - 1))
            depth_filtersolver_m = depth_filtersolver / 256.  # As meters
            cv2.imwrite(SAVE_PATH+"depth_grey.jpg", (depth_filtersolver_m*256.).astype(np.uint16))
            self.signal0.emit(SAVE_PATH + "depth_grey.jpg")

class Mythread(QThread):
    signal0 = pyqtSignal(str,bool)
    signal1 = pyqtSignal(str,str)
    signal2 = pyqtSignal(str)
    last_mode_flag = 0
    def __init__(self):
        super().__init__()
        self.num_classes = 7 if MODEL_FLAG == "KITTI" else 8
        self.train_fine_tune_weights = "../models/kitti_depth_transformer_fixup_48.pth" if MODEL_FLAG == "KITTI" else "../models/cityscapes_depth_transformer_fixup_41.pth"
        self.train_weights = "../models/kitti_20211222_39.pth" if MODEL_FLAG == "KITTI" else "../models/cityscapes_20220219_39.pth"
        self.derain_weights = "../models/kitti_derain_depth_cft.pth" if MODEL_FLAG == "KITTI" else "../models/cityscapes_derain_depth_cft.pth"
        self.derain_basic_weights = "../models/kitti_derain_basic.pth" if MODEL_FLAG == "KITTI" else "../models/cityscapes_derain_basic.pth"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)

        self.de_rain_depth_trans_model = DDGN_Depth_CFT()
        self.de_rain_basic_model = DDGN_Basic()
        self.de_rain_depth_trans_model.load_state_dict(torch.load(self.derain_weights))
        self.de_rain_basic_model.load_state_dict(torch.load(self.derain_basic_weights))

        self.fine_tune_model = FasterRCNN(backbone=self.backbone, num_classes=self.num_classes + 1, rpn_score_thresh=0.5)
        self.fine_tune_model.load_state_dict(torch.load(self.train_fine_tune_weights, map_location=self.device)["model"])
        self.basic_model = FasterRCNN(backbone=self.backbone, num_classes=self.num_classes + 1, rpn_score_thresh=0.5)
        self.basic_model.load_state_dict(torch.load(self.train_weights, map_location=self.device)["model"])
        self.de_rain_faster_rcnn_model = Derain_FasterRCNN(FasterRCNN=self.basic_model, DerainNet=self.de_rain_basic_model, device=self.device)
        self.de_rain_depth_trans_faster_rcnn_model = Derain_FasterRCNN(FasterRCNN=self.fine_tune_model, DerainNet=self.de_rain_depth_trans_model, device=self.device)


    def run(self):
        label_json_path = '../pascal_voc_classes_kitti.json' if MODEL_FLAG == "KITTI" else '../pascal_voc_classes_cityscapes.json'
        json_file = open(label_json_path, 'r')
        class_dict = json.load(json_file)
        json_file.close()
        category_index = {v: k for k, v in class_dict.items()}
        if IMAGE_PATH == "":
            return
        original_img = Image.open(IMAGE_PATH).convert('RGB')
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        img = torch.unsqueeze(img, dim=0)

        if MODE_FLAG == 1:
            self.de_rain_faster_rcnn_model.to(self.device)
            self.de_rain_faster_rcnn_model.eval()
        elif MODE_FLAG == 2:
            self.basic_model.to(self.device)
            self.basic_model.eval()
        else:
            self.de_rain_depth_trans_faster_rcnn_model.to(self.device)
            self.de_rain_depth_trans_faster_rcnn_model.eval()
        with torch.no_grad():
            t_start = time_synchronized()
            if MODE_FLAG == 1:
                out_imgs, predictions = self.de_rain_faster_rcnn_model(rains=img)
            elif MODE_FLAG == 2:
                predictions = self.basic_model(img.cuda())
            else:
                if DEPTH_PATH == "":
                    print("检测真实图片请先预测深度图")
                    return
                else:
                    original_depth = Image.open(DEPTH_PATH).convert('L')
                    depth = data_transform(original_depth)
                    depth = torch.unsqueeze(depth, dim=0)
                    out_imgs, predictions = self.de_rain_depth_trans_faster_rcnn_model(rains=img, depths=depth)
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))
            predict_boxes = predictions[0]["boxes"].to("cpu").numpy()
            predict_classes = predictions[0]["labels"].to("cpu").numpy()
            predict_scores = predictions[0]["scores"].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
                original_img.save(SAVE_PATH + "result.png")
            else:
                if MODE_FLAG!=2:
                    to_pil = transforms.ToPILImage()
                    to_pil(out_imgs[0]).save(SAVE_PATH + "result_derain.png")
                    temp_derain_img = Image.open(SAVE_PATH + "result_derain.png").convert('RGB')
                if COMBINE_FLAG and MODE_FLAG!=2:
                    json_Data = draw_box(temp_derain_img,
                         predict_boxes,
                         predict_classes,
                         predict_scores,
                         category_index,
                         thresh=0.5,
                         line_thickness=3)
                    temp_derain_img.save(SAVE_PATH + "result.png")
                else:
                    json_Data = draw_box(original_img,
                         predict_boxes,
                         predict_classes,
                         predict_scores,
                         category_index,
                         thresh=0.5,
                         line_thickness=3)
                    original_img.save(SAVE_PATH + "result.png")
                fileObject = open(SAVE_PATH + "temp_data.json", 'w')
                fileObject.write(json_Data)
                fileObject.close()

            if MODE_FLAG == 1:
                self.signal1.emit(SAVE_PATH + "result.png", SAVE_PATH + "result_derain.png")
            elif MODE_FLAG == 2:
                self.signal2.emit(SAVE_PATH + "result.png")
            else:
                if len(predict_boxes) == 0:
                    self.signal0.emit(SAVE_PATH + "result.png",False)
                else:
                    self.signal0.emit(SAVE_PATH + "result.png", True)

class Basic_Derain_Widget(QWidget, basic_derain_Ui_Form):
    def __init__(self,thread):
        QWidget.__init__(self)
        gui.Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.thread = thread
        self.thread.signal1.connect(self.show_res_1)

    def show_res_1(self,img_path,restore_path):
        show_img(img_path,self.graphics_derain_detect)
        show_img(restore_path, self.graphics_derain_restore)

class Basic_Widget(QWidget, basic_Ui_Form):
    def __init__(self,thread):
        QWidget.__init__(self)
        gui.Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.thread = thread
        self.thread.signal2.connect(self.show_res_2)

    def show_res_2(self,img_path):
        show_img(img_path,self.graphics_raw)

def show_img(img_path,graphic:QtWidgets.QGraphicsView):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    minsize = QSize(img.shape[1] * scale, img.shape[0] * scale)
    pix = QtGui.QPixmap(
        QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888).scaled(minsize, QtCore.Qt.IgnoreAspectRatio))
    scene = QGraphicsScene()
    scene.addItem(QGraphicsPixmapItem(pix))
    graphic.setScene(scene)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_teal.xml')
    md = MainCode()
    md.show()
    bdw = Basic_Derain_Widget(md.thread1)
    bdw.show()
    bw = Basic_Widget(md.thread1)
    bw.show()
    sys.exit(app.exec_())
