import sys
from PyQt5.QtWidgets import QApplication

from apps.multi_object_gui_eval_nogt import App

import os


class App_CVPR2021(App):
    def __init__(self, model, root, video_name, target_obj, save_imgs=True):
        super().__init__(model, root, video_name, target_obj, save_imgs=save_imgs)


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    ##################### Configs ########################
    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
    from model_CVPR2021.model import model
    model = model()
    root = '/home/yuk/data_ssd/datasets/DAVIS/JPEGImages/480p/blackswan'
    video_name = 'blackswan'
    target_obj = 1
    save_imgs = True
    ##################### Configs ########################




    app = QApplication(sys.argv)

    ex = App_CVPR2021(model, root, video_name, target_obj=target_obj, save_imgs=save_imgs)
    app.exec_()
    ex = None

