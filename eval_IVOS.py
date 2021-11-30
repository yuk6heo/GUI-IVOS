import sys
from PyQt5.QtWidgets import QApplication

from apps.multi_object_gui_eval import App
from libs.davis_interactive_evaluator_mo import Davis_Interactive_Evaluator as DIE

import os


class App_CVPR2021(App):
    def __init__(self, DIE, model, root, video_indices, save_imgs=False):
        super().__init__(DIE, model, root, video_indices, save_imgs=save_imgs)


class Davis_Interactive_Evaluator(DIE):
    def __init__(self, root, algorithm_name, user_name, imset='2017/val.txt', resolution='480p'):
        super().__init__(root, algorithm_name, user_name, imset=imset, resolution=resolution)


if __name__ == '__main__':

    ##################### Configs ########################
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
    root = '/home/yuk/data_ssd/datasets/DAVIS'
    user_name = 'A'
    ##################### Configs ########################

    from model_CVPR2021.model import model
    DIE = Davis_Interactive_Evaluator(root,algorithm_name='RAmap_IVOS',user_name=user_name)
    DIE.write_info()
    model = model()

    app = QApplication(sys.argv)

    for val_idx in range(0,30):
        ex = App_CVPR2021(DIE, model, root, video_indices=val_idx, save_imgs=True)
        app.exec_()
        ex = None

