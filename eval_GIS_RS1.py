import sys
from PyQt5.QtWidgets import QApplication

from apps.multi_object_gui_eval import App
from libs.davis_interactive_evaluator_mo import Davis_Interactive_Evaluator as DIE
import time

import os
import numpy as np
from davisinteractive.utils.visualization import overlay_mask


class App_CVPR2021(App):
    def __init__(self, DIE, model, root, video_indices, save_imgs=False):
        super().__init__(DIE, model, root, video_indices, save_imgs=save_imgs)

    def on_run(self):
        if len(self.scribbles['scribbles'][self.cursur])>=1:
            self.scribble_timesteps.append(time.time()-self.time_init)
            self.VOS_once_executed_bool = True
            self.model.Run_propagation(self.cursur)
            self.current_mask = self.model.Get_mask()

            self.current_round +=1

            print('[Overlaying segmentations...]')
            for fr in range(self.num_frames):
                self.vis_frames[fr] = overlay_mask(self.frames[fr], self.current_mask[fr], alpha=0.5, contour_thickness=2)
            print('[Overlaying Done.] \n')


            # clear scribble and reset
            self.cursur = np.argmin(self.model.scores_nf)
            self.show_current()
            self.clear_strokes()
            self.reset_scribbles()
            self.lcd2.setText('Currunt round : {:2d}'.format(self.current_round + 1))

            self.operate_timesteps.append(time.time() - self.time_init)
            self.finding_timesteps.append(time.time() - self.time_init)
            self.slider.setDisabled(True)
            self.text_print += 'Providing scribble...\n'
            self.lcd3.setText(self.text_print)

    def on_select(self):
        a=1

class Davis_Interactive_Evaluator(DIE):
    def __init__(self, root, algorithm_name, user_name, imset='2017/val.txt', resolution='480p'):
        super().__init__(root, algorithm_name, user_name, imset=imset, resolution=resolution)


if __name__ == '__main__':


    ##################### Configs ########################
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

    root = '/home/yuk/data_ssd/datasets/DAVIS'
    user_name = 'A'
    ##################### Configs ########################

    from model_CVPR2021.model import model as model
    DIE = Davis_Interactive_Evaluator(root,algorithm_name='RAmap_RS1',user_name=user_name)
    DIE.write_info()
    model = model()

    app = QApplication(sys.argv)

    for val_idx in range(0,30):
        ex = App_CVPR2021(DIE, model, root, video_indices=val_idx, save_imgs=False)
        app.exec_()
        ex = None

