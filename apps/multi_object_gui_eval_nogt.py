import sys
from PyQt5.QtWidgets import (QWidget, QApplication, QComboBox, QHBoxLayout,
                             QLabel, QPushButton, QTextEdit,
                             QVBoxLayout, QSlider, QDesktopWidget, QMainWindow)

from PyQt5.QtCore import QTimer, QTime, QCoreApplication, Qt
from PyQt5.QtGui import QFont


from davisinteractive.metrics import batched_jaccard, batched_f_measure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import matplotlib.pyplot as plt
import numpy as np
import time
import os
from datetime import datetime
from PIL import Image

from davisinteractive.utils.visualization import overlay_mask, _pascal_color_map
from libs import utils_custom

class App(QWidget):
    def __init__(self, model, root, video_name, n_obj, save_imgs=False):
        super().__init__()
        self.model = model
        self.root = root
        self.video_idx = 0
        self.frames = utils_custom.load_frames(self.root)  # f h w 3
        self.num_frames, self.height, self.width = self.frames.shape[:3]
        self.vis_frames = self.frames.copy()
        self.cmap = _pascal_color_map()
        self.video_name = video_name
        self.current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.n_obj = n_obj

        # init model
        self.model.init_with_new_video(self.frames, self.n_obj)
        self.current_object = 1

        # Other variables
        self.first_scr = None
        self.current_round = 0
        self.scribble_timesteps = []
        self.operate_timesteps = []
        self.finding_timesteps = []
        self.VOS_once_executed_bool = False
        self.not_started = True

        self.text_print = ''
        self.save_imgs = save_imgs
        self._palette = Image.open('etc/00000.png').getpalette()

        # window settings
        self.setWindowTitle('Demo: CVPR2021_GIS-RAmap')
        self.setGeometry(100, 100, int(self.width*1.2)+300, (int(self.height*1.2)+200))
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        self.show()

        # object buttons

        self.obj1_button = QPushButton('\nAnnotate \nobject 1 [1]\n')
        self.obj1_button.clicked.connect(self.obj1_pressed)
        self.obj1_button.setMaximumHeight(80)
        self.obj1_button.setStyleSheet("background-color: red")
        self.obj1_button.setCheckable(True)
        self.obj1_button.setShortcut('1')

        self.obj2_button = QPushButton('\nAnnotate \nobject 2 [2]\n')
        self.obj2_button.clicked.connect(self.obj2_pressed)
        self.obj2_button.setMaximumHeight(80)
        self.obj2_button.setStyleSheet("background-color: green")
        self.obj2_button.setCheckable(True)

        self.obj3_button = QPushButton('\nAnnotate \nobject 3 [3]\n')
        self.obj3_button.clicked.connect(self.obj3_pressed)
        self.obj3_button.setMaximumHeight(80)
        self.obj3_button.setStyleSheet("background-color: yellow")
        self.obj3_button.setCheckable(True)

        self.obj4_button = QPushButton('\nAnnotate \nobject 4 [4]\n')
        self.obj4_button.clicked.connect(self.obj4_pressed)
        self.obj4_button.setMaximumHeight(80)
        self.obj4_button.setStyleSheet("background-color: blue")
        self.obj4_button.setCheckable(True)

        self.obj5_button = QPushButton('\nAnnotate \nobject 5 [5]\n')
        self.obj5_button.clicked.connect(self.obj5_pressed)
        self.obj5_button.setMaximumHeight(80)
        self.obj5_button.setStyleSheet("background-color: purple")
        self.obj5_button.setCheckable(True)

        if self.n_obj>=2:
            self.obj2_button.setShortcut('2')
            if self.n_obj>=3:
                self.obj3_button.setShortcut('3')
                if self.n_obj>=4:
                    self.obj4_button.setShortcut('4')
                    if self.n_obj>=5:
                        self.obj5_button.setShortcut('5')
        # buttons
        self.prev_button = QPushButton('Prev [<-]')
        self.prev_button.clicked.connect(self.on_prev)
        self.prev_button.setShortcut(Qt.Key_Left)
        self.next_button = QPushButton('Next [->]')
        self.next_button.clicked.connect(self.on_next)
        self.next_button.setShortcut(Qt.Key_Right)
        self.play_button = QPushButton('Play [P]')
        self.play_button.clicked.connect(self.on_play)
        self.play_button.setShortcut('P')
        self.restart_button = QPushButton('Restart the video')
        self.restart_button.clicked.connect(self.restart_video)
        self.run_button = QPushButton('Run VOS [R]')
        self.run_button.pressed.connect(self.on_run_dschange)
        self.run_button.clicked.connect(self.on_run)
        self.run_button.setShortcut('R')
        self.end_button = QPushButton('Satisfied [S]')
        self.end_button.clicked.connect(self.on_end)
        self.end_button.setShortcut('S')

        self.select_button = QPushButton('Select the frame for annotation [Space]')
        self.select_button.clicked.connect(self.on_select)
        self.select_button.setShortcut('space')


        # LCD
        self.lcd1 = QTextEdit()
        self.lcd1.setReadOnly(True)
        self.lcd1.setMaximumHeight(28)
        self.lcd1.setMaximumWidth(100)
        self.lcd1.setText('{: 3d} / {: 3d}'.format(0, self.num_frames-1))

        # LCD#2
        self.lcd2 = QTextEdit()
        self.lcd2.setReadOnly(True)
        self.lcd2.setMaximumHeight(28)
        self.lcd2.setMaximumWidth(self.width)
        self.lcd2.setText('Current round : {:02d}'.format(self.current_round+1))

        # LCD#3
        self.lcd3 = QTextEdit()
        self.lcd3.setReadOnly(True)
        self.lcd3.setMaximumHeight(600)
        self.lcd3.setMaximumWidth(600)
        self.text_print += 'Round [{:02d}]\n'.format(self.current_round+1)
        self.lcd3.setText(self.text_print)

        # slide
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.num_frames-1)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.slide)

        # main figure
        self.fig1 = plt.Figure()
        self.ax1 = plt.Axes(self.fig1, [0., 0., 1., 1.])
        self.ax1.set_axis_off()
        self.fig1.add_axes(self.ax1)
        self.canvas1 = FigureCanvas(self.fig1)

        self.cidpress = self.fig1.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.fig1.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.fig1.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # object buttons
        obj_buttons = QVBoxLayout()
        obj_buttons.addSpacing(20)
        obj_buttons.addWidget(self.obj1_button)
        obj_buttons.addWidget(self.obj2_button)
        obj_buttons.addWidget(self.obj3_button)
        obj_buttons.addWidget(self.obj4_button)
        obj_buttons.addWidget(self.obj5_button)
        obj_buttons.addSpacing(20)


        # navigator for layout
        navi = QHBoxLayout()
        navi.addWidget(self.lcd1)
        navi.addWidget(self.prev_button)
        navi.addWidget(self.play_button)
        navi.addWidget(self.next_button)
        navi.addStretch(1)
        navi.addStretch(1)
        navi.addWidget(self.restart_button)
        navi.addWidget(self.run_button)
        navi.addWidget(self.end_button)
        navi_s = QHBoxLayout()
        navi_s.addWidget(self.select_button)

        # main layout
        layout_main = QVBoxLayout()
        layout_main.addWidget(self.canvas1)
        layout_main.addWidget(self.slider)
        layout_main.addWidget(self.lcd2)
        layout_main.addLayout(navi)
        layout_main.addLayout(navi_s)
        layout_main.setStretchFactor(navi, 1)
        layout_main.setStretchFactor(self.canvas1, 0)

        # demo
        final_demo = QHBoxLayout()
        final_demo.addLayout(obj_buttons)
        final_demo.addSpacing(30)
        final_demo.addLayout(layout_main)
        self.setLayout(final_demo)

        # timer
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.on_time)

        # initialize visualize
        self.current_mask = np.zeros((self.num_frames, self.height, self.width), dtype=np.uint8)
        self.cursur = 0
        self.on_showing = None
        self.show_current()


        # initialize action
        self.reset_scribbles()
        self.pressed = False
        self.on_drawing = None
        self.drawn_strokes = []
        self.obj1_button.setChecked(True)
        self.show()


    def restart_video(self):
        self.__init__(self.model, self.root)

    def show_current(self):
        if self.on_showing:
            self.on_showing.remove()
        self.on_showing = self.ax1.imshow(self.vis_frames[self.cursur])
        self.canvas1.draw()
        self.lcd1.setText('{: 3d} / {: 3d}'.format(self.cursur, self.num_frames-1))
        self.slider.setValue(self.cursur)

    def show_current_anno(self):
        viz = overlay_mask(self.frames[self.cursur], self.current_mask[self.cursur], alpha=0.5, contour_thickness=2)

        if self.on_showing:
            self.on_showing.remove()
        self.on_showing = self.ax1.imshow(viz)
        self.canvas1.draw()
        self.lcd1.setText('{: 3d} / {: 3d}'.format(self.cursur, self.num_frames - 1))
        self.slider.setValue(self.cursur)

    def reset_scribbles(self):
        self.scribbles = {}
        self.scribbles['scribbles'] = [[] for _ in range(self.num_frames)]
        self.scribbles['sequence'] = self.video_name

    def clear_strokes(self):
        # clear drawn scribbles
        if len(self.drawn_strokes) > 0:
            for line in self.drawn_strokes:
                if line is not None:
                    line.pop(0).remove()
            self.drawn_strokes= []
            self.canvas1.draw()

    def slide(self):
        self.clear_strokes()
        self.reset_scribbles()
        self.cursur = self.slider.value()
        self.show_current()
        # print('slide')

    def on_run_dschange(self):
        if len(self.scribbles['scribbles'][self.cursur])>=1:
            self.text_print += 'Running VOS...\n'
            self.lcd3.setText(self.text_print)

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
            self.show_current()
            self.reset_scribbles()
            self.clear_strokes()

            self.lcd2.setText('Current round : {:02d}'.format(self.current_round + 1))
            self.text_print += '\nRound [{:02d}]\n'.format(self.current_round+1)

            self.operate_timesteps.append(time.time() - self.time_init)
            self.slider.setDisabled(False)
            self.text_print += 'Finding a unsatisfying frame...\n'
            self.lcd3.setText(self.text_print)

    def on_select(self):
        self.slider.setDisabled(True)
        if len(self.finding_timesteps) == (len(self.operate_timesteps)-1):
            self.finding_timesteps.append(time.time()-self.time_init)
            self.text_print += 'Providing scribble...\n'
            self.lcd3.setText(self.text_print)

    def on_end(self):
        if self.VOS_once_executed_bool and (len(self.scribbles['scribbles'][self.cursur])==0):
            if len(self.finding_timesteps) == (len(self.operate_timesteps)-1):
                self.finding_timesteps.append(time.time()-self.time_init)
            final_mask = self.model.Get_mask()

            if self.save_imgs:
                save_dir = os.path.join('result_video', '{}_{}'.format(self.video_name, self.current_time), '{}'.format(self.video_name))
                utils_custom.mkdir(save_dir)
                for fr_idx in range(self.num_frames):
                    savefname = os.path.join(save_dir,'{:05d}.png'.format(fr_idx))
                    tmpPIL = Image.fromarray(final_mask[fr_idx].astype(np.uint8), 'P')
                    tmpPIL.putpalette(self._palette)
                    tmpPIL.save(savefname)

            QCoreApplication.instance().quit()

    def on_prev(self):
        self.clear_strokes()
        self.reset_scribbles()
        self.cursur = max(0, self.cursur-1)
        self.show_current()
        # print('prev')

    def on_next(self):
        self.clear_strokes()
        self.reset_scribbles()
        self.cursur = min(self.cursur+1, self.num_frames-1)
        self.show_current()
        # print('next ')

    def on_time(self):
        self.clear_strokes()
        self.reset_scribbles()
        self.cursur += 1
        if self.cursur > self.num_frames-1:
            self.cursur = 0
        self.show_current()

    def on_play(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(300 / 10)

    def on_press(self, event):
        if len(self.finding_timesteps)-len(self.operate_timesteps)==0:
            self.slider.setDisabled(True)
            if self.not_started:
                self.text_print += 'Providing scribble...\n'
                self.lcd3.setText(self.text_print)
                self.time_init = time.time()
                self.not_started = False

            if event.xdata and event.ydata:
                self.pressed = True
                self.stroke = {}
                self.stroke['path'] = []
                self.stroke['path'].append([event.xdata/self.width, event.ydata/self.height])
                if event.button == 1:
                    self.stroke['object_id'] = self.current_object
                else:
                    self.stroke['object_id'] = 0
                self.stroke['start_time'] = time.time()
                self.visualize_annotation(event)

    def on_motion(self, event):
        if len(self.finding_timesteps)-len(self.operate_timesteps)==0:
            self.visualize_annotation(event)


    def on_release(self, event):
        if len(self.finding_timesteps)-len(self.operate_timesteps)==0:
            self.pressed = False
            if event.xdata and event.ydata:
                self.stroke['path'].append([event.xdata/self.width, event.ydata/self.height])
            self.stroke['end_time'] = time.time()
            self.scribbles['annotated_frame'] = self.cursur
            self.scribbles['scribbles'][self.cursur].append(self.stroke)
            self.drawn_strokes.append(self.on_drawing)
            self.on_drawing = None

            self.model.Run_interaction(self.scribbles)
            self.current_mask[self.cursur] = self.model.Get_mask_index(self.cursur)
            self.show_current_anno()

    def visualize_annotation(self, event):
        if self.pressed and event.xdata and event.ydata:
            self.stroke['path'].append([event.xdata/self.width, event.ydata/self.height])

            x = [p[0]*self.width for p in self.stroke['path']]
            y = [p[1]*self.height for p in self.stroke['path']]
            if self.on_drawing:
                self.on_drawing.pop(0).remove()

            if self.stroke['object_id'] == 0:
                self.on_drawing = self.ax1.plot(x,y, marker='o', markersize=4, linewidth=5, color=[0,0,0])
            if self.stroke['object_id'] == self.current_object:
                self.on_drawing = self.ax1.plot(x,y, marker='o', markersize=4, linewidth=5, color=(self.cmap[self.current_object])/320 +0.2)
            self.canvas1.draw()

    def obj1_pressed(self):
        if self.pressed: self.obj1_button.toggle()
        else:
            self.current_object = 1
            self.obj1_button.setChecked(True),  self.obj2_button.setChecked(False), self.obj3_button.setChecked(False)
            self.obj4_button.setChecked(False), self.obj5_button.setChecked(False)
    def obj2_pressed(self):
        if self.pressed: self.obj2_button.toggle()
        else:
            if self.n_obj>=2:
                self.current_object = 2
                self.obj1_button.setChecked(False),  self.obj2_button.setChecked(True), self.obj3_button.setChecked(False)
                self.obj4_button.setChecked(False),  self.obj5_button.setChecked(False)
    def obj3_pressed(self):
        if self.pressed: self.obj3_button.toggle()
        else:
            if self.n_obj>=3:
                self.current_object = 3
                self.obj1_button.setChecked(False),  self.obj2_button.setChecked(False), self.obj3_button.setChecked(True)
                self.obj4_button.setChecked(False),  self.obj5_button.setChecked(False)
    def obj4_pressed(self):
        if self.pressed: self.obj4_button.toggle()
        else:
            if self.n_obj>=4:
                self.current_object = 4
                self.obj1_button.setChecked(False),  self.obj2_button.setChecked(False), self.obj3_button.setChecked(False)
                self.obj4_button.setChecked(True),  self.obj5_button.setChecked(False)
    def obj5_pressed(self):
        if self.pressed: self.obj5_button.toggle()
        else:
            if self.n_obj>=5:
                self.current_object = 5
                self.obj1_button.setChecked(False),  self.obj2_button.setChecked(False), self.obj3_button.setChecked(False)
                self.obj4_button.setChecked(False),  self.obj5_button.setChecked(True)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()