
import numpy as np
import time
import os
import csv
import random
from datetime import datetime

from libs import utils_custom


class Davis_Interactive_Evaluator():
    def __init__(self, root, algorithm_name, user_name, imset='2017/val.txt', resolution='480p'):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)

        self.videos = sorted(self.videos)

        self.current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_root = 'results/Alg[{}]_{}'.format(algorithm_name, self.current_time)
        self.algorithm_name = algorithm_name
        utils_custom.mkdir(self.save_root)

        self.savefname_csv = os.path.join(self.save_root+'/result_{}.csv'.format(user_name))

    def write_info(self):
        with open(self.savefname_csv, mode='a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['sequence', 'obj_id', 'N_rounds', 'final_J', 'final_F', 'scribble_time', 'operation_time', 'finding_time', 'total_time'])

    def write_in_csv(self,sequence, n_obj, final_J, final_F, scribble_timesteps, operate_timesteps, finding_timesteps):
        # write csv
        n_rounds = len(operate_timesteps)
        totaltime = finding_timesteps[-1]

        scribble_time = np.sum(np.array(scribble_timesteps) - np.array([0] + finding_timesteps[:-1]))
        operation_time = np.sum(np.array(operate_timesteps) - np.array(scribble_timesteps))
        finding_time = np.sum(np.array(finding_timesteps) - np.array(operate_timesteps))

        with open(self.savefname_csv, mode='a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for obj_id in range(1,n_obj+1):
                writer.writerow([sequence, obj_id, n_rounds, final_J[obj_id-1], final_F[obj_id-1], scribble_time, operation_time, finding_time, totaltime])

# scr oper find scr oper find scr oper find//



