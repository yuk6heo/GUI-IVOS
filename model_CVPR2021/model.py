from __future__ import division
import torch

# general libs
import numpy as np
import torch.nn.functional as F

# my libs
# from Algorithm_Heo.networks.network_CerBerusnet_v6 import CBnet
from .networks.network import NET_GAmap
from libs import utils_custom

# davis


class model():
    def __init__(self):
        self.net = NET_GAmap()
        self.net.cuda()
        self.net.load_state_dict(torch.load('checkpoints/GIS-ckpt_standard.pth'))
        self.net.eval()
        for param in self.net.parameters(): param.requires_grad = False

        self.mean, self.var = torch.Tensor([0.485, 0.456, 0.406]), torch.Tensor([0.229, 0.224, 0.225])
        self.mean, self.var = self.mean.view(1,1,3,1,1).cuda(),  self.var.view(1,1,3,1,1).cuda()


    def init_with_new_video(self, frames, n_obj=1):
        self.n_objects = n_obj
        self.frames = frames.copy() # f h w 3
        self.num_frames, self.height, self.width = self.frames.shape[:3]
        self.scores_nf = np.zeros([self.num_frames])
        self.annotated_frames=[]

        pad_info = utils_custom.apply_pad(self.frames[0])[1]
        self.hpad1, self.wpad1 = pad_info[0][0], pad_info[1][0]
        self.hpad2, self.wpad2 = pad_info[0][1], pad_info[1][1]
        self.padding = pad_info[1] + pad_info[0]
        self.prob_map_of_frames = torch.zeros((self.num_frames, self.n_objects+1, self.height + sum(pad_info[0]), self.width + sum(pad_info[1]))).requires_grad_(False).cuda() # f,1,p_h,p_w {cudatensor}

        self.all_F = torch.unsqueeze(torch.nn.ReflectionPad2d(self.padding)(
            torch.from_numpy(np.transpose(frames, (0,3,1,2))).float() / 255.), dim=1).requires_grad_(False).cuda()  # fr,1,3,p_h,p_w {cudatensor}
        self.all_F = (self.all_F -self.mean) / self.var
        self.current_round_masks = np.zeros([self.num_frames, self.height, self.width])  # f,h,w {numpy}
        self.anno_6chEnc_r4_list = []
        self.anno_3chEnc_r4_list = []
        self.current_round = 1

        self.one_hot_outputs_anno = None
        self.current_round = 1

    def Run_propagation(self, annotated_now, ):

        print('[Propagation running...]')
        self.current_round +=1
        prop_list = utils_custom.get_prop_list(self.annotated_frames, annotated_now, self.num_frames, proportion=0.99)
        annotated_frames_np = np.array(self.annotated_frames)
        prop_fore = sorted(prop_list)[0]
        prop_rear = sorted(prop_list)[-1]

        flag = 0  # 1: propagating backward, 2: propagating forward
        for operating_frame in prop_list:
            if operating_frame == annotated_now:
                if flag == 0:
                    flag += 1
                    adjacent_to_anno = True
                    continue
                elif flag == 1:
                    flag += 1
                    adjacent_to_anno = True
                    continue
                else:
                    raise NotImplementedError
            else:
                print('operating in : {:03d}'.format(operating_frame))
                if adjacent_to_anno:
                    r4_neighbor = self.r4_anno
                    neighbor_pred_onehot = self.anno_onehot_prob
                    adjacent_to_anno = False
                else:
                    r4_neighbor = r4_que
                    neighbor_pred_onehot = targ_onehot_prob

                output_logit, r4_que, score = self.net.forward_prop(
                    self.anno_3chEnc_r4_list, self.all_F[operating_frame].repeat(self.n_objects,1,1,1), self.anno_6chEnc_r4_list,
                    r4_neighbor, neighbor_pred_onehot,
                    anno_fr_list= annotated_frames_np, que_fr= operating_frame)  # [nobj, 1, P_H, P_W]

                output_prob_tmp = F.softmax(output_logit, dim=1) # [nobj, 2, P_H, P_W]
                output_prob_tmp = output_prob_tmp[:, 1] # [nobj, P_H, P_W]
                one_hot_outputs_t = F.softmax(self.soft_aggregation(output_prob_tmp), dim=0) # [nobj+1, P_H, P_W]


                smallest_alpha = 0.5
                if flag==1:
                    sorted_frames = annotated_frames_np[annotated_frames_np < annotated_now]
                    if len(sorted_frames) ==0:
                        alpha = 1
                    else:
                        closest_addianno_frame = np.max(sorted_frames)
                        alpha = smallest_alpha+(1-smallest_alpha)*((operating_frame-closest_addianno_frame)/(annotated_now - closest_addianno_frame))
                else:
                    sorted_frames = annotated_frames_np[annotated_frames_np > annotated_now]
                    if len(sorted_frames) == 0:
                        alpha = 1
                    else:
                        closest_addianno_frame = np.min(sorted_frames)
                        alpha = smallest_alpha+(1-smallest_alpha)*((closest_addianno_frame - operating_frame) / (closest_addianno_frame - annotated_now))


                one_hot_outputs_t = (alpha * one_hot_outputs_t) + ((1 - alpha) * self.prob_map_of_frames[operating_frame])
                self.prob_map_of_frames[operating_frame] = one_hot_outputs_t
                targ_onehot_prob = one_hot_outputs_t.clone()[1:].unsqueeze(1) # [nobj, 1, P_H, P_W]

                self.scores_nf[operating_frame] = score

        self.current_round_masks = torch.argmax(self.prob_map_of_frames,dim=1).cpu().numpy().astype(np.uint8)[:,self.hpad1:-self.hpad2, self.wpad1:-self.wpad2]

        print('[Propagation process is done.]')


    def Run_interaction(self, scribbles):

        print('[Interaction running...]')
        annotated_now = scribbles['annotated_frame']
        scribbles_list = scribbles['scribbles']

        pm_ps_ns_3ch_t=[] # n_obj,3,h,w
        if self.current_round == 1:
            for obj_id in range(1, self.n_objects + 1):
                pos_scrimg, neg_scrimg = utils_custom.scribble_to_image(scribbles_list, annotated_now, obj_id,
                                                                        prev_mask=self.current_round_masks[annotated_now], blur=True,
                                                                        singleimg=False, seperate_pos_neg=True)
                pm_ps_ns_3ch_t.append(np.stack([np.ones_like(pos_scrimg)/2, pos_scrimg, neg_scrimg], axis=0))
            pm_ps_ns_3ch_t = np.stack(pm_ps_ns_3ch_t, axis=0) # n_obj,3,h,w

        else:
            for obj_id in range(1, self.n_objects + 1):
                prev_round_input = (self.current_round_masks[annotated_now] == obj_id).astype(np.float32)  # H,W
                pos_scrimg, neg_scrimg = utils_custom.scribble_to_image(scribbles_list, annotated_now, obj_id,
                                                                        prev_mask=self.current_round_masks[annotated_now], blur=True,
                                                                        singleimg=False, seperate_pos_neg=True)
                pm_ps_ns_3ch_t.append(np.stack([prev_round_input, pos_scrimg, neg_scrimg], axis=0))
            pm_ps_ns_3ch_t = np.stack(pm_ps_ns_3ch_t, axis=0)  # n_obj,3,h,w

        batched_F = self.all_F[annotated_now].repeat(self.n_objects,1,1,1)

        pm_ps_ns_3ch_t = torch.from_numpy(pm_ps_ns_3ch_t).cuda()
        pm_ps_ns_3ch_t = torch.nn.ReflectionPad2d(self.padding)(pm_ps_ns_3ch_t)
        inputs = torch.cat([batched_F, pm_ps_ns_3ch_t], dim=1)

        anno_3chEnc_r4, r2_prev_fromanno = self.net.encoder_3ch.forward(batched_F)
        neighbor_pred_onehot_sal, anno_6chEnc_r4 = self.net.forward_obj_feature_extractor(inputs)  # [nobj, 1, P_H, P_W], # [n_obj,2048,h/16,w/16]

        output_logit, self.r4_anno, score = self.net.forward_prop(
            [anno_3chEnc_r4], batched_F, [anno_6chEnc_r4],
            anno_3chEnc_r4, torch.sigmoid(neighbor_pred_onehot_sal))  # [nobj, 1, P_H, P_W]

        output_prob_tmp = F.softmax(output_logit, dim=1) # [nobj, 2, P_H, P_W]
        output_prob_tmp = output_prob_tmp[:, 1] # [nobj, P_H, P_W]
        one_hot_outputs_t = F.softmax(self.soft_aggregation(output_prob_tmp), dim=0) # [nobj+1, P_H, P_W]

        self.anno_onehot_prob = one_hot_outputs_t.clone()[1:].unsqueeze(1) # [nobj, 1, P_H, P_W]
        self.prob_map_of_frames[annotated_now] = one_hot_outputs_t
        self.current_round_masks[annotated_now] = \
            torch.argmax(self.prob_map_of_frames[annotated_now],dim=0).cpu().numpy().astype(np.uint8)[self.hpad1:-self.hpad2, self.wpad1:-self.wpad2]
        self.scores_nf[annotated_now] = score


        if len(self.anno_6chEnc_r4_list) < self.current_round:
            self.anno_6chEnc_r4_list.append(anno_6chEnc_r4)
            self.anno_3chEnc_r4_list.append(anno_3chEnc_r4)
            self.annotated_frames.append(annotated_now)
        elif len(self.anno_6chEnc_r4_list) == self.current_round:
            self.anno_6chEnc_r4_list[self.current_round-1] = anno_6chEnc_r4
            self.anno_3chEnc_r4_list[self.current_round-1] = anno_3chEnc_r4
        else:
            raise NotImplementedError




        print('[Interaction process is done.]')

    def Get_mask(self):
        return self.current_round_masks


    def Get_mask_index(self, index):
        return self.current_round_masks[index]


    def soft_aggregation(self, ps):
        num_objects, H, W = ps.shape
        em = torch.zeros(num_objects +1, H, W).cuda()
        em[0] =  torch.prod(1-ps, dim=0) # bg prob
        em[1:num_objects+1] = ps # obj prob
        em = torch.clamp(em, 1e-7, 1-1e-7)
        logit = torch.log((em /(1-em)))
        return logit
