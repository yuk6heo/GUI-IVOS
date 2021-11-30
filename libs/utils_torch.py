import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import skimage.color as color

def combine_masks_with_batch(masks, n_obj, th=0.5):
    """ Combine mask for different objects.

    Different methods are the following:

    * `max_per_pixel`: Computes the final mask taking the pixel with the highest
                       probability for every object.

    # Arguments
        masks: Tensor with shape[B, nobj, H, W]. H, W on batches must be same
        method: String. Method that specifies how the masks are fused.

    # Returns
        [B, 1, H, W]
    """

    # masks : B, nobj, h, w
    # output : h,w
    marker = torch.argmax(masks, dim=1, keepdim=True)
    out_mask = torch.unsqueeze(torch.zeros_like(masks)[:,0],1) #[B, 1, H, W]
    for obj_id in range(n_obj):
        try :tmp_mask = (marker == obj_id) * (masks[:,obj_id].unsqueeze(1) > th)
        except: raise NotImplementedError
        out_mask[tmp_mask] = obj_id + 1

    return out_mask


def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]



