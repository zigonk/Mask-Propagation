"""
YouTubeVOS has a label structure that is more complicated than DAVIS 
Labels might not appear on the first frame (there might be no labels at all in the first frame)
Labels might not even appear on the same frame (i.e. Object 0 at frame 10, and object 1 at frame 15)
0 does not mean background -- it is simply "no-label"
and object indices might not be in order, there are missing indices somewhere in the validation set

Dealing with these makes the logic a bit convoluted here
It is not necessarily hacky but do understand that it is not as straightforward as DAVIS

Validation set only.
"""


import os
from os import path
import time
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json

from model.eval_network import PropagationNetwork
from dataset.yv_test_dataset import YouTubeVOSTestDataset
from inference_core_yv import InferenceCore

from progressbar import progressbar

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/propagation_model.pth')
parser.add_argument('--yv', default='../YouTube')
parser.add_argument('--mask', default=None)
parser.add_argument('--metadata', default=None, help='metadata is used for dataloader')
parser.add_argument('--meta_exp', default=None, help='meta expression is used for export mask')
parser.add_argument('--in_range', default=None, nargs='+', type=int)
parser.add_argument('--output')
parser.add_argument('--split', default='valid')
parser.add_argument('--use_km', action='store_true')
parser.add_argument('--no_top', action='store_true')
parser.add_argument('--num_workers', default=0)
args = parser.parse_args()

yv_path = args.yv
out_path = args.output
mask_root = args.mask
metadata = args.metadata
meta_exp_path = args.meta_exp

in_range = None
if args.in_range is not None:
    in_range = slice(args.in_range[0], args.in_range[1], 1)

meta_exp = {}
with open(meta_exp_path) as f:
    meta_exp = json.load(f)

# Simple setup
os.makedirs(out_path, exist_ok=True)
# palette = Image.open(path.expanduser(yv_path + '/valid/Annotations/0a49f5265b/00000.png')).getpalette()
palette = [0, 0, 0, 236, 95, 103, 249, 145, 87, 250, 200, 99, 153, 199, 148, 98, 179, 178, 102, 153, 204, 197, 148, 197, 171, 121, 103, 255, 255, 255, 101, 115, 126, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191, 192, 192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255]

torch.autograd.set_grad_enabled(False)

# Setup Dataset
test_dataset = YouTubeVOSTestDataset(data_root=yv_path, split=args.split, mask_root = mask_root, metadata=metadata, in_range = in_range)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

# Load our checkpoint
prop_saved = torch.load(args.model)
top_k = None if args.no_top else 50
if args.use_km:
    prop_model = PropagationNetwork(top_k=top_k, km=5.6).cuda().eval()
else:
    prop_model = PropagationNetwork(top_k=top_k, km=None).cuda().eval()
prop_model.load_state_dict(prop_saved)

total_process_time = 0
total_frames = 0

# Start eval
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):
    rgb = data['rgb']
    msk = data['gt'][0]
    info = data['info']
    name = info['name'][0]
    k = len(info['labels'][0])
    gt_obj = info['gt_obj']
    size = info['size']
    eid = info['exp_id'][0]
    skip = data['skip']
    print('Processing video ', name, '_', eid)
    if skip:
        print('No available mask')
        this_out_path = path.join(out_path, name, eid)
        os.makedirs(this_out_path, exist_ok=True)
    #     export_frames = meta_exp['videos'][name]['frames']
        msk = msk.detach().cpu().numpy()
        for f in range(msk.shape[0]):
            img_E = Image.fromarray(msk[f])
            img_E.save(os.path.join(this_out_path, info['frames'][f][0].replace('.jpg','.png')))
        continue
    torch.cuda.synchronize()
    process_begin = time.time()

    # Frames with labels, but they are not exhaustively labeled
    frames_with_gt = sorted(list(gt_obj.keys()))

    processor = InferenceCore(prop_model, rgb, num_objects=k)
    # min_idx tells us the starting point of propagation
    # Propagating before there are labels is not useful
    min_idx = 0
    for i, frame_idx in enumerate(frames_with_gt):
        # min_idx = min(frame_idx, min_idx)
        # Note that there might be more than one label per frame
        obj_idx = gt_obj[frame_idx][0].tolist()
        # Map the possibly non-continuous labels into a continuous scheme
        obj_idx = [info['label_convert'][o].item() for o in obj_idx]

        # Append the background label
        with_bg_msk = torch.cat([
            1 - torch.sum(msk[:,frame_idx], dim=0, keepdim=True),
            msk[:,frame_idx],
        ], 0).cuda()

        # We perform propagation from the current frame to the next frame with label
        prev_frame_idx = -1 if (i == 0) else (frame_idx + frames_with_gt[i-1]) // 2
        next_frame_idx = rgb.shape[1] if (i == len(frames_with_gt) - 1) else (frame_idx + frames_with_gt[i+1]) // 2 + 1
        processor.interact(with_bg_msk, frame_idx, prev_frame_idx, next_frame_idx, obj_idx)

    # Do unpad -> upsample to original size (we made it 480p)
    out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
    for ti in range(processor.t):
        prob = processor.prob[:,ti]

        if processor.pad[2]+processor.pad[3] > 0:
            prob = prob[:,:,processor.pad[2]:-processor.pad[3],:]
        if processor.pad[0]+processor.pad[1] > 0:
            prob = prob[:,:,:,processor.pad[0]:-processor.pad[1]]

        prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
        out_masks[ti] = torch.argmax(prob, dim=0)
    
    out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

    # Remap the indices to the original domain
    idx_masks = np.zeros_like(out_masks)
    for i in range(1, k+1):
        backward_idx = info['label_backward'][i].item()
        idx_masks[out_masks==i] = backward_idx

    torch.cuda.synchronize()
    total_process_time += time.time() - process_begin
    total_frames += (idx_masks.shape[0] - min_idx)

    # Save the results
    this_out_path = path.join(out_path, name, eid)
    os.makedirs(this_out_path, exist_ok=True)
#     export_frames = meta_exp['videos'][name]['frames']
    for f in range(idx_masks.shape[0]):
        if f >= min_idx:
            img_E = Image.fromarray(idx_masks[f])
            img_E.putpalette(palette)
            img_E.save(os.path.join(this_out_path, info['frames'][f][0].replace('.jpg','.png')))

#     for fid in export_frames:
#         f = int(fid) - int(first_frame_id)
#         img_E = Image.fromarray(idx_masks[f])
#         img_E.putpalette(palette)
#         img_E.convert('L')
#         img_E.save(os.path.join(this_out_path, f'{fid}.png'))

    del rgb
    del msk
    del processor

print('Total processing time: ', total_process_time)
print('Total processed frames: ', total_frames)
print('FPS: ', total_frames / total_process_time)
