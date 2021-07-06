import os
from os import path
import time
from argparse import ArgumentParser
from collections import defaultdict
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

from model.eval_network import PropagationNetwork
from dataset.davis_test_dataset import DAVISTestDataset
from inference_core import InferenceCore

from dataset.range_transform import im_normalization
from dataset.util import all_to_onehot

from progressbar import progressbar

def load_image_frames(vid, frame_ids):
    global args, im_transform
    images = []
    for fid in frame_ids:
        img = Image.open(path.join(args.imdir, vid, fid)).convert('RGB')
        images.append(im_transform(img))
    images = torch.stack(images, 0)
    return images

def load_mask_frames(vid, eid, frame_ids, shape):
    global args
    masks = []
    masks = np.zeros((len(frame_ids), shape[0], shape[1]))
    gt_mask_file = path.join(args.output, vid, eid, frame_ids[0].replace('jpg', 'png'))
    gt_mask = np.array(Image.open(gt_mask_file).resize(shape[::-1], resample=Image.NEAREST).convert('P'), dtype=np.uint8)
    masks[0] = gt_mask
    return masks

def prepare_data(vid, eid, frame_ids):
    global mask_transform
    info = {}
    first_frame = np.array(Image.open(path.join(args.imdir, vid, frame_ids[0])))
    shape = np.shape(first_frame)[:2]
    info['size'] = shape
    images = load_image_frames(vid, frame_ids)
    masks = load_mask_frames(vid, eid, frame_ids, shape)
    labels = [1]
    masks = torch.from_numpy(all_to_onehot(masks, labels)).float()

    # Resize to 480p
    masks = mask_transform(masks)
    masks = masks.unsqueeze(2)
    info['labels'] = labels
    
    return {
        'rgb': images,
        'gt': masks,
        'info': info
    }


    


# Start eval
def propagate(data, prop_model):
    global total_process_time
    global total_frames
    global palette
    rgb = data['rgb'].cuda()
    msk = data['gt'].cuda()
    info = data['info']
    k = len(info['labels'])
    size = info['size']

    torch.cuda.synchronize()
    process_begin = time.time()

    processor = InferenceCore(prop_model, rgb, k)
    with_bg_msk = torch.cat([
        1 - torch.sum(msk[:,0], dim=0, keepdim=True),
        msk[:,0],
    ], 0).cuda()
    print(with_bg_msk.size())
    processor.interact(with_bg_msk, 0, rgb.shape[1])

    # Do unpad -> upsample to original size 
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

    torch.cuda.synchronize()
    total_process_time += time.time() - process_begin
    total_frames += out_masks.shape[0]

    mask_E = out_masks[-1]
    mask_E.putpallette(palette)
    
    del rgb
    del msk
    del processor
    return mask_E

def compare_iou(mask1, mask2):
    mask1 = np.asarray(mask1 > 0)
    mask2 = np.asarray(mask2 > 0)
    inter = np.sum(mask1 * mask2)
    union = np.sum(mask1 + mask2) - inter
    return inter / (union + 1e-6)


if __name__ == "__main__":
    """
    Arguments loading
    """
    parser = ArgumentParser()
    parser.add_argument('--model', default='saves/propagation_model.pth')
    parser.add_argument('--imdir', default='../DAVIS/2017')
    parser.add_argument('--mskdir', default='../DAVIS/2017')
    parser.add_argument('--metapath', default='../DAVIS/2017')
    parser.add_argument('--output')
    parser.add_argument('--split', help='val/testdev', default='val')
    parser.add_argument('--use_km', action='store_true')
    parser.add_argument('--no_top', action='store_true')
    args = parser.parse_args()

    # Simple setup
    # os.makedirs(out_path, exist_ok=True)
    # palette = Image.open(path.expanduser(davis_path + '/trainval/Annotations/480p/blackswan/00000.png')).getpalette()
    palette = [0, 0, 0, 236, 95, 103, 249, 145, 87, 250, 200, 99, 153, 199, 148, 98, 179, 178, 102, 153, 204, 197, 148, 197, 171, 121, 103, 255, 255, 255, 101, 115, 126, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191, 192, 192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255]
    torch.autograd.set_grad_enabled(False)

    # Load our checkpoint
    prop_saved = torch.load(args.model)
    top_k = None if args.no_top else 50
    if args.use_km:
        prop_model = PropagationNetwork(top_k=top_k, km=5.6).cuda().eval()
    else:
        prop_model = PropagationNetwork(top_k=top_k, km=None).cuda().eval()
    prop_model.load_state_dict(prop_saved)

    im_transform = transforms.Compose([
        transforms.ToTensor(),
        im_normalization,
        transforms.Resize(480, interpolation=Image.BICUBIC),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize(480, interpolation=Image.NEAREST),
    ])

    iou_threshold = 0.5

    meta_data = json.load(open(args.metapath))
    valid_video_list = ['a9f23c9150', '6cc8bce61a', '03fe6115d4', 'a46012c642', 'c42fdedcdd', 'ee9415c553', '7daa6343e6', '4fe6619a47', '0e8a6b63bb', '65e0640a2a', '8939473ea7', 'b05faf54f7', '5d2020eff8', 'a00c3fa88e', '44e5d1a969', 'deed0ab4fc', 'b205d868e6', '48d2909d9e', 'c9ef04fe59', '1e20ceafae', '0f3f8b2b2f', 'b83923fd72', 'cb06f84b6e', '17cba76927', '35d5e5149d', '62bf7630b3', '0390fabe58', 'bf2d38aefe', '8b7b57b94d', '8d803e87f7', 'c16d9a4ade', '1a1dbe153e', 'd975e5f4a9', '226f1e10f7', '6cb5b08d93', '77df215672', '466734bc5c', '94fa9bd3b5', 'f2a45acf1c', 'ba8823f2d2', '06cd94d38d', 'b772ac822a', '246e38963b', 'b5514f75d8', '188cb4e03d', '3dd327ab4e', '8e2e5af6a8', '450bd2e238', '369919ef49', 'a4bce691c6', '64c6f2ed76', '0782a6df7e', '0062f687f1', 'c74fc37224', 'f7255a57d0', '4f5b3310e3', 'e027ebc228', '30fe0ed0ce', '6a75316e99', 'a2948d4116', '8273b59141', 'abae1ce57d', '621487be65', '45dc90f558', '9787f452bf', 'cdcfd9f93a', '4f6662e4e0', '853ca85618', '13ca7bbcfd', 'f143fede6f', '92fde455eb', '0b0c90e21a', '5460cc540a', '182dbfd6ba', '85968ae408', '541ccb0844', '43115c42b2', '65350fd60a', 'eb49ce8027', 'e11254d3b9', '20a93b4c54', 'a0fc95d8fc', '696e01387c', 'fef7e84268', '72d613f21a', '8c60938d92', '975be70866', '13c3cea202', '4ee0105885', '01c88b5b60', '33e8066265', '8dea7458de', 'c280d21988', 'fd8cf868b2', '35948a7fca', 'e10236eb37', 'a1251195e7', 'b2256e265c', '2b904b76c9', '1ab5f4bbc5', '47d01d34c8', 'd7a38bf258', '1a609fa7ee', '218ac81c2d', '9f16d17e42', 'fb104c286f', 'eb263ef128', '37b4ec2e1a', '0daaddc9da', 'cd69993923', '31d3a7d2ee', '60362df585', 'd7ff44ea97', '623d24ce2b', '6031809500', '54526e3c66', '0788b4033d', '3f4bacb16a', '06a5dfb511', '9f21474aca', '7a19a80b19', '9a38b8e463', '822c31928a', 'd1ac0d8b81', 'eea1a45e49', '9f429af409', '33c8dcbe09', '9da2156a73', '3be852ed44', '3674b2c70a', '547416bda1', '4037d8305d', '29c06df0f2', '1335b16cf9', 'b7b7e52e02', 'bc9ba8917e', 'dab44991de', '9fd2d2782b', 'f054e28786', 'b00ff71889', 'eeb18f9d47', '559a611d86', 'dea0160a12', '257f7fd5b8', 'dc197289ef', 'c2bbd6d121', 'f3678388a7', '332dabe378', '63883da4f5', 'b90f8c11db', 'dce363032d', '411774e9ff', '335fc10235', '7775043b5e', '3e03f623bb', '19cde15c4b', 'bf4cc89b18', '1a894a8f98', 'f7d7fb16d0', '61fca8cbf1', 'd69812339e', 'ab9a7583f1', 'e633eec195', '0a598e18a8', 'b3b92781d9', 'cd896a9bee', 'b7928ea5c0', '69c0f7494e', 'cc1a82ac2a', '39b7491321', '352ad66724', '749f1abdf9', '7f26b553ae', '0c04834d61', 'd1dd586cfd', '3b72dc1941', '39bce09d8d', 'cbea8f6bea', 'cc7c3138ff', 'd59c093632', '68dab8f80c', '1e0257109e', '4307020e0f', '4b783f1fc5', 'ebe7138e58', '1f390d22ea', '7a72130f21', 'aceb34fcbe', '9c0b55cae5', 'b58a97176b', '152fe4902a', 'a806e58451', '9ce299a510', '97b38cabcc', 'f39c805b54', '0620b43a31', '0723d7d4fe', '7741a0fbce', '7836afc0c2', 'a7462d6aaf', '34564d26d8', '31e0beaf99']
    for vid in valid_video_list:
        video_info = meta_data['videos'][vid]
        full_frames_ids = sorted(os.listdir(os.path.join(args.imdir, vid)))
        query_frames_ids = sorted(video_info['frames'])
        for eid in video_info['expressions'].keys():
            previous_mask = None
            output_dir = os.path.join(args.output, vid, eid)
            os.makedirs(output_dir, exist_ok=True)
            for ind in range(0, len(query_frames_ids)):
                current_frame_id = query_frames_ids[ind]
                output_path = os.path.join(output_dir, f'{current_frame_id}.png')
                mask_file = os.path.join(args.mskdir, vid, eid, f'{current_frame_id}.png')
                mask_predicted = Image.open(mask_file)
                if (ind == 0 or np.sum(np.asarray(previous_mask)) == 0):
                    mask_predicted.save(output_path)
                    previous_mask = mask_predicted
                    continue
                prev_frame_id = query_frames_ids[ind - 1]
                data = {}
                start_idx = full_frames_ids.index(prev_frame_id + '.jpg')
                end_idx = full_frames_ids.index(current_frame_id + '.jpg')

                data = prepare_data(vid, eid, full_frames_ids[start_idx:end_idx+1])
                
                mask_propagate = propagate(data, prop_model)
                
                iou_score = compare_iou(mask_propagate, mask_predicted)
                result = mask_propagate if iou_score > iou_threshold else mask_predicted
                result.save(output_path)
                previous_mask = result

    total_process_time = 0
    total_frames = 0

    print('Total processing time: ', total_process_time)
    print('Total processed frames: ', total_frames)
    print('FPS: ', total_frames / total_process_time)