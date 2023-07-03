#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 16:13
# @Author  : Can Cui
# @File    : eval_multi_cell_seg.py
# @Software: PyCharm
# @Comment:

import os, json
import torch.nn as nn
import torch
import cv2
import numpy as np
from scipy.misc import imsave, imread
import os, sys
import pandas as pd
import math
import matplotlib.pyplot as plt
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_root)
from options import Options
from multi_cls_cell_seg import cal_ki67_np
import shutil
global label_dict,color_dict
def Hex_to_RGB(hex):
    r = int(hex[1:3],16)
    g = int(hex[3:5],16)
    b = int(hex[5:7], 16)
    rgb = (r,g,b)
    return rgb
label_dict = label_dict = {
        '微弱的不完整膜阳性肿瘤细胞': 0,
        '弱-中等的完整细胞膜阳性肿瘤细胞': 1,
        '强度的完整细胞膜阳性肿瘤细胞': 1,
        '中-强度的不完整细胞膜阳性肿瘤细胞': 0,
        '阴性肿瘤细胞': 2,

    }
Reverse_label_dicr={ind:name for ind,name in enumerate(label_dict.keys())}
# color_dict ={'0': (100, 0, 255),
#              '1': (181, 53, 236),
#              '2': (13, 250, 199),
#              '3': (201, 64, 30),
#              '4': (205, 255, 228),
#              '5': (184, 189, 2),
#              '6': (124, 210, 54),
#              '7': (0, 0, 255),
#              '8': (30, 100, 178)}


color_card_hex={'阴性肿瘤细胞':'#00FF00',
                        '微弱的不完整膜阳性肿瘤细胞':'#FF3399',
                        '强度的完整细胞膜阳性肿瘤细胞':'#FF0033',
                        '弱-中等的完整细胞膜阳性肿瘤细胞':'#FF6633',
                        '中-强度的不完整细胞膜阳性肿瘤细胞':'#4051B5',
                        '淋巴细胞':'#660066',
                        '纤维细胞':'#FFFF00',
                        '血管内皮细胞':'#8DA1D5',
                        '组织细胞':'#0033F',
                        '脂肪细胞':'#80DEFF',
                        '难以区分的非肿瘤细胞':'#AAEA63',
                        '导管内癌阳性肿瘤细胞':'#FF59C2',
                        '导管内癌阴性肿瘤细胞':'#B7AD79',
}
color_card={name:Hex_to_RGB(color_card_hex[name]) for name in color_card_hex.keys() }
color_dict={str(i):color_card[name] for i,name in enumerate(label_dict.keys())}
color_dict={'0': (0, 51, 255), '1': (255, 0, 0), '2': (255, 0, 51), '3': (64, 81, 181),'4': (181,64, 81)}
# color_dict={str(i):tuple(random.randint(0, 255) for _ in range(3)) for i in range(len(list(label_dict.keys())))}
label_card=[]
for cor_ind,key in enumerate(label_dict.keys()):
    this_card=np.ones((100,200,3))
    this_card=this_card*np.array(color_dict[str(cor_ind)])[np.newaxis,np.newaxis,:]
    label_card.append(this_card)
label_card=np.concatenate(label_card,axis=0).astype(np.float32)
imsave('color_card.jpg',label_card)
# cv2.imwrite('color_card.jpg',label_card)
pass
def walk_dir(data_dir, file_types):
    # file_types = ['.txt', '.kfb']
    path_list = []
    for dirpath, dirnames, files in os.walk(data_dir):
        for f in files:
            for this_type in file_types:
                if f.endswith(this_type):
                    path_list.append( os.path.join(dirpath, f) )
                    break
    return path_list

def draw_center(img, center, label):
    img = img.copy()
    num_center = center.shape[0]



    radius = 4
    thickness = -2

    for i in range(num_center):
        if str(label[i]) in color_dict:
            img = cv2.circle(img, (center[i,0], center[i,1]), radius, color_dict[str(label[i])], thickness)
    return img

def gaus_noise(image,mu=0.0,sigma=0.1):
    image=image/255.
    noise=np.random.normal(mu,sigma,image.shape)
    gausian_noise=image+noise
    gausian_noise=np.clip(gausian_noise,0,1)
    gausian_noise=np.uint8(gausian_noise*255)
    return gausian_noise
def compare_with_annotations(test_data_dir, anno_dir,save_title, net, cell_count_dict=None, is_display=False):

    result_save_dir = './result_1022'+'/'+save_title
    os.makedirs(result_save_dir, exist_ok=True)
    os.makedirs(os.path.join(result_save_dir,'images'), exist_ok=True)
    os.makedirs(os.path.join(result_save_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(result_save_dir, 'masks'), exist_ok=True)
    annotation_count_dict = { }
    test_count_dict = {}
    result_summary = {}
    for key in label_dict.keys():
        annotation_count_dict[key]=0
        test_count_dict[key]=0
        result_summary[key]=[]
    annotation_count_dict['细胞总数']=0
    test_count_dict['细胞总数']=0
    result_summary['细胞总数']=[]

    result_dict={}
    image_list = walk_dir(test_data_dir, ['.jpg', '.png'])

    for image_path in image_list:
        image_cur_path, image_fullname = os.path.split(image_path)
        label_fullname = image_fullname.replace('png', 'jpg')
        annotation_path = image_path.replace(test_data_dir, anno_dir).replace('.png', '.jpg')
        print("evaluating image {}".format(image_path))
        image = imread(image_path)
        # image=gaus_noise(image,0.1,0.05)
        cell_count_dict, test_center_coords, test_labels,cell_masks,membrane_masks = count_test_summary(image, net, test_count_dict)
        if is_display:
            final_masks=np.zeros_like(image).astype(np.float)
            membrane_masks=np.where(membrane_masks[:, :, 0] < 0.3, 1, 0).astype(np.uint8)
            cell_masks=np.where(cell_masks[:, :, 0] > 0.5, 1, 0).astype(np.uint8)
            membrane_masks=membrane_masks-cell_masks
            final_masks[:,:,2]=membrane_masks*255
            final_masks[:, :, 1] = cell_masks * 255
            cv2.imwrite(os.path.join(result_save_dir,'images',image_fullname),image[:,:,::-1])
            cv2.imwrite(os.path.join(result_save_dir,'masks',image_fullname),final_masks)
            shutil.copy(annotation_path,os.path.join(result_save_dir,'labels',label_fullname))
def count_annotation_summary(label_dict,annotation_path, annotation_dict):
    total_count = 0


    center_coords = []
    labels = []

    with open(annotation_path, "r", encoding='utf-8') as f:
        annotation_info = json.load(f)
        if "roilist" in annotation_info:
            roilist = annotation_info['roilist']
            for roi in roilist:
                if "remark" in roi:
                    remark = roi['remark']
                    if remark in annotation_dict.keys():
                        annotation_dict[remark] += 1
                        total_count+=1
                        x, y = roi['path']['x'][0], roi['path']['y'][0]
                        center_coords.append([x,y])
                        labels.append(label_dict[remark])


    annotation_dict["细胞总数"] = total_count
    center_coords = np.array(center_coords).astype(np.int)
    labels = np.array(labels).astype(np.int)
    return annotation_dict, center_coords, labels


def count_test_summary(image, net,cell_count_dict):
    center_coords, labels,cell_masks,membrane_masks = cal_ki67_np(image, net)
    total_count = 0
    label_dict = {key:ind for ind,key in enumerate(cell_count_dict.keys())}
    reverse_dict = {}
    for k, v in label_dict.items():
        reverse_dict[v] = k
    for i in range(np.max(labels)):
        this_num = int(np.sum(labels == i))
        cell_count_dict[reverse_dict[i]] = this_num
        total_count += this_num
    cell_count_dict["细胞总数"] = total_count
    return cell_count_dict, center_coords, labels,cell_masks,membrane_masks


def load_partial_state_dict(model, state_dict):
    own_state = model.state_dict()
    # print('own_Dict', own_state.keys(), 'state_Dict',state_dict.keys())
    for a, b in zip(own_state.keys(), state_dict.keys()):
        print(a, '_from model =====_loaded: ', b)
    for name, param in state_dict.items():
        if name is "device_id":
            pass
        else:
            if name not in own_state:
                print('unexpected key "{}" in state_dict'.format(name))
            # if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    name, own_state[name].size(), param.size()))
                # raise
    print('>> load partial state dict: {} initialized'.format(len(state_dict)))
def run_test(test_dir,anno_dir, weight_path,save_title):

    if opt.model['name'] == 'FullNet':
        from FullNet import FullNet as detnet
        net = detnet(opt.model['in_c'], opt.model['out_c'], n_layers=opt.model['n_layers'],
                        growth_rate=opt.model['growth_rate'], drop_rate=opt.model['drop_rate'],
                        dilations=opt.model['dilations'], is_hybrid=opt.model['is_hybrid'],
                        compress_ratio=opt.model['compress_ratio'], layer_type=opt.model['layer_type'])
    elif opt.model['name'] == 'Unet':
        from attention_unet import U_Net as detnet
        net = detnet(opt.model['in_c'], opt.model['out_c'])
    elif opt.model['name'] == 'Att_Unet':
        from attention_unet import AttU_Net as detnet
        net = detnet(opt.model['in_c'], opt.model['out_c'])
    elif opt.model['name'] == 'FCN_pooling':
        from FullNet import FCN_pooling as detnet
        net = detnet(opt.model['in_c'], opt.model['out_c'], n_layers=opt.model['n_layers'],
                        growth_rate=opt.model['growth_rate'], drop_rate=opt.model['drop_rate'],
                        dilations=opt.model['dilations'],compress_ratio=opt.model['compress_ratio'], layer_type=opt.model['layer_type'])
    net=nn.DataParallel(net)
    if torch.cuda.is_available():
        net.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        print('You are using GPU')
    else:
        print('You are using CPU')
    print("=> loading trained model from {}".format(weight_path))
    best_checkpoint = torch.load(weight_path)
    net.load_state_dict(best_checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(best_checkpoint['epoch']))
    net = net.module
    net.eval()
    print('reload detection net weights from {}'.format(weight_path))

    test_name = os.path.basename(test_dir) + '_' +os.path.basename(os.path.dirname(weight_path)) + '_' + os.path.splitext(os.path.basename(weight_path))[0]

    compare_with_annotations(test_data_dir=test_dir, anno_dir=anno_dir,save_title=save_title,net=net, is_display=True)

if __name__ == "__main__":
    import os
    global opt
    opt = Options(isTrain=False)
    opt.parse()
    opt.save_options()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    for mode in ['test']:
        test_dir = "../new_data/data_trainvaltest/%s/images"%mode
        anno_dir="../new_data/data_trainvaltest/%s/labels"%mode
        save_title='ours_gaussian'
        run_test(test_dir, anno_dir,
                 weight_path=opt.test['model_path'],save_title=save_title)
    # run_test(test_dir, 1,
    #          weight_path= os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    #                                    "Model", "att_unet_mix_1_2", "weights_epoch_933_1.5434136871368653.pth" ))

    # run_test(test_dir, 2,
    #          weight_path= os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    #                                    "Model", "nested_unet_mix_2_1", "weights_epoch_219_0.579858023673296.pth" ))
    # run_test(test_dir, 3,
    #          weight_path= os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    #                                    "Model", "unet_mix_1_2", "weights_epoch_243_0.6628976203501225.pth" ))
