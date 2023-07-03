import os
import torch
import cv2
import torch.nn.functional as F
import numpy as np
from PIL import Image
import skimage.morphology as morph
from skimage import measure
from scipy import misc
import utils
from pydaily import filesystem


def scan_image(data_dir,suffix):
    png_list = filesystem.find_ext_files(data_dir, suffix)
    return png_list
def main(label_dir,pred_dir,save_title):
    print("=> Test begins:")

    labels=scan_image(label_dir,'.jpg')
    membrane_metric=[]
    nuclei_metric=[]
    f=open(save_title+'.txt','w')
    f.write('Type'+'\t'+'acc'+'\t'+'iou'+'\t'+'recall'+'\t'+'precison'+'\t'+'F1'+'\t'+'performance'+'\n')
    for a_label_path in labels:
        img_name=os.path.basename(a_label_path).replace('.jpg','.png')
        print('=> Processing image {:s}'.format(img_name))
        pred_path=os.path.join(pred_dir,img_name)
        if not os.path.exists(pred_path):continue
        label=cv2.imread(a_label_path)[:,:,::-1]
        pred=cv2.imread(pred_path)[:,:,::-1]
        if pred.shape[0]!=label.shape[0] or pred.shape[1]!=label.shape[1]:
            pred=cv2.resize(pred,(label.shape[1],label.shape[0]))
        # print(np.unique(label))
        membrane_label=label[:,:,0]
        nuclei_label = label[:, :, 1]
        membrane_pred = pred[:, :, 0]
        nuclei_pred = pred[:, :, 1]
        # load test image

        img_path = '{:s}/{:s}'.format(pred_dir, img_name)
        print('\tComputing output probability maps...')
        membrane_ratio=np.sum(membrane_label)/(np.sum(np.ones_like(membrane_label)*255))
        if membrane_ratio>0.01:
            [m_acc, m_iou, m_recall, m_precision, m_F1, m_performance]= utils.accuracy_pixel_level(np.expand_dims(membrane_label>100,0), np.expand_dims(membrane_pred==255, 0))
        else:
            [m_acc, m_iou, m_recall, m_precision, m_F1, m_performance]=[1.,1.,1.,1.,1.,1.]
        [n_acc, n_iou, n_recall, n_precision, n_F1, n_performance] = utils.accuracy_pixel_level(
            np.expand_dims(nuclei_label>100, 0), np.expand_dims(nuclei_pred==255, 0))

        membrane_metric.append([m_acc, m_iou, m_recall, m_precision, m_F1, m_performance])
        nuclei_metric.append([n_acc, n_iou, n_recall, n_precision, n_F1, n_performance])
        # f.write(img_name+'\t'+'%.4f'+'\t'+'%.4f'+'\t'+'%.4f'+'\t'+'%.4f'+'\t'+'%.4f'+'\t'+'%.4f'+'\n')%(m_acc, m_iou, m_recall, m_precision, m_F1, m_performance)
        # f.write(
        #     img_name + '\t' + '%.4f' + '\t' + '%.4f' + '\t' + '%.4f' + '\t' + '%.4f' + '\t' + '%.4f' + '\t' + '%.4f' + '\n') % (n_acc, n_iou, n_recall, n_precision, n_F1, n_performance)

    membrane_metric=np.array(membrane_metric)
    nuclei_metric=np.array(nuclei_metric)
    average_membrane=np.mean(membrane_metric,axis=0)
    average_nuclei = np.mean(nuclei_metric, axis=0)
    f.write('membrane\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n'%(average_membrane[0], average_membrane[1], average_membrane[2],
                                                              average_membrane[3], average_membrane[4], average_membrane[5]))
    f.write('nuclei\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (
    average_nuclei[0], average_nuclei[1], average_nuclei[2], average_nuclei[3], average_nuclei[4],
    average_nuclei[5]))
    f.close()


def get_probmaps(input, model, opt):
    size = opt.test['patch_size']
    overlap = opt.test['overlap']

    if size == 0:
        with torch.no_grad():
            output = model(input.cuda())
    else:
        output = utils.split_forward(model, input, size, overlap, opt.model['out_c'])
    output = output.squeeze(0)
    prob_maps = F.softmax(output, dim=0).cpu().numpy()

    return prob_maps


def save_results(header, avg_results, all_results, filename, mode='w'):
    """ Save the result of metrics
        results: a list of numbers
    """
    N = len(header)
    assert N == len(avg_results)
    with open(filename, mode) as file:
        # header
        file.write('Metrics:\t')
        for i in range(N - 1):
            file.write('{:s}\t'.format(header[i]))
        file.write('{:s}\n'.format(header[N - 1]))

        # average results
        file.write('Average:\t')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(avg_results[i]))
        file.write('{:.4f}\n'.format(avg_results[N - 1]))
        file.write('\n')

        # all results
        for key, values in sorted(all_results.items()):
            file.write('{:s}:'.format(key))
            for value in values:
                file.write('\t{:.4f}'.format(value))
            file.write('\n')


if __name__ == '__main__':
    label_dir='/media/lihansheng/5603d8f3-704d-4241-903d-fa30f6c8f69c/diyingjia/her2/segmembrane/new_data/data_trainvaltest/test/labels'
    pred_dir='/media/lihansheng/5603d8f3-704d-4241-903d-fa30f6c8f69c/diyingjia/her2/segmembrane/seg_memrane_Octoberb2/result_1022/ours/masks'
    save_title='ours'
    main(label_dir,pred_dir,save_title)
