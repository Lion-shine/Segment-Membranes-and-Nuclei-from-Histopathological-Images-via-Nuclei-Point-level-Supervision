import os, json
import deepdish as dd
from imageio import imread, imsave
import numpy as np
import scipy.ndimage as ndimage
from collections import OrderedDict
import cv2
import scipy
from scipy.ndimage.morphology import distance_transform_edt
from data_preprocess.utils import voronoi_finite_polygons_2d, poly2mask
from skimage import morphology
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from skimage import io
HDF5_DISABLE_VERSION_CHECK = 10

def walk_dir(data_dir, file_types):
    #file_types = ['.txt', '.kfb']
    path_list = []
    for dirpath, dirnames, files in os.walk(data_dir):
        for f in files:
            for this_type in file_types:
                if f.endswith(this_type):
                    path_list.append( os.path.join(dirpath, f)  )
                    break
    return path_list
def getColorList():

    dict = OrderedDict()

    # 橙色
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list

    # 黄色
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list
    return dict
def get_color(frame, color_dict):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    countors=[]
    for d in ['yellow','orange']:
        mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        cnts, hiera = cv2.findContours(binary.copy(),  cv2.RETR_LIST, 2)
        countors.append(cnts)

    return countors
def get_label_point(image_shape, label_points):
    width,height=image_shape[1],image_shape[0]
    label_regions_center = np.zeros((image_shape[0], image_shape[1]),
                                    dtype=np.uint8)

    if os.path.exists(label_points):
        with open(label_points, "r", encoding="utf-8") as f:
            index_info = json.load(f)
            if "roilist" in index_info:
                roilist = index_info['roilist']
                for roi in roilist:
                    remark = roi.get("remark")
                    path = roi.get("path")
                    if remark in label_dict and 'x' in path and 'y' in path:
                        x = int(path['x'][0])
                        y = int(path['y'][0])
                        x = np.clip(x, 0, width-1 ).astype(int)
                        y = np.clip(y, 0, height-1).astype(int)
                        label_regions_center[y,x]=255

    return label_regions_center
def get_voronoi_edge(label_point):
    h, w = label_point.shape

    points = np.argwhere(label_point > 0)
    vor = Voronoi(points)

    regions, vertices = voronoi_finite_polygons_2d(vor)
    box = Polygon([[0, 0], [0, w], [h, w], [h, 0]])
    region_masks = np.zeros((h, w), dtype=np.int16)
    edges = np.zeros((h, w), dtype=np.bool)
    count = 1
    for region in regions:
        polygon = vertices[region]
        # Clipping polygon
        poly = Polygon(polygon)
        poly = poly.intersection(box)
        # print(poly)
        polygon = np.array([list(p) for p in poly.exterior.coords])

        mask = poly2mask(polygon[:, 0], polygon[:, 1], (h, w))
        edge = mask * (~morphology.erosion(mask, morphology.disk(1)))
        edges += edge
        region_masks[mask] = count
        count += 1

    edges = (edges > 0).astype(np.uint8) * 255

    return edges
def get_guassion_point(image_path,label_path,map_dis=80):

    image_np =io.imread(image_path)
    # label_np = io.imread(label_name)

    print(image_path)
    # print(np.unique(label_np))

    label_regions_center = get_label_point(image_np.shape, label_path)

    # fuse Voronoi edge and dilated points
    label_point_dilated = morphology.dilation(
        label_regions_center,
        morphology.disk(3))

    edges = get_voronoi_edge(label_regions_center)

    edges_dis = distance_transform_edt(255 - edges)
    point_dis = distance_transform_edt(255 - label_point_dilated)

    edges_dis_norm = edges_dis * map_dis
    point_dis_norm = point_dis * map_dis

    return point_dis_norm
color_dict=getColorList()
def generate_ki67_mask(img_root,json_root, label_dict, label_num, resize_ratio=1,  file_name='train', radius=1 ):

    '''
    :param data_root:  data are stored as  xxxx.png   xxxx.json
    :param label_dict:
    :param label_num:   label_num  =  types of labels + 1
    :param resize_ratio:   ratio to resize labelled images
    :param split: whether to split train/validation  sets,  if None -> No
        Example: if split == 0.1 -> split to 9:1 train/validation
    :param radius: cell mask radius
    :param check_sanity:  if True,  superimpose masks on images
    :param use_grey:
    :fold:divide the dataset into "fold" h5 files
    :return: a h5 file contains images and masks  {"image path" : { "img": {img_np}, "mask": {mask_np} }
    '''
    save_root = img_root.replace('images','h5')
    print(save_root)
    # mask_dir=os.path.join(save_root,file_name,'mask')
    # image_save_dir=os.path.join(save_root,file_name,'image')
    os.makedirs(save_root, exist_ok=True)
    # os.makedirs(image_save_dir, exist_ok=True)
    img_list = walk_dir(img_root, ['.png', '.jpeg', '.tiff', '.jpg'])
    for f_index,img_path in enumerate(img_list):
        _, image_fullname = os.path.split(img_path)
        label_count_dict = {}
        annotation_path = img_path.replace(img_root,json_root).replace('.png','.json')
        if os.path.exists(annotation_path):

            img = imread(img_path)
            height, width,channel = img.shape

            mask = np.zeros((height, width, label_num))
            nuclei_mask = np.zeros((height, width))
            with open(annotation_path, "r", encoding="utf-8") as f:
                index_info = json.load(f)
                if "roilist" in index_info:
                    roilist = index_info['roilist']
                    for roi in roilist:
                        remark = roi.get("remark")
                        path = roi.get("path")
                        if remark in label_dict and 'x' in path and 'y' in path:
                            x = int(path['x'][0] * resize_ratio)
                            y = int(path['y'][0] * resize_ratio)

                            if x < radius - 1 or x > width - radius or y < radius - 1 or y > height - radius:
                                continue

                            r = radius - 1
                            if r > 0:
                                nuclei_mask[y - r:y + r, x - r:x + r] = 1

                            mask[y, x, label_dict[remark]] = 1

                            if remark not in label_count_dict:
                                label_count_dict[remark] = 1
                            else:
                                label_count_dict[remark] += 1
                    smooth_nuclei_mask = scipy.ndimage.filters.gaussian_filter(nuclei_mask,
                                                                        int(3.4 * resize_ratio * 2))
                    _min, _max = np.min(smooth_nuclei_mask), np.max(smooth_nuclei_mask)
                    smooth_nuclei_mask = (smooth_nuclei_mask - _min + 1e-10) / (_max - _min + 1e-10)
                    smooth_nuclei_mask = (smooth_nuclei_mask * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join('nuclei_masks', os.path.basename(img_path)), smooth_nuclei_mask)
                    num_mask = mask.shape[2]
                    for i in range(num_mask - 1):
                        if np.sum(mask[:, :, i])==0.:
                            continue
                        smooth_mask = scipy.ndimage.filters.gaussian_filter(mask[:, :, i],
                                                                            int(3.4 * resize_ratio * 2))
                        _min, _max = np.min(smooth_mask), np.max(smooth_mask)
                        smooth_mask = (smooth_mask - _min + 1e-10) / (_max - _min + 1e-10)
                        smooth_mask[mask[:, :, i] == 1] = 1
                        mask[:, :, i] = (smooth_mask * 255).astype(np.uint8)
                        # cv2.imwrite("mask_{}.png".format(i), mask[:,:,i])

                    mask[:, :, -1] = 255 - np.sum(mask[:, :, 0:-1], axis=2)
                    if f_index==0:
                        for i in range(mask.shape[-1]):
                            cv2.imwrite("mask_{}.png".format("%d"%i), mask[:, :, i])
                    data_dict = {"img": img, "mask": mask, "point_dist": smooth_nuclei_mask}
                    dd.io.save(
                        os.path.join(save_root, "%s_%d.h5" % (os.path.basename(img_root), f_index)), data_dict)
                    # import pdb; pdb.set_trace()


def get_annotation_summary(data_root):
    # get the number of different labels
    json_list = walk_dir(data_root, ['.json'])
    annotation_dict = {}
    for json_path in json_list:
        with open(json_path, "r", encoding="utf-8") as f:
            index_info = json.load(f)
            if "roilist" in index_info:
                roilist = index_info['roilist']
                for roi in roilist:
                    remark = roi.get("remark")
                    if remark not in annotation_dict.keys():
                        annotation_dict[remark] = 1
                    else:
                        annotation_dict[remark] += 1
    annotation_label, annotation_count = [], []

    for k, v in annotation_dict.items():
        annotation_label.append(k)
        annotation_count.append(v)
    result = {"Label":annotation_label, "Count":annotation_count}
    print(annotation_dict)
    # df = pandas.DataFrame(result)
    # df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ki67_region_{}_summary.csv".format(os.path.basename(data_root))), index=False, encoding="utf-8")

if __name__ == "__main__":
    label_dict = {
        '微弱的不完整膜阳性肿瘤细胞': 0,
        '弱-中等的完整细胞膜阳性肿瘤细胞': 1,
        '阴性肿瘤细胞': 2,
        #'纤维细胞': 3,
        #'淋巴细胞': 4,
        #'难以区分的非肿瘤细胞': 5,
        #'组织细胞': 6,
        '强度的完整细胞膜阳性肿瘤细胞': 1,
        '中-强度的不完整细胞膜阳性肿瘤细胞': 0

    }
    label_num = 3 + 1
    for mode in ['train','test']:
        # img_root="../new_data/data_trainvaltest/%s/images"%mode
        json_root="../new_data/data_trainvaltest/%s/jsons"%mode
        # generate_ki67_mask(img_root,json_root,label_dict, label_num, resize_ratio=1, radius=10)
        get_annotation_summary(json_root)
    # generate_ki67_mask("F:\project\KI67\KI67_region",label_dict, label_num, check_sanity=True, use_grey=False)

