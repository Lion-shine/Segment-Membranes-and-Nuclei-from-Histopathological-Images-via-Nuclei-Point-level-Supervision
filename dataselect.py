import os
import cv2
import shutil
from pydaily import filesystem
def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir

    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist

def prepare_data(imglist,save_dir,pre_dix):
    count = 1
    for index,img_path in enumerate(imglist):
        base_name = os.path.basename(img_path)
        ext = base_name.split('.')[1]
        ext = '.' + ext
        base_name=os.path.basename(img_path)
        if pre_dix==None:
            if count%3==0:
                try:
                    label_path = img_path.replace(base_name, 'result.json')
                    shutil.copy(label_path, os.path.join(save_dir, 'val', 'labels', str(count) + '.json'))
                    shutil.copy(img_path,os.path.join(save_dir,'val','images',str(count)+ext))
                    count+=1
                except:
                    continue
            else:
                try:
                    label_path = img_path.replace(base_name, 'result.json')
                    shutil.copy(label_path, os.path.join(save_dir, 'train', 'labels', str(count) + '.json'))
                    shutil.copy(img_path, os.path.join(save_dir, 'train', 'images', str(count) +ext))
                    count += 1
                except:
                    continue
        else:
            if count % 3 == 0:

                label_path = img_path.replace(base_name, 'result.json')
                if os.path.exists(label_path):
                    shutil.copy(label_path, os.path.join(save_dir, 'val', 'labels', pre_dix+'_'+str(count) + '.json'))
                    shutil.copy(img_path, os.path.join(save_dir, 'val', 'images', pre_dix+'_'+str(count) + ext))
                    count += 1
                else:
                    print(label_path)
            else:

                label_path = img_path.replace(base_name, 'result.json')
                if os.path.exists(label_path):
                    shutil.copy(label_path, os.path.join(save_dir, 'train', 'labels', pre_dix+'_'+str(count) + '.json'))
                    shutil.copy(img_path, os.path.join(save_dir, 'train', 'images', pre_dix+'_'+str(count) + ext))
                    count += 1
                else:
                    print(label_path)

org_img_folder = '/media/deepin/Research/Research/bixirou/region/ori_data/220615bxr数据整理'
save_dir='../data'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    os.mkdir(os.path.join(save_dir,'train'))
    os.mkdir(os.path.join(save_dir, 'train','images'))
    os.mkdir(os.path.join(save_dir, 'train', 'labels'))
    os.mkdir(os.path.join(save_dir, 'val'))
    os.mkdir(os.path.join(save_dir, 'val', 'images'))
    os.mkdir(os.path.join(save_dir, 'val', 'labels'))
# 检索文件
pre_dix=1
for site in os.listdir(org_img_folder):
    if not os.path.isdir(os.path.join(org_img_folder,site)):continue
    file_list=[]
    imglist = filesystem.find_ext_files(os.path.join(org_img_folder,site), ".jpg")
    imglist+=filesystem.find_ext_files(os.path.join(org_img_folder,site), ".tif")
    imglist=sorted(imglist)
    print('本次执行检索到 ' + str(len(imglist)) + ' 张图像\n')
    prepare_data(imglist,save_dir,str(pre_dix))
    pre_dix+=1

# for imgpath in imglist:
#     imgname = os.path.splitext(os.path.basename(imgpath))[0]
#     img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
