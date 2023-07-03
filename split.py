import shutil
import os

num_split=9

root_dir='/media/lihansheng/5603d8f3-704d-4241-903d-fa30f6c8f69c/diyingjia/her2/segmembrane/seg_memrane_Octoberb2/result/train'
save_dir_root='/media/lihansheng/5603d8f3-704d-4241-903d-fa30f6c8f69c/diyingjia/her2/segmembrane/seg_memrane_Octoberb2/result/split'
os.makedirs(save_dir_root, exist_ok=True)
imgs=os.listdir(os.path.join(root_dir,'images'))
total_num=len(imgs)
for num in range(num_split):
    start=(total_num//num_split)*num
    end=(total_num//num_split)*(num+1)
    if end>=total_num:
        end=total_num-1
    cur_dir=os.path.join(save_dir_root,str(num+1))
    os.makedirs(os.path.join(cur_dir), exist_ok=True)
    os.makedirs(os.path.join(cur_dir,'images'), exist_ok=True)
    os.makedirs(os.path.join(cur_dir, 'jsons'), exist_ok=True)
    os.makedirs(os.path.join(cur_dir, 'masks'), exist_ok=True)
    move_imgs=imgs[start:end]
    for a_imgs_name in move_imgs:
        s_img_path=os.path.join(root_dir,'images',a_imgs_name)
        s_json_path = os.path.join(root_dir, 'jsons', a_imgs_name.replace('png', 'json'))
        s_mask_path=os.path.join(root_dir,'masks',a_imgs_name)
        t_img_path = os.path.join(cur_dir, 'images', a_imgs_name)
        t_json_path = os.path.join(cur_dir, 'jsons', a_imgs_name.replace('png', 'json'))
        t_mask_path = os.path.join(cur_dir, 'masks', a_imgs_name)
        shutil.copy(s_img_path,t_img_path)
        shutil.copy(s_json_path, t_json_path)
        shutil.copy(s_mask_path, t_mask_path)
