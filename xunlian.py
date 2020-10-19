import os
import csv
import os
import shutil
import time
with open("video_info_new.csv",'r') as csvfile:
    read = csv.reader(csvfile)
    items = list(read)
    # print("read",read)
    # print("items",items[1][2])
old_path =  '/data/zqs/ssj/datasets/tianchi/feature/video/'
new_path = '/home/yanhao/pycharm/BMN_pgcn_val//tcdata/video/'
for i in range(1,1500):
    if items[i][2] == "validation":
        a = items[i][0]
        if os.path.exists(old_path+items[i][0]):
            shutil.copyfile(old_path+items[i][0],new_path+items[i][0])
old_path =  '/data/zqs/ssj/datasets/tianchi/feature/i3d/'
new_path = '/home/yanhao/pycharm/BMN_pgcn_val//tcdata/i3d_feature/'
for i in range(1,1500):
    if items[i][2] == "validation":
        a = items[i][0].split(".")[0]
        if os.path.exists(old_path+a+".npy"):
            shutil.copyfile(old_path+a+".npy",new_path+a+".npy")





