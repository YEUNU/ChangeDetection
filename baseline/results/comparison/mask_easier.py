import cv2
import os
import sys
import numpy as np
from tqdm import tqdm
raw_path = os.path.join(os.getcwd(),"mask")

new_path = os.path.join(os.getcwd(),"new_mask")

os.makedirs(new_path, exist_ok=True)

raw_list = os.listdir(raw_path)

for i in tqdm(range(len(raw_list))):
    data = cv2.imread(os.path.join(raw_path,raw_list[i]),cv2.IMREAD_GRAYSCALE)
    sep1,sep2,sep3 = data, data, data
    sep1 = np.where(sep1 == 1, 255, 0)
    sep2 = np.where(sep2 == 2, 255, 0)
    sep3 = np.where(sep3 == 3, 255, 0)

    new_data = np.dstack([sep1,sep3,sep2])

    cv2.imwrite(os.path.join(new_path,raw_list[i]),new_data)


