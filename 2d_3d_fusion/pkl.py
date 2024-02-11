# 两种方法都能打开
import pickle
import sys

import numpy as np
import os

sys.path.append('F:/futr3d-main')
f = open('F:/detr3d-main/output/output.pkl','rb')
data = pickle.load(f)
print(data)

# img_path = 'F:/futr3d-main/output/output.pkl'
# img_data = np.load(img_path)
# print(img_data)

