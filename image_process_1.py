import cv2
import os
import numpy as np

lr_dir = "train-set/valid/LR"
hr_dir = "train-set/valid/HR"
hr_hs_out = "train-set-2/valid/HR"
hr_ls_out = "train-set-2/valid/LR"

lr_paths = os.listdir(lr_dir)
lr_paths.sort()
for i in range(len(lr_paths)):
    lr_paths[i] = os.path.join(lr_dir, lr_paths[i])

hr_paths = os.listdir(hr_dir)
hr_paths.sort()
hr_hs_paths = []
hr_ls_paths = []
for i in range(len(hr_paths)):
    temp = hr_paths[i]
    hr_paths[i] = os.path.join(hr_dir, temp)
    hr_hs_paths.append(os.path.join(hr_hs_out, temp))
    hr_ls_paths.append(os.path.join(hr_ls_out, temp))

for i in range(len(hr_paths)):
    print(1, i)
    lr_image = cv2.imread(lr_paths[i])
    hr_image = cv2.imread(hr_paths[i])
    lr_double = cv2.resize(lr_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    diff = lr_double - hr_image
    cropped_hs = np.zeros(hr_image.shape)
    cropped_ls = np.zeros(hr_image.shape)
    for i1 in range(hr_image.shape[0]):
        for j1 in range(hr_image.shape[1]):
            for k1 in range(hr_image.shape[2]):
                if lr_double[i1, j1, k1] != hr_image[i1, j1, k1]:
                    cropped_hs[i1, j1, k1] = hr_image[i1, j1, k1]
                    cropped_ls[i1, j1, k1] = lr_double[i1, j1, k1]
    cv2.imwrite(hr_hs_paths[i], cropped_hs)
    cropped_ls = cv2.resize(cropped_ls, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(hr_ls_paths[i], cropped_ls)

lr_dir = "train-set/train/LR"
hr_dir = "train-set/train/HR"
hr_hs_out = "train-set-2/train/HR"
hr_ls_out = "train-set-2/train/LR"

lr_paths = os.listdir(lr_dir)
lr_paths.sort()
for i in range(len(lr_paths)):
    lr_paths[i] = os.path.join(lr_dir, lr_paths[i])

hr_paths = os.listdir(hr_dir)
hr_paths.sort()
hr_hs_paths = []
hr_ls_paths = []
for i in range(len(hr_paths)):
    temp = hr_paths[i]
    hr_paths[i] = os.path.join(hr_dir, temp)
    hr_hs_paths.append(os.path.join(hr_hs_out, temp))
    hr_ls_paths.append(os.path.join(hr_ls_out, temp))

for i in range(len(hr_paths)):
    print(2, i)
    lr_image = cv2.imread(lr_paths[i])
    hr_image = cv2.imread(hr_paths[i])
    lr_double = cv2.resize(lr_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    diff = lr_double - hr_image
    cropped_hs = np.zeros(hr_image.shape)
    for i1 in range(hr_image.shape[0]):
        for j1 in range(hr_image.shape[1]):
            for k1 in range(hr_image.shape[2]):
                if lr_double[i1, j1, k1] != hr_image[i1, j1, k1]:
                    cropped_hs[i1, j1, k1] = hr_image[i1, j1, k1]
    cv2.imwrite(hr_hs_paths[i], cropped_hs)
    cropped_ls = cv2.resize(cropped_hs, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(hr_ls_paths[i], cropped_ls)
