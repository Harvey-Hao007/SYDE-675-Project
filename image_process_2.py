import numpy as np
import os
import tensorflow as tf
import cv2

from model import get_model
from image_functions import load_and_preprocess_image_l

test_low_dir = "train-set/valid/LR"
hr_dir = "train-set/valid/HR"
hr_hs_out = "train-set-3/valid/HR"
hr_ls_out = "train-set-3/valid/LR"

BATCH_SIZE = 8

test_low_paths = os.listdir(test_low_dir)
test_low_paths.sort()
test_high_paths = os.listdir(hr_dir)
test_high_paths.sort()

for i in range(len(test_low_paths)):
    test_low_paths[i] = os.path.join(test_low_dir, test_low_paths[i])
test_path_ds = tf.data.Dataset.from_tensor_slices(test_low_paths)
test_ds = test_path_ds.map(load_and_preprocess_image_l, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
hr_hs_paths = []
hr_ls_paths = []
for i in range(len(test_high_paths)):
    temp = test_high_paths[i]
    test_high_paths[i] = os.path.join(hr_dir, temp)
    hr_hs_paths.append(os.path.join(hr_hs_out, temp))
    hr_ls_paths.append(os.path.join(hr_ls_out, temp))

model = get_model()
checkpoint_filepath = "checkpoint.ckpt"
model.load_weights(checkpoint_filepath)

prediction = model.predict(test_ds)

for i in range(len(test_high_paths)):
    tmp_img = prediction[i] * 255.0
    pred_img = np.zeros(tmp_img.shape)
    pred_img[:, :, 0] = tmp_img[:, :, 2]
    pred_img[:, :, 1] = tmp_img[:, :, 1]
    pred_img[:, :, 2] = tmp_img[:, :, 0]
    hr_image = cv2.imread(test_high_paths[i])
    pred_img = cv2.resize(pred_img, (hr_image.shape[1], hr_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    cropped_hs = np.zeros(hr_image.shape)
    n = 0
    for i1 in range(hr_image.shape[0]):
        for j1 in range(hr_image.shape[1]):
            for k1 in range(hr_image.shape[2]):
                if abs(pred_img[i1, j1, k1] - hr_image[i1, j1, k1]) > 10:
                    cropped_hs[i1, j1, k1] = hr_image[i1, j1, k1]
                    n += 1
    cv2.imwrite(hr_hs_paths[i], cropped_hs)
    cropped_ls = cv2.resize(cropped_hs, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(hr_ls_paths[i], cropped_ls)

# process train set part 1
test_low_dir = "train-set/train/LR"
hr_dir = "train-set/train/HR"
hr_hs_out = "train-set-3/train/HR"
hr_ls_out = "train-set-3/train/LR"

test_low_paths = os.listdir(test_low_dir)
test_low_paths.sort()
test_high_paths = os.listdir(hr_dir)
test_high_paths.sort()
test_low_paths = test_low_paths[0:400]
test_high_paths = test_high_paths[0:400]

# test_low_paths = test_low_paths[400:]
# test_high_paths = test_high_paths[400:]

for i in range(len(test_low_paths)):
    test_low_paths[i] = os.path.join(test_low_dir, test_low_paths[i])
test_path_ds = tf.data.Dataset.from_tensor_slices(test_low_paths)
test_ds = test_path_ds.map(load_and_preprocess_image_l, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
hr_hs_paths = []
hr_ls_paths = []
for i in range(len(test_high_paths)):
    temp = test_high_paths[i]
    test_high_paths[i] = os.path.join(hr_dir, temp)
    hr_hs_paths.append(os.path.join(hr_hs_out, temp))
    hr_ls_paths.append(os.path.join(hr_ls_out, temp))

prediction = model.predict(test_ds)

for i in range(len(test_high_paths)):
    tmp_img = prediction[i] * 255.0
    pred_img = np.zeros(tmp_img.shape)
    pred_img[:, :, 0] = tmp_img[:, :, 2]
    pred_img[:, :, 1] = tmp_img[:, :, 1]
    pred_img[:, :, 2] = tmp_img[:, :, 0]
    hr_image = cv2.imread(test_high_paths[i])
    pred_img = cv2.resize(pred_img, (hr_image.shape[1], hr_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    cropped_hs = np.zeros(hr_image.shape)
    n = 0
    for i1 in range(hr_image.shape[0]):
        for j1 in range(hr_image.shape[1]):
            for k1 in range(hr_image.shape[2]):
                if abs(pred_img[i1, j1, k1] - hr_image[i1, j1, k1]) > 10:
                    cropped_hs[i1, j1, k1] = hr_image[i1, j1, k1]
                    n += 1
    cv2.imwrite(hr_hs_paths[i], cropped_hs)
    print(i, n, (hr_image.shape[0] * hr_image.shape[1] * hr_image.shape[2]))
    cropped_ls = cv2.resize(cropped_hs, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(hr_ls_paths[i], cropped_ls)

# process train set part 2
test_low_paths = os.listdir(test_low_dir)
test_low_paths.sort()
test_high_paths = os.listdir(hr_dir)
test_high_paths.sort()

test_low_paths = test_low_paths[400:]
test_high_paths = test_high_paths[400:]

for i in range(len(test_low_paths)):
    test_low_paths[i] = os.path.join(test_low_dir, test_low_paths[i])
test_path_ds = tf.data.Dataset.from_tensor_slices(test_low_paths)
test_ds = test_path_ds.map(load_and_preprocess_image_l, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
hr_hs_paths = []
hr_ls_paths = []
for i in range(len(test_high_paths)):
    temp = test_high_paths[i]
    test_high_paths[i] = os.path.join(hr_dir, temp)
    hr_hs_paths.append(os.path.join(hr_hs_out, temp))
    hr_ls_paths.append(os.path.join(hr_ls_out, temp))

prediction = model.predict(test_ds)

for i in range(len(test_high_paths)):
    tmp_img = prediction[i] * 255.0
    pred_img = np.zeros(tmp_img.shape)
    pred_img[:, :, 0] = tmp_img[:, :, 2]
    pred_img[:, :, 1] = tmp_img[:, :, 1]
    pred_img[:, :, 2] = tmp_img[:, :, 0]
    hr_image = cv2.imread(test_high_paths[i])
    pred_img = cv2.resize(pred_img, (hr_image.shape[1], hr_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    cropped_hs = np.zeros(hr_image.shape)
    n = 0
    for i1 in range(hr_image.shape[0]):
        for j1 in range(hr_image.shape[1]):
            for k1 in range(hr_image.shape[2]):
                if abs(pred_img[i1, j1, k1] - hr_image[i1, j1, k1]) > 10:
                    cropped_hs[i1, j1, k1] = hr_image[i1, j1, k1]
                    n += 1
    cv2.imwrite(hr_hs_paths[i], cropped_hs)
    print(i, n, (hr_image.shape[0] * hr_image.shape[1] * hr_image.shape[2]))
    cropped_ls = cv2.resize(cropped_hs, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(hr_ls_paths[i], cropped_ls)
