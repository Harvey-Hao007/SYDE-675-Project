import numpy as np
import os
import cv2
import tensorflow as tf
import tensorflow_datasets as tfds
from model import get_model
from image_functions import load_and_preprocess_image_l, load_and_preprocess_image_h

test_low_dir = "test-set/LR"
test_high_dir = "test-set/HR"
BATCH_SIZE = 8

# load test LR images
test_low_paths = os.listdir(test_low_dir)
test_low_paths.sort()
for i in range(len(test_low_paths)):
    test_low_paths[i] = os.path.join(test_low_dir, test_low_paths[i])
test_path_ds = tf.data.Dataset.from_tensor_slices(test_low_paths)
test_ds = test_path_ds.map(load_and_preprocess_image_l, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# load model
checkpoint_filepath = "checkpoint.ckpt"
model = get_model()
model.load_weights(checkpoint_filepath)
prediction = model.predict(test_ds)

# load HR images
test_high_paths = os.listdir(test_high_dir)
test_high_paths.sort()
result_paths = []
for i in range(len(test_high_paths)):
    temp = test_high_paths[i]
    test_high_paths[i] = os.path.join(test_high_dir, temp)
    result_paths.append(os.path.join('results', temp))
test_high_path_ds = tf.data.Dataset.from_tensor_slices(test_high_paths)
test_high_ds = test_high_path_ds.map(load_and_preprocess_image_h, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_hr_ds = tfds.as_numpy(test_high_ds)

# get PSNR
total_test_psnr = 0.0
i = 0
for hr_img in test_hr_ds:
    test_psnr = tf.image.psnr(prediction[i], hr_img, max_val=1)
    total_test_psnr += test_psnr
    i += 1
print("Avg. PSNR of reconstructions is %.4f" % (total_test_psnr / prediction.shape[0]))

# output result images
for i in range(len(test_high_paths)):
    tmp_img = prediction[i] * 255.0
    pred_img = np.zeros(tmp_img.shape)
    pred_img[:, :, 0] = tmp_img[:, :, 2]
    pred_img[:, :, 1] = tmp_img[:, :, 1]
    pred_img[:, :, 2] = tmp_img[:, :, 0]
    hr_image = cv2.imread(test_high_paths[i])
    pred_img = cv2.resize(pred_img, (hr_image.shape[1], hr_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(result_paths[i], pred_img)
