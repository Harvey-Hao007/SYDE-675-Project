import os
import tensorflow as tf

from model import get_model, get_callbacks
from image_functions import load_and_preprocess_image_l, load_and_preprocess_image_h


data_dir = "train-set-2"
train_low_dir = os.path.join(data_dir, "train/LR")
train_high_dir = os.path.join(data_dir, "train/HR")
valid_low_dir = os.path.join(data_dir, "valid/LR")
valid_high_dir = os.path.join(data_dir, "valid/HR")
test_low_dir = "test-set/LR"
test_high_dir = "test-set/HR"

train_low_paths = os.listdir(train_low_dir)
train_low_paths.sort()
for i in range(len(train_low_paths)):
    train_low_paths[i] = os.path.join(train_low_dir, train_low_paths[i])
train_low_path_ds = tf.data.Dataset.from_tensor_slices(train_low_paths)
train_low_ds = train_low_path_ds.map(load_and_preprocess_image_l, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_high_paths = os.listdir(train_high_dir)
train_high_paths.sort()
for i in range(len(train_high_paths)):
    train_high_paths[i] = os.path.join(train_high_dir, train_high_paths[i])
train_high_path_ds = tf.data.Dataset.from_tensor_slices(train_high_paths)
train_high_ds = train_high_path_ds.map(load_and_preprocess_image_h, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds = tf.data.Dataset.zip((train_low_ds, train_high_ds))
BATCH_SIZE = 8
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

valid_low_paths = os.listdir(valid_low_dir)
valid_low_paths.sort()
for i in range(len(valid_low_paths)):
    valid_low_paths[i] = os.path.join(valid_low_dir, valid_low_paths[i])
valid_low_path_ds = tf.data.Dataset.from_tensor_slices(valid_low_paths)
valid_low_ds = valid_low_path_ds.map(load_and_preprocess_image_l, num_parallel_calls=tf.data.experimental.AUTOTUNE)

valid_high_paths = os.listdir(valid_high_dir)
valid_high_paths.sort()
for i in range(len(valid_high_paths)):
    valid_high_paths[i] = os.path.join(valid_high_dir, valid_high_paths[i])
valid_high_path_ds = tf.data.Dataset.from_tensor_slices(valid_high_paths)
valid_high_ds = valid_high_path_ds.map(load_and_preprocess_image_h, num_parallel_calls=tf.data.experimental.AUTOTUNE)

valid_ds = tf.data.Dataset.zip((valid_low_ds, valid_high_ds))
valid_ds = valid_ds.batch(BATCH_SIZE)
valid_ds = valid_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_low_paths = os.listdir(test_low_dir)
test_low_paths.sort()
for i in range(len(test_low_paths)):
    test_low_paths[i] = os.path.join(test_low_dir, test_low_paths[i])
test_path_ds = tf.data.Dataset.from_tensor_slices(test_low_paths)
test_ds = test_path_ds.map(load_and_preprocess_image_l, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

model = get_model()

epochs = 100

# load weights before training when needed
# checkpoint_filepath = "checkpoint.ckpt"
# model.load_weights(checkpoint_filepath)

model.fit(
    train_ds, epochs=epochs, callbacks=get_callbacks(), validation_data=valid_ds, verbose=2
)
