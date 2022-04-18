import tensorflow as tf


def preprocess_image_l(image):
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, [300, 300])
    image /= 255.0
    return image


def load_and_preprocess_image_l(path):
    image = tf.io.read_file(path)
    return preprocess_image_l(image)


def preprocess_image_h(image):
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, [600, 600])
    image /= 255.0
    return image


def load_and_preprocess_image_h(path):
    image = tf.io.read_file(path)
    return preprocess_image_h(image)