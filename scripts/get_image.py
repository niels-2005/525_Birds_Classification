import tensorflow as tf


def get_image(path, img_size, channels):
    """
    Loads and preprocesses an image from the given file path.

    The function:
    - Reads the image file.
    - Decodes the image as a JPEG with the specified number of color channels.
    - Resizes the image to match the given input dimensions.

    Args:
        path (str): Path to the image file.
        img_size (tuple): Target size for resizing the image (height, width).
        channels (int): Number of color channels to decode the image with.

    Returns:
        tf.Tensor: The preprocessed image tensor.
    """
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=channels)
    img = tf.image.resize(img, img_size)
    return img
