import numpy as np


def depth2space(x, block_size, channels_last=True):

    x_dim = x.ndim
    if x_dim == 3:
        x = np.expand_dims(x, axis=0)

    if channels_last:
        # NHWC格式
        n, h, w, c = x.shape

        if c % (block_size * block_size) != 0:
            raise ValueError("channel should bu divisible by {} * {}".format(block_size, block_size))


        h_new = h * block_size
        w_new = w * block_size
        c_new = c // (block_size * block_size)

        x_reshaped = x.reshape(n, h, w, block_size, block_size, c_new)
        x_transposed = x_reshaped.transpose(0, 1, 3, 2, 4, 5)
        return x_transposed.reshape(n, h_new, w_new, c_new)  if x_dim == 4 else x_transposed.reshape(h_new, w_new, c_new)

    else:

        n, c, h, w = x.shape

        # 验证通道数
        if c % (block_size * block_size) != 0:
            raise ValueError("channel should bu divisible by {} * {}".format(block_size, block_size))

        h_new = h * block_size
        w_new = w * block_size
        c_new = c // (block_size * block_size)

        x_reshaped = x.reshape(n, block_size, block_size, c_new, h, w)
        x_transposed = x_reshaped.transpose(0, 3, 4, 1, 5, 2)
        return x_transposed.reshape(n, c_new, h_new, w_new)  if x_dim == 4 else x_transposed.reshape(h_new, w_new, c_new)


def space2depth(x, block_size, channels_last=True):
    if x.ndim == 2:
        x = np.expand_dims(x, axis=-1) if channels_last else np.expand_dims(x, axis=0)
    x_dim = x.ndim
    if x_dim == 3:
        x = np.expand_dims(x, axis=0)

    if channels_last:
        # NHWC
        n, h, w, c = x.shape

        # 验证尺寸
        if h % block_size != 0 or w % block_size != 0:
            raise ValueError("h and w should be divisible by {}".format(block_size))

        # 计算新形状
        h_new = h // block_size
        w_new = w // block_size
        c_new = c * block_size * block_size

        x_reshaped = x.reshape(n, h_new, block_size, w_new, block_size, c)
        x_transposed = x_reshaped.transpose(0, 1, 3, 2, 4, 5)
        return x_transposed.reshape(n, h_new, w_new, c_new) if x_dim == 4 else x_transposed.reshape(h_new, w_new, c_new)

    else:
        n, c, h, w = x.shape

        # 验证尺寸
        if h % block_size != 0 or w % block_size != 0:
            raise ValueError("h and w should be divisible by {}".format(block_size))

        h_new = h // block_size
        w_new = w // block_size
        c_new = c * block_size * block_size

        x_reshaped = x.reshape(n, c, h_new, block_size, w_new, block_size)
        x_transposed = x_reshaped.transpose(0, 3, 5, 1, 2, 4)
        return x_transposed.reshape(n, c_new, h_new, w_new) if x_dim == 4 else x_transposed.reshape(h_new, w_new, c_new)


def crop_image(x, sr, er, sc, ec):
    return x[sr:er, sc:ec]