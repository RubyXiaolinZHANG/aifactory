

def normalization(x, src_min, src_max, dst_min=0, dst_max=1.0):
    return (x - src_min) / (src_max - src_min) * (dst_max - dst_min) + dst_min