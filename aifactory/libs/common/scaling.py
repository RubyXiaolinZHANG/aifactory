

def normalization(x, src_min, src_max, dst_min, dst_max):
    return (x - src_min) / (src_max - src_min) * (dst_max - dst_min) + dst_min


def normalization_span(x, src_center, src_span, dst_center, dst_span):
    return normalization(x, src_center - src_span/2, src_center + src_span/2,
                         dst_center - dst_span/2, dst_center + dst_span/2)

