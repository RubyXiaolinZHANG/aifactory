import numpy as np


def roughness(std_val):
    return  1 - 1 / (1 + std_val**2)


def roughness_norm(std_val, bits):
    return 1 - 1 / (1 + (std_val/((1 << bits) - 1)) ** 2)


def histogram(x, edge, axis=None):
    if axis is not None:
        histogram = []
        c = x.shape[axis]
        for i in range(c):
            hist, edges = np.histogram(x[:, :, i].reshape(-1), edge)
            pdf = hist / hist.sum()
            mean_val = (edges[:-1] * pdf).sum()
            moment2 = ((edges[:-1] - mean_val) ** 2 * pdf).sum()
            std_val = np.sqrt(moment2)
            moment3 = ((edges[:-1] - mean_val) ** 3 * pdf).sum()
            p = pdf(pdf > 0)
            entropy = -np.sum(p * np.log2(p))
            histogram.append({"hist": hist,
                              "edges": edges,
                              "pdf": pdf,
                              "cdf": np.cumsum(pdf),
                              "mean": mean_val,
                              "std": std_val,
                              "roughness": roughness(std_val),
                              "moment2": moment2,
                              "moment3": moment3,
                              "entropy": entropy})
    else:
        hist, edges = np.histogram(x.reshape(-1), edge)
        pdf = hist / hist.sum()
        mean_val = (edges[:-1] * pdf).sum()
        moment2 = ((edges[:-1] - mean_val) ** 2 * pdf).sum()
        std_val = np.sqrt(moment2)
        moment3 = ((edges[:-1] - mean_val) ** 3 * pdf).sum()
        p = pdf[pdf > 0]
        entropy = -np.sum(p * np.log2(p))
        histogram = {"hist": hist,
                     "edges": edges,
                     "pdf": pdf,
                     "cdf": np.cumsum(pdf),
                     "mean": mean_val,
                     "std": std_val,
                     "roughness": roughness(std_val),
                     "moment2": moment2,
                     "moment3": moment3,
                     "entropy": entropy}
    return histogram
