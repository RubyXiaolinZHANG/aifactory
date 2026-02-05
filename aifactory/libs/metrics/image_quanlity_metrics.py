import torch


def psnr(x, x_hat, maximum, dim=None):
    assert x.shape == x_hat.shape
    if dim is None:
        dim = [ i for i in range(1, x.dim())]
    mse = ((x.to(torch.float32) - x_hat.to(torch.float32)) ** 2).mean(dim=[1,2,3])
    return 10 * torch.log10(maximum **2 / mse)


def channel_l1(x, x_hat, dim=None):

    assert x.shape == x_hat.shape
    if dim is None:
        dim = [ i for i in range(2, x.dim())]
    return (x.to(torch.float32) - x_hat.to(torch.float32)).abs().mean(dim=dim)


def pixel_l1(x, x_hat, dim=None):

    assert x.shape == x_hat.shape
    if dim is None:
        dim = [ i for i in range(1, x.dim())]
    return (x.to(torch.float32) - x_hat.to(torch.float32)).abs().mean(dim=dim)


class PSNR(torch.nn.Module):

    def __init__(self, maximum=255, dim=None):
        super().__init__()
        self._maximum = maximum
        self._dim = dim

    def forward(self, output, target):
        return psnr(output, target, self._maximum)



class ChannelL1(torch.nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self._dim = dim


    def forward(self, output, target):
        return channel_l1(output, target, dim=self._dim)


class PixelL1(torch.nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self._dim = dim


    def forward(self, output, target):
        return pixel_l1(output, target, dim=self._dim)