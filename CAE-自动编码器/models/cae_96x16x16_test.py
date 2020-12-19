import torch
import torch.nn as nn
import torch.nn.functional as F

# 二值量化器
class Binarizer(torch.autograd.Function):
    """
    An elementwise function that bins values
    to 0 or 1 depending on a threshold of
    0.5

    Input: a tensor with values in range(0,1)

    Returns: a tensor with binary values: 0 or 1
    based on a threshold of 0.5

    Equation(1) in paper
    """
    @staticmethod
    def forward(ctx, i):
        return (i>0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def bin_values(x):
    return Binarizer.apply(x)


# 一种上采样方法，深度到广度
class DepthToSpace(torch.nn.Module):

    """
    A class used by the decoder while regenerating the image that moves values
    from the depth dimension to the height and width dimensions (spatial)

    Input: A tensor of size [N,C,H,W]
    Returns: A tensor of size [N,C/(block_size*block_size),H*block_size,W*block_size]

    Parameters
    ----------
    block_size: An int that is greater than 2. It decide

    Extra
    -----
    https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
    """
    def __init__(self,block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)
        return x


class ResBlock(nn.Module):
    def __init__(self, ni, nh=128):
        super().__init__()

        self.conv1 = conv(ni, nh)
        self.conv2 = conv(nh, ni)
        # initilize 2nd conv with zeros to preserve variance
        # known as Fixup initialization
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


def conv(ni, nf, ks=3, stride=1, padding=1, **kwargs):
    _conv = nn.Conv2d(ni, nf, kernel_size=ks,stride=stride,padding=padding, **kwargs)
    nn.init.kaiming_normal_(_conv.weight)
    nn.init.zeros_(_conv.bias)
    return _conv


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)


relu = nn.ReLU()


class CAE(nn.Module):
    """
    This AE module will be fed 3x128x128 patches from the original image
    Shapes are (batch_size, channels, height, width)

    Latent representation: 32x32x32 bits per patch => 240KB per image (for 720p)
    """

    def __init__(self):
        super(CAE, self).__init__()

        self.encoded = None

        self.encoder = nn.Sequential(conv(3, 128, 8, 4, 2), relu,
                                    ResBlock(128), relu,
                                    conv(128, 256, 4, 2, 1), relu,
                                    ResBlock(256), relu,
                                    ResBlock(256), relu,
                                    conv(256, 128, 3, 1, 1),
                                    nn.Sigmoid(),
                                    Lambda(bin_values)
                                     )

        self.decoder = nn.Sequential(conv(128, 512, 1, 1, 0), relu,
                                     ResBlock(512), relu,
                                     ResBlock(512), relu,
                                     DepthToSpace(2),
                                     conv(128, 256), relu,
                                     ResBlock(256), relu,
                                     DepthToSpace(4),
                                     conv(16, 32), relu,
                                     conv(32, 3))
        # self.head = nn.Sequential(conv(256, 64, 3, 1, 1),
        #                            nn.Sigmoid(),
        #                            Lambda(bin_values))


        # ENCODER

        # 64x64x64
        self.e_conv_1 = nn.Sequential(
            # nn.ReflectionPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(8, 8), stride=(4, 4), padding=2),
            # nn.LeakyReLU(),
            nn.ReLU(),
        )

        # 128x32x32
        self.e_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

        # 128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU(),
        )

        # 128x32x32
        self.e_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

        # 128x32x32
        self.e_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

        # 96x16x16
        self.e_conv_3 = nn.Sequential(
            # nn.ReflectionPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Sigmoid()
        )

        # DECODER

        # 128x32x32
        self.d_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.ZeroPad2d((1, 0, 1, 0)),         # 改(1, 1, 1, 1)->(1, 0, 1, 0)
            # nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(4, 4), stride=(2, 2))  # 改(2,2)->(4,4)
        )

        # 128x32x32
        self.d_block_1 = nn.Sequential(
            # nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

        # 128x32x32
        self.d_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

        # 128x32x32
        self.d_block_3 = nn.Sequential(

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

        self.d_up1 = nn.Sequential(
            DepthToSpace(2),
        )

        self.d_up2 = nn.Sequential(
            DepthToSpace(4),
        )

        # 256x64x64
        self.d_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),   # 改(3,3)->(2,2)
            # nn.LeakyReLU(),
            nn.ReLU(),
            # nn.PixelShuffle(2),

            # nn.ZeroPad2d((0, 1, 0, 1)),             # 改(1, 1, 1, 1)->(1, 0, 1, 0)
            # nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),    # 改(2,2)->(4,4)
            # nn.LeakyReLU(),
        )

        # 3x128x128
        self.d_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),      # 改(3,3)->(2,2)
            # nn.LeakyReLU(),
            nn.ReLU(),
            # nn.PixelShuffle(2),
            # DepthToSpace(2),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x):
        # ec1 = self.e_conv_1(x)
        # eblock1 = self.e_block_1(ec1) + ec1
        # ec2 = self.e_conv_2(eblock1)
        # eblock2 = self.e_block_2(ec2) + ec2
        # eblock3 = self.e_block_3(eblock2) + eblock2
        # ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from tanh activation

        return self.decoder(self.encoder(x))

    def decode(self, encoded):

        # y = encoded * 2.0 - 1  # (0|1) -> (-1|1)
        y = encoded
        c1 = self.d_conv_1(y)
        dblock1 = self.d_block_1(c1) + c1
        dblock2 = self.d_block_2(dblock1) + dblock1
        u1 = self.d_up1(dblock2)
        c2 = self.d_conv_2(u1)
        dblock3 = self.d_block_3(c2) + c2
        u2 = self.d_up2(dblock3)
        dec = self.d_conv_3(u2)
        # dec = dec*128
        # # 假定三通道符合RGB的顺序，进行去归一化
        # dec[0] = dec[0] * std[0] + mean[0]
        # dec[1] = dec[1] * std[1] + mean[1]
        # dec[2] = dec[2] * std[2] + mean[2]

        # denormalize
        # with torch.no_grad():
        #     mean = [0.4431991, 0.42826223, 0.39535823]
        #     std = [0.25746644, 0.25306803, 0.26591763]
        #     dec[-1:, 0] = dec[-1:, 0] * std[0] + mean[0]
        #     dec[-1:, 1] = dec[-1:, 1] * std[1] + mean[1]
        #     dec[-1:, 2] = dec[-1:, 2].mul(std[2]) + mean[2]

        return dec
