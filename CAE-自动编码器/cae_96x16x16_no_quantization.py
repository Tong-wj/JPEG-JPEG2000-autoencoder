import torch
import torch.nn as nn

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


class CAE(nn.Module):
    """
    This AE module will be fed 3x128x128 patches from the original image
    Shapes are (batch_size, channels, height, width)

    Latent representation: 32x32x32 bits per patch => 240KB per image (for 720p)
    """

    def __init__(self):
        super(CAE, self).__init__()

        self.encoded = None

        # ENCODER

        # 64x64x64
        self.e_conv_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.ReflectionPad2d((2, 1, 2, 1)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 96x16x16
        self.e_conv_3 = nn.Sequential(
            nn.ReflectionPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=(5, 5), stride=(2, 2)),
            # nn.Sigmoid()
        )

        # DECODER

        # 128x32x32
        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # 改(3,3)->(2,2)
            nn.LeakyReLU(),
            nn.PixelShuffle(2),
            # nn.LeakyReLU(),
            # nn.ZeroPad2d((1, 0, 1, 0)),         # 改(1, 1, 1, 1)->(1, 0, 1, 0)
            # nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(4, 4), stride=(2, 2))  # 改(2,2)->(4,4)
        )

        # 128x32x32
        self.d_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.d_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.d_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 256x64x64
        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),   # 改(3,3)->(2,2)
            nn.LeakyReLU(),
            nn.PixelShuffle(2),
            # DepthToSpace(2),

            # nn.ZeroPad2d((0, 1, 0, 1)),             # 改(1, 1, 1, 1)->(1, 0, 1, 0)
            # nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),    # 改(2,2)->(4,4)
            # nn.LeakyReLU(),
        )

        # 3x128x128
        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),      # 改(3,3)->(2,2)
            nn.LeakyReLU(),

            nn.PixelShuffle(2),
            # DepthToSpace(2),

            # nn.ZeroPad2d((1, 1, 1, 1)),             # 改(1, 1, 1, 1)->(1, 0, 1, 0)
            # nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=(2, 2), stride=(2, 2)),  # 改(2,2)->(4,4)
            # nn.Tanh()
        )

    def forward(self, x):
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        eblock1 = self.e_block_1(ec2) + ec2
        eblock2 = self.e_block_2(eblock1) + eblock1
        eblock3 = self.e_block_3(eblock2) + eblock2
        ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from tanh activation

        # with torch.no_grad():
        #     encode = (ec3 > 0.5).float()
        # const rounding quantization

        # with torch.no_grad():
        #     ec3 = ec3 * 1000
        #     eps = torch.zeros(ec3.shape).cuda()
        #     prob = ec3 - torch.floor(ec3)
        #     const = 0.5 * torch.ones(ec3.shape).cuda()
        #     eps[prob >= const] = torch.ceil(ec3[prob >= const])
        #     eps[prob < const] = torch.floor(ec3[prob < const])
        #     encode = eps
        #     maximum = torch.max(encode)
        #     minimum = torch.min(encode)
        #     mean_value = torch.mean(encode)

        # additive noise
        # with torch.no_grad():
        #     noise = torch.rand(ec3)
        #     ec3 = ec3 + noise

        # rounding quantization
        with torch.no_grad():
            maximum = torch.max(ec3)
            minimum = torch.min(ec3)
            mean_value = torch.mean(ec3)

            encode = torch.round(ec3)

        # rand rounding quantization
        # with torch.no_grad():
        #     eps = torch.zeros(ec3.shape).cuda()
        #     prob = ec3 - torch.floor(ec3)
        #     rand = torch.rand(ec3.shape).cuda()
        #     eps[rand <= prob] = torch.floor(ec3[rand <= prob])
        #     eps[rand > prob] = torch.ceil(ec3[rand > prob])
        #     encode = eps
        # # stochastic binarization
        # with torch.no_grad():
        #     rand = torch.rand(ec3.shape).cuda()
        #     prob = (1 + ec3) / 2
        #     eps = torch.zeros(ec3.shape).cuda()
        #     eps[rand <= prob] = (1 - ec3)[rand <= prob]
        #     eps[rand > prob] = (-ec3 - 1)[rand > prob]
        #
        # # encoded tensor
        # self.encoded = 0.5 * (ec3 + eps + 1)  # (-1|1) -> (0|1)

        return self.decode(encode)

    def decode(self, encoded):

        # y = encoded * 2.0 - 1  # (0|1) -> (-1|1)
        y = encoded
        uc1 = self.d_up_conv_1(y)
        dblock1 = self.d_block_1(uc1) + uc1
        dblock2 = self.d_block_2(dblock1) + dblock1
        dblock3 = self.d_block_3(dblock2) + dblock2
        uc2 = self.d_up_conv_2(dblock3)
        dec = self.d_up_conv_3(uc2)
        # dec = dec*128
        # # 假定三通道符合RGB的顺序，进行去归一化
        # dec[0] = dec[0] * std[0] + mean[0]
        # dec[1] = dec[1] * std[1] + mean[1]
        # dec[2] = dec[2] * std[2] + mean[2]

        # clipping
        # with torch.no_grad():
        return dec
