# JPEG-JPEG2000-autoencoder
静止图像压缩编码——传统方法和深度学习方法对比

## JPEG_gray
MATLAB实现，只针对灰度图像进行JPEG压缩，没有进行熵编码，只做理论上的压缩率计算

## JPEG2000
MATLAB实现，详见JPEG2000的README

## CAE
Python实现，一种典型的自动编码器实现图像压缩，训练数据集选用STL10

## 不提供分类网络
分类网络采用的是简单的Resnet18实现，有需要的话，可以自己找经典的分类代码。数据集同样选用STL10
