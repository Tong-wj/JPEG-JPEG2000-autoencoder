% 数字图像处理：lena512图像的JPEG压缩
% by Tong, Beijing
clear;close all;clc;


Q = 20;
if Q <= 50
    quality = 5000 / Q;
else
    quality = 200 - Q * 2;
end
%% Image Segmentation
% lena512: 512*512
% Block: 32768*8
load('lena512.mat','lena512');
img = double(imread('lena512.bmp'));

Block=[];
for numi=1:64 %逐行取方阵
    m=(numi-1)*8+1; %每块行的开头
    for numj=1:64 %逐列取方阵
        n=(numj-1)*8+1; %每块列的开头
        Block=[Block; img(m:m+7,n:n+7)];
    end
end
 
%% DCT
% Block: 32768*8
% FBlock: 32768*8
for num=1:4096
    start=(num-1)*8+1;
    FBlock(start:start+7,:)=dct2(Block(start:start+7,:));
end
 
%% Quantization
% FBlock: 32768*8
% QBlock: 32768*8

% load('lighttable.mat','lighttable');
load('lighttable.mat')
lighttable = floor((lighttable * quality + 50) / 100);
for i = 1:length(lighttable)
    for j = length(lighttable)
        if lighttable(i,j) <= 0
            lighttable(i,j) = 1;
        elseif lighttable(i,j) > 255
            lighttable(i,j) = 255;
        end
    end
end
for num=1:4096
    start=(num-1)*8+1;
    QBlock(start:start+7,:)=round(FBlock(start:start+7,:)./lighttable);
end
 
%% Inverse Quantization
% QBlock: 32768*8
% reFBlock: 32768*8
for num=1:4096
    start=(num-1)*8+1;
    reFBlock(start:start+7,:)=QBlock(start:start+7,:).*lighttable;
end
 
%% IDCT
for num=1:4096
    start=(num-1)*8+1;
    Block(start:start+7,:)=idct2(reFBlock(start:start+7,:));
end
 
%% Image Reconstrucion
reimage=[];
for numi=1:64
    m=(numi-1)*512+1;
    % 分成64个512*8阵列，每个阵列有64个8*8方阵
    A=[];
    for numj=1:64
        n=(numj-1)*8;
        A=[A Block(m+n:m+n+7,:)];
    end
    reimage=[reimage; A];
end
 
%% JPEG & JPEG2000 Figure
figure();
subplot(1,2,1);
imshow(img./256);
xlabel('Origin');
subplot(1,2,2);
imshow(reimage./256);
xlabel('JPEG');
suptitle('Origin vs. JPEG');
 
figure();
subplot(1,3,1);
imshow(reimage./256);
xlabel('JPEG');
 

% lena_bmp = imread('lena512.bmp');
imwrite(uint8(img),'img_j2k.j2k','CompressionRatio',10.4);
subplot(1,3,2);
imshow('img_j2k.j2k')
xlabel('JPEG2000');

imwrite(uint8(img),'img_jpg.jpg','Quality',Q);
subplot(1,3,3);
imshow('img_jpg.jpg')
xlabel('JPEG');
suptitle('JPEG vs. JPEG2000');

%% PSNR
delta=img-reimage;
delta=delta.^2;
MSE=sum(delta(:))/512/512;
PSNR=10*log10(255^2/MSE);
disp(['PSNR_JPEG:               ',num2str(PSNR)]);
 
img_j2k=imread('img_j2k.j2k');
delta=img-double(img_j2k);
delta=delta.^2;
MSE=sum(delta(:))/512/512;
PSNR=10*log10(255^2/MSE);
disp(['PSNR_JPEG2000:       ',num2str(PSNR)]);
 
%% CODE
% 以下只计算编码长度，不实际存储编码。
 
%% ZIG-ZAG
% QBlock: 32768*8
% QLine: 4096*64
QLine=[];
load('zigzag.mat','zigzag');
zigzag = zigzag(:);
 
for num=1:4096
    start=(num-1)*8+1;
    A=reshape(QBlock(start:start+7,:),1,64);% 变成行向量
    QLine=[QLine;A(zigzag)];
end
 
%% DPCM for DC
% QLine: 4096*64
% VLIDC: 4096*1     VLI 变长整数编码
% 对第一列进行DPCM编码，第一个值记为DC，并赋0
DC=QLine(1,1);%保留备用
sumcode=0;%计算编码长度
 
QLine(1,1)=0;
for num=4096:-1:2
    QLine(num,1)=QLine(num,1)-QLine(num-1,1);
end
 
VLIDC=ones(4096,1);% VLI分组
for num=1:4096
    temp=abs(QLine(num,1));%用绝对值判断组别
    if temp==0
        VLIDC(num)=0;
    else
        for k=1:7%经测试，第一列最大值为80，前7组够用
            if (temp>=2^(k-1)) && (temp<2^k)
                VLIDC(num)=k;
                break;
            end
        end
    end
end
 
for num=1:4096
    %先根据DC亮度huffman表计算sumcode
    if (VLIDC(num)<=5) && (VLIDC(num)>=0)
        sumcode=sumcode+3;
    elseif VLIDC(num)==6
        sumcode=sumcode+4;
    else
        sumcode=sumcode+5;
    end
    
    %再根据VLI表计算sumcode
    sumcode=sumcode+VLIDC(num);
end
%DC计算结果为24096，比8*4096=32768小得多。
 
%% RLC for AC
% QLine: 4096*64
% 经测试，后63列最大值为58，VLI前6组够用。
eob=max(QLine(:))+1; %设一个超大值作为每一行结束符

for numn=1:4096 %放eob
    for numm=64:-1:2
        if QLine(numn,numm)~=0
            QLine(numn,numm+1)=eob;
            break;
        end
        if (numm==2)%没找到
            QLine(numn,2)=eob;
        end
    end
end
test=QLine';
[col,~]=find(test==eob);%我们只要eob列位置
validAC=col-1; %每一行保留的AC数据量，含EOB
 
for numn=1:4096 %逐行计算并加至sumcode
    cz=[];%记录前0数
    VLIAC=[];%记录组号
    count=0;%记录连0数
    for numm=2:1+validAC(numn)
        if QLine(numn,numm)==eob
            cz=[cz 0];
            VLIAC=[VLIAC 0];
        elseif QLine(numn,numm)==0
            count=count+1;
        else %遇到非0值
            if count>15 %遇到连0大于15的
                cz=[cz 15];
                count=0;
                VLIAC=[VLIAC 0];
                continue;
            end
            cz=[cz count];
            count=0;
            
            temp=abs(QLine(numn,numm));%用绝对值判断组别
            for k=1:6%经测试，后63列最大值为58，前6组够用
                if (temp>=2^(k-1)) && (temp<2^k)
                    VLIAC=[VLIAC k];
                    break;
                end
            end
        end
    end%该行cz和VLIAC已定，开始计算
    
    sumcode=sumcode+4;%EOB对应1010，就是4bit
    czlen=length(cz)-1; %czlen不包括EOB
    load('codelength.mat');
    for k=1:czlen
        if VLIAC(k)==0
            sumcode=sumcode+11;
        else
            sumcode=sumcode+codelength(cz(k)+1,VLIAC(k));
        end
    end 
end
%% Compression Rate
OB=512*512*8;
CR=OB/sumcode;
PD=sumcode/512/512;
disp(['Original Bit:               ',num2str(OB),' bit']);
disp(['Compressed Bit:       ',num2str(sumcode),' bit']);
disp(['Compression Ratios: ',num2str(CR)]);
disp(['Pixel Depth:              ',num2str(PD),' bpp']);
disp('                                   ——Calculated by Tong');

img_jpg = double(imread('img_jpg.jpg'));
PSRN_JPEG = psnr(uint8(img_jpg),uint8(img));
disp(['PSNR_JPEG(jpg):               ',num2str(PSRN_JPEG)]);

delta=img_jpg-img;
delta=delta.^2;
MSE=sum(delta(:))/512/512;
PSRN_JPEG=10*log10(255^2/MSE);
disp(['PSNR_JPEG(jpg):               ',num2str(PSRN_JPEG)]);

%% 读取二进制的jpg文件，获得JPEG压缩后的大小
fid = fopen('img_jpg.jpg','rb');
[A, count] = fread(fid);
disp(['count:               ',num2str(count*8)]);

fid = fopen('img_j2k.j2k','rb');
[A_j2k, count_j2k] = fread(fid);