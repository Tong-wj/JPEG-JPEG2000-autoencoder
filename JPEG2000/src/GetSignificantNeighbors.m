function [y,h,v,d]=GetSignificantNeighbors(m,width,height,i,j)
%% function GetSignificantNeighbors.m
% Description:
% �ú������ڻ�ȡ����λ������Ӧ���ھӵ���Ҫ����Ϣ
% m�����ݼ�
% width�����ݼ��Ŀ�ȣ�������
% height�����ݼ��ĸ߶ȣ�������
% i,j���������λ������
%%
y=0;v=0;h=0;d=0;
if i>1 && m(i-1,j)
	v=v+1;
	y=1;
end
if i < height-1 && m(i+1,j)
	v=v+1;
	y=1;
end

if j>1 && m(i,j-1)
	h=h+1;
	y=1;
end
if j<width-1 && m(i,j+1) 
	h=h+1;
	y=1;
end
if i>1 && j>1 && m(i-1,j-1)
	d=d+1;
	y=1;
end
if i>1 && j<width-1 && m(i-1,j+1)
	d=d+1;
	y=1;
end
if i<height-1 && j>1 && m(i+1,j-1)
	d=d+1;
	y=1;
end
if i<height-1 && j<width-1 && m(i+1,j+1)
	d=d+1;
	y=1;
end
end