function y=SaveBlock(widthBlock,heightBlock,level,subband,Block)
%% function SaveBlock.m
% Descritption��
% �ú������ڱ������ݿ����Ϣ������һ���ṹ��
% widthBlock�����ݿ�Ŀ��
% heightBlock�����ݿ�ĸ߶�
% level�����ݿ�ļ�����Ϣ
% subband�����ݿ���Ӵ���Ϣ
% Block�����ݿ����ֵ
%%
y.width=widthBlock;
y.height=heightBlock;
y.level=level;
y.subband=subband;
y.data=Block;
end