function [a_new]=EBCODE_main(tile)
%% function EBCODE_main
% Description:
% ����������ڶ�����������ݽ���EBCOT���������룬Huffman����������
% tile���ṹ�飬���а����ֿ�������Լ��������Ӵ�����Ϣ
% a_new�������֮��õ��Ļָ����ݷֿ飬ΪԪ������ṹ��ÿ��Ԫ�ش洢һ���ֿ�
%%
a=tile;
for i=1:length(a)
    for j=1:size(a{i}.data,3)
        % ��ʼ���洢�������ļ�
        x.output=0;
        x.input=0;
        x.pos=8;
        x.quedan=0;
        x.file=0;                                       % �洢�����ĵط�
        b=a{i}.data(:,:,j);                             % ��ÿ��ͨ�����б���
        [a_quantization,step]=quantization(b,a{i}.sub); % �����������ز�����Ϣ
        datum.data=a_quantization;
        datum.width=size(datum.data,2);
        datum.height=size(datum.data,1);
        datum.level=a{i}.level;                         % ��ȡ������Ϣ
        datum.subband=a{i}.sub;                         % ��ȡ�Ӵ���Ϣ
        [FLength,FCtxt,x_new]=EBCOT(datum,1,x);         % EBCOT����
        text=num2str(x_new.file);                       % ������ת��Ϊ�ַ���
        [code,efficent,codetable,time1]=Huffmanencode(text,1);  % Huffman����
        [decode,time2] = Huffmandecode(codetable,code);         % Huffman�����
        decode=str2num(decode);                         % ���������ַ�����ת��Ϊ��������
        x_new.file=decode;
        [datum_new,x_new]=DeEBCOT(1,x_new,FCtxt,FLength);       % EBCOT�����
        datum_dequantization=dequantization(datum_new{1}.data,step);    % ������
        b_new(:,:,j)=datum_dequantization;
    end
    a_new{i}=b_new;
end
end

