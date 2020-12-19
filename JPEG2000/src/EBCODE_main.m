function [a_new]=EBCODE_main(tile)
%% function EBCODE_main
% Description:
% 这个函数用于对量化后的数据进行EBCOT编码与解编码，Huffman编码与解编码
% tile：结构块，其中包含分块的数据以及级数，子带等信息
% a_new：解编码之后得到的恢复数据分块，为元胞数组结构，每个元素存储一个分块
%%
a=tile;
for i=1:length(a)
    for j=1:size(a{i}.data,3)
        % 初始化存储码流的文件
        x.output=0;
        x.input=0;
        x.pos=8;
        x.quedan=0;
        x.file=0;                                       % 存储码流的地方
        b=a{i}.data(:,:,j);                             % 对每个通道进行编码
        [a_quantization,step]=quantization(b,a{i}.sub); % 量化，并返回步长信息
        datum.data=a_quantization;
        datum.width=size(datum.data,2);
        datum.height=size(datum.data,1);
        datum.level=a{i}.level;                         % 获取级数信息
        datum.subband=a{i}.sub;                         % 获取子带信息
        [FLength,FCtxt,x_new]=EBCOT(datum,1,x);         % EBCOT编码
        text=num2str(x_new.file);                       % 将码流转换为字符串
        [code,efficent,codetable,time1]=Huffmanencode(text,1);  % Huffman编码
        [decode,time2] = Huffmandecode(codetable,code);         % Huffman解编码
        decode=str2num(decode);                         % 将解码后的字符串流转换为整型数组
        x_new.file=decode;
        [datum_new,x_new]=DeEBCOT(1,x_new,FCtxt,FLength);       % EBCOT解编码
        datum_dequantization=dequantization(datum_new{1}.data,step);    % 反量化
        b_new(:,:,j)=datum_dequantization;
    end
    a_new{i}=b_new;
end
end

