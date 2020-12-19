function y=WriteBit(bit,x)
%% function WriteBit.m
% Description：
% 该函数用于向文件写入特定的比特位
% bit：待写入的比特位
% x：存储比特流的文件
% y：存储好以后的文件
% p.s.这部分代码所参考的代码中使用了十六进制数进行位运算，这在MATLAB中无法实现所以，
%     造成写入的码流与原来的二进制码流不太一样，但仍能恢复出来
%%
y=x;
nbits=1;
while nbits>y.pos
    nbits=nbits-y.pos;
    y.output=bitor(y.output,bitshift(int32(bit),int32(-nbits)));  % 这里原代码使用了十六进制进行运算
    y.file(end+1)=int32(y.output);
    y.pos=8;
    y.output=0;
end
y.pos=y.pos-nbits;
y.output=bitor(int32(y.output),bitshift(int32(bit),int32(y.pos)));% 这里原代码使用了十六进制进行运算
end