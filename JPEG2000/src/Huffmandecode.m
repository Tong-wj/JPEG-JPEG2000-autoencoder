function [decode,time] = Huffmandecode(codetable,code)
%% function Huffmandecode.m
% Description:
% 该函数用于根据编码表对序列进行Huffman解编码
% codetable：编码表
% code：编码信息序列
% decode：解码后的序列
% time：解码时间
%%
% 译码
t1=clock;
decode='';
unfinish=1;
while(unfinish)  
    flag=0;
    % 循环取更长的码 直到可以翻译
    for o=1:length(code)
            str=code(1:o);
            % 寻找是否存在译码
            for p=1:length(codetable)
                % 找到则译码
                if strcmp(codetable{2,p},str)
                    decode=[decode,codetable{1,p}];
                    flag=1;
                    break;
                end
            end

            if flag==1;                 % 这次的找到了 跳出for进行下一个
                if length(code)-o==0    % 判断是否完全以译码
                    unfinish=0;
                else                    % 截断后继续
                    code=code(o+1:length(code));
                end
                break;
            end
    end                                 % 找到了一个 继续下一个
end
t2=clock;
time=etime(t2,t1);
end