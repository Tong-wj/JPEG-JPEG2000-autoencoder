function [code,efficent,codetable,time] = Huffmanencode(text,type)
%% function Huffmanencode.m
% Description:
% 该函数用于对输入字符串或文本进行Huffman编码
% text：输入文本
% type：文本类型，1：英文 0：汉语
% code：编码符号
% efficent：压缩率
% codetable：码表
% time：耗时
%%
t1=clock;

Hztar=' ';
Hznum=0;
% 读取文本
target=text;
for i=1:length(target);
    if target(i)==' '
       Hznum(1)=Hznum(1)+1;
    elseif strfind(Hztar,target(i))>0
        position=strfind(Hztar,target(i));
        Hznum(position)=Hznum(position)+1;
    else
        Hztar=[Hztar,target(i)];
        Hznum=[Hznum;1];
    end   
end

% 符号数
numstar=length(Hznum);
% 编码表
codetable=cell(2,numstar);
% 保存好对应的符号与个数
oritable=cell(2,numstar);
for i=1:numstar
    codetable{1,i}=Hztar(i);
    codetable{2,i}='';
    oritable{1,i}=Hztar(i);
    oritable{2,i}=Hznum(i);
end
% 根据是否只剩下最后两个判断是否完成编码
while (length(find(Hznum==100000))<=numstar-2)
    % 编码选择符号 两个
    select=[0,0];
    num=[0,0];                          % 频数
    posi=-1;                            % 记录位置
    tempp=find(Hznum==min(Hznum));      % 找第一个最小
    select(1)=tempp(1);
    numz(1)=Hznum(tempp(1));
    Hznum(tempp(1))=100000;             % 大幅度提升频数 不让参与下次分配
    tempp=find(Hznum==min(Hznum));      % 找第二个
    select(2)=tempp(1);
    numz(2)=Hznum(tempp(1));
    Hznum(tempp(1))=100000;             % 大幅度提升频数 不让参与下次分配
    Hznum(select(1))=numz(2)+numz(1);   % 将两个合成一个节点 保存在左侧子树
        % 组合成为一个二叉树
        % 进行倒序编码 第一个编码0 第二个为1
        for p=1:2
                str=oritable{1,select(p)};
                for j=1:length(str);
                    % 找出每个位置 增加前面的根节点编码
                    for k=1:numstar
                        if (codetable{1,k}==str(j));
                            posi=k;
                            break;
                        end
                    end     % 位置遍历完毕
                    codetable{2,posi}=[num2str(p-1),codetable{2,posi}]; % 扩充编码
                end
        end
     % 组合新的节点符号与频数
     oritable{1,select(1)}=[oritable{1,select(1)},oritable{1,select(2)}];
     oritable{2,select(1)}=Hznum(select(1));
end

% 编码原文
code='';
a=[];
posi=0;
for h=1:length(target)
    posi=find(Hztar==target(h));
    code=[code,codetable{2,posi}];
    l1=length(codetable(1,:));
end
t4=clock;

% 计算效率 编码后长度算上码表的
nowlength=length(code); % 编码长度
if type==1
    origainlength=7*length(target);
else
    origainlength=16*length(target);
end
efficent=nowlength/origainlength;
time=etime(t4,t1);
end