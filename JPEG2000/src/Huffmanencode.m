function [code,efficent,codetable,time] = Huffmanencode(text,type)
%% function Huffmanencode.m
% Description:
% �ú������ڶ������ַ������ı�����Huffman����
% text�������ı�
% type���ı����ͣ�1��Ӣ�� 0������
% code���������
% efficent��ѹ����
% codetable�����
% time����ʱ
%%
t1=clock;

Hztar=' ';
Hznum=0;
% ��ȡ�ı�
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

% ������
numstar=length(Hznum);
% �����
codetable=cell(2,numstar);
% ����ö�Ӧ�ķ��������
oritable=cell(2,numstar);
for i=1:numstar
    codetable{1,i}=Hztar(i);
    codetable{2,i}='';
    oritable{1,i}=Hztar(i);
    oritable{2,i}=Hznum(i);
end
% �����Ƿ�ֻʣ����������ж��Ƿ���ɱ���
while (length(find(Hznum==100000))<=numstar-2)
    % ����ѡ����� ����
    select=[0,0];
    num=[0,0];                          % Ƶ��
    posi=-1;                            % ��¼λ��
    tempp=find(Hznum==min(Hznum));      % �ҵ�һ����С
    select(1)=tempp(1);
    numz(1)=Hznum(tempp(1));
    Hznum(tempp(1))=100000;             % ���������Ƶ�� ���ò����´η���
    tempp=find(Hznum==min(Hznum));      % �ҵڶ���
    select(2)=tempp(1);
    numz(2)=Hznum(tempp(1));
    Hznum(tempp(1))=100000;             % ���������Ƶ�� ���ò����´η���
    Hznum(select(1))=numz(2)+numz(1);   % �������ϳ�һ���ڵ� �������������
        % ��ϳ�Ϊһ��������
        % ���е������ ��һ������0 �ڶ���Ϊ1
        for p=1:2
                str=oritable{1,select(p)};
                for j=1:length(str);
                    % �ҳ�ÿ��λ�� ����ǰ��ĸ��ڵ����
                    for k=1:numstar
                        if (codetable{1,k}==str(j));
                            posi=k;
                            break;
                        end
                    end     % λ�ñ������
                    codetable{2,posi}=[num2str(p-1),codetable{2,posi}]; % �������
                end
        end
     % ����µĽڵ������Ƶ��
     oritable{1,select(1)}=[oritable{1,select(1)},oritable{1,select(2)}];
     oritable{2,select(1)}=Hznum(select(1));
end

% ����ԭ��
code='';
a=[];
posi=0;
for h=1:length(target)
    posi=find(Hztar==target(h));
    code=[code,codetable{2,posi}];
    l1=length(codetable(1,:));
end
t4=clock;

% ����Ч�� ����󳤶���������
nowlength=length(code); % ���볤��
if type==1
    origainlength=7*length(target);
else
    origainlength=16*length(target);
end
efficent=nowlength/origainlength;
time=etime(t4,t1);
end