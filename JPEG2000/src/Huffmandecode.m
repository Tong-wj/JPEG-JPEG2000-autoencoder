function [decode,time] = Huffmandecode(codetable,code)
%% function Huffmandecode.m
% Description:
% �ú������ڸ��ݱ��������н���Huffman�����
% codetable�������
% code��������Ϣ����
% decode������������
% time������ʱ��
%%
% ����
t1=clock;
decode='';
unfinish=1;
while(unfinish)  
    flag=0;
    % ѭ��ȡ�������� ֱ�����Է���
    for o=1:length(code)
            str=code(1:o);
            % Ѱ���Ƿ��������
            for p=1:length(codetable)
                % �ҵ�������
                if strcmp(codetable{2,p},str)
                    decode=[decode,codetable{1,p}];
                    flag=1;
                    break;
                end
            end

            if flag==1;                 % ��ε��ҵ��� ����for������һ��
                if length(code)-o==0    % �ж��Ƿ���ȫ������
                    unfinish=0;
                else                    % �ضϺ����
                    code=code(o+1:length(code));
                end
                break;
            end
    end                                 % �ҵ���һ�� ������һ��
end
t2=clock;
time=etime(t2,t1);
end