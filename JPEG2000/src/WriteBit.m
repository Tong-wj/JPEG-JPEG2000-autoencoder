function y=WriteBit(bit,x)
%% function WriteBit.m
% Description��
% �ú����������ļ�д���ض��ı���λ
% bit����д��ı���λ
% x���洢���������ļ�
% y���洢���Ժ���ļ�
% p.s.�ⲿ�ִ������ο��Ĵ�����ʹ����ʮ������������λ���㣬����MATLAB���޷�ʵ�����ԣ�
%     ���д���������ԭ���Ķ�����������̫һ���������ָܻ�����
%%
y=x;
nbits=1;
while nbits>y.pos
    nbits=nbits-y.pos;
    y.output=bitor(y.output,bitshift(int32(bit),int32(-nbits)));  % ����ԭ����ʹ����ʮ�����ƽ�������
    y.file(end+1)=int32(y.output);
    y.pos=8;
    y.output=0;
end
y.pos=y.pos-nbits;
y.output=bitor(int32(y.output),bitshift(int32(bit),int32(y.pos)));% ����ԭ����ʹ����ʮ�����ƽ�������
end