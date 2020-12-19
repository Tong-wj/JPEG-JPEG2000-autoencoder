function y=Context(h,v,d,subband)
%% function Context.m
% Description��
% �ú������ڻ�ȡλ�õ���������Ϣ
% h��ˮƽ�ھӵ���Ҫ����Ϣ
% v����ֱ�ھӵ���Ҫ����Ϣ
% d���Խ��ھӵ���Ҫ����Ϣ
% subband�����ڵ��Ӵ���Ϣ
% y������������ĵ�ֵ
%%
switch subband
	case 0 % LL
        y=initializeContextLL();
        y=y(h+1,v+1,d+1);
        return;
	case 1 % LH
		y=initializeContextLL();
        y=y(h+1,v+1,d+1);
        return;
	case 2 % HL
		y=initializeContextHL();
        y=y(h+1,v+1,d+1);
        return;
	case 3 % HH
		y=initializeContextHH();
        y=y(h+1,v+1,d+1);
        return;
    otherwise
		y=-100;
        return
end
end