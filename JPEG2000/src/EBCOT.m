function [FLength,FCtxt,x_new]=EBCOT(filename,numOfBlocks,x)
%% function EBCOT.m
% Description��
% �ú������ڶ����ݿ����Ƕ��ʽ����
% filename����������ݿ飬��СҪ��Ϊ64*64������
% numOfBlocks�����ݿ�ĸ���
% x�����洢�������ļ�
% FLength����¼���볤����Ϣ���ļ�
% FCtxt����¼��������Ϣ���ļ�
% x_new���洢��������ļ�
%%
% ��ʼ��
sign=cell(numOfBlocks,1);       % ���ż�
significant=cell(numOfBlocks,1);% ��Ҫ�Լ�
refinement=cell(numOfBlocks,1); % ϸ�����뼯
Block=cell(numOfBlocks,1);      % ���
FCtxt=cell(numOfBlocks,1);
FLength=cell(numOfBlocks,1);

% ��ʼ���룬��ÿ��������δ���
for nb=1:numOfBlocks
    widthBlock=filename.width;
    heightBlock=filename.height;
    level=filename.level;
    subband=filename.subband;
    Block{nb}=filename.data; % data should be 64*64 and integer
    [~,x_new]=InitializeWrite(x);   % �Դ洢�������ļ����г�ʼ��
    FCtxt{nb}=[];
    FLength{nb}=[];
    
    sign{nb}=zeros(heightBlock,widthBlock);
    significant{nb}=zeros(heightBlock,widthBlock);
    refinement{nb}=zeros(heightBlock,widthBlock);
    
    % �Է��ż����и�ֵ���������ݼ�ȫ��ȡΪ����
    [h,v]=find(Block{nb}<0);
    for i=1:length(h)
        sign{nb}(h(i),v(i))=1;
    end
    Block{nb}=abs(Block{nb});
    maximum=max(max(Block{nb}));
    
    % ͨ��λ�������ȡλƽ����
    numOfPlanes=1;
    for i=32:-1:1
        if bitand(maximum,bitshift(1,i-1))>0
            numOfPlanes=i;
            break;
        end
    end
    encoded=cell(numOfPlanes,1);    % ��¼ÿ��λ���Ƿ��ѱ���

    % ����λƽ��
    planeOfBits=cell(numOfPlanes,1);
    for i=1:numOfPlanes
        planeOfBits{i}=bitand(Block{nb},bitshift(1,i-1))>0;
    end
    
    % ��ʼ����
    FLength{nb}(end+1)=widthBlock;
    FLength{nb}(end+1)=heightBlock;
    FLength{nb}(end+1)=level;
    FLength{nb}(end+1)=subband;
    FLength{nb}(end+1)=numOfPlanes;
    
    for n=numOfPlanes:-1:1
        % ��¼ÿ��ͨ����ɨ��ĸ���
        bitsPropagation=0;
        bitsRefinement=0;
        bitsCleaning=0;
        bitsGenProp=0;
        bitsGenRef=0;
        bitsGenClea=0;
        encoded{n}=zeros(heightBlock,widthBlock);
        
       %% start propagation
        for k=1:4:heightBlock
            for j=1:widthBlock
                for i=k:k+3
                    [y,h,v,d]=GetSignificantNeighbors(...
                        significant{nb},widthBlock,heightBlock,i,j);    % ��ȡ�ھӵ���Ҫ����Ϣ
                    if ~significant{nb}(i,j) && y                       % �ж������Ƿ���Ҫ
                        if planeOfBits{n}(i,j)
                            x_new=WriteBit(1,x_new);                    % ���ļ���д��1
                            FCtxt{nb}(end+1)=Context(h,v,d,subband);    % ��¼��������Ϣ
                            x_new=WriteBit(sign{nb}(i,j),x_new);        % д���λ�õķ���
                            [FCtxt{nb}(end+1),~,~]=ContextSign(h,v,significant{nb},sign{nb});   % ��¼������������Ϣ
                            bitsGenProp=bitsGenProp+4;
                        else
                            x_new=WriteBit(0,x_new);                    % ���ļ�д��0
                            FCtxt{nb}(end+1)=Context(h,v,d,subband);    % ��¼��������Ϣ
                            bitsGenProp=bitsGenProp+2;
                        end
                        encoded{n}(i,j)=1;                              % ���Ϊ�ѱ���
                        bitsPropagation=bitsPropagation+1;
                    end
                end
            end
        end
        
       %% start refinement
        for k=1:4:heightBlock
            for j=1:widthBlock
                for i=k:k+3
                    if ~encoded{n}(i,j) && significant{nb}(i,j)         % �ж��Ƿ��ѱ����Լ���ǰλ�õ���Ҫ��
                        x_new=WriteBit(planeOfBits{n}(i,j),x_new);      % ���ļ�д�뵱ǰ����λƽ�����ֵ
                        if refinement{nb}(i,j)
                            context=16;                                 % ��������Ϣ��Ϊ16
                            [y,~,~,~]=GetSignificantNeighbors(...
                                significant{nb},widthBlock,heightBlock,i,j);    % ��ȡ�ھӵ���Ҫ����Ϣ
                        else if y
                                context=15;                             % ��������Ϣ��Ϊ15
                            else
                                context=14;                             % ��������Ϣ��Ϊ14
                            end
                            refinement{nb}(i,j)=1;                      % ��ǰλ�ü�Ϊ��ͨ��ϸ������ɨ��
                        end
                        FCtxt{nb}(end+1)=context;                       % ��¼��������Ϣ
                        encoded{n}(i,j)=1;                              % ��Ϊ�ѱ���
                        bitsGenRef=bitsGenRef+2;
                        bitsRefinement=bitsRefinement+1;
                    end
                end
            end
        end
        %% start clean
        for k=1:4:heightBlock
            for j=1:widthBlock
                for i=k:k+3
                    if ~encoded{n}(i,j)
                        if planeOfBits{n}(i,j)
                            x_new=WriteBit(1,x_new);
                            [y,h,v,d]=GetSignificantNeighbors(...
                                significant{nb},widthBlock,heightBlock,i,j);
                            FCtxt{nb}(end+1)=Context(h,v,d,subband);
                            x_new=WriteBit(sign{nb}(i,j),x_new);
                            [FCtxt{nb}(end+1),~,~]=ContextSign(h,v,...
                                significant{nb},sign{nb});              % ��¼������������Ϣ
                            significant{nb}(i,j)=1;                     % ��λ�õ���Ҫ�Լ�Ϊ1
                            bitsGenClea=bitsGenClea+4;
                        else
                            x_new=WriteBit(0,x_new);
                            [y,h,v,~]=GetSignificantNeighbors(...
                                significant{nb},widthBlock,heightBlock,i,j);
                            [FCtxt{nb}(end+1),~,~]=ContextSign(h,v,...
                                significant{nb},sign{nb});
                            bitsGenClea=bitsGenClea+2;
                        end
                        bitsCleaning=bitsCleaning+1;
                    end
                end
            end
        end
        % ��¼��ͨ���ı������
        FLength{nb}(end+1)=idivide(int32(bitsPropagation),int32(256));
        FLength{nb}(end+1)=mod(bitsPropagation,256);
        FLength{nb}(end+1)=idivide(int32(bitsRefinement),int32(256));
        FLength{nb}(end+1)=mod(bitsRefinement,256);
        FLength{nb}(end+1)=idivide(int32(bitsCleaning),int32(256));
        FLength{nb}(end+1)=mod(bitsCleaning,256);
    end
    x_new=EndWriting(x_new);    % ����д����Ϣ
end
end