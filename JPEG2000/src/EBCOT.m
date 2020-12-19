function [FLength,FCtxt,x_new]=EBCOT(filename,numOfBlocks,x)
%% function EBCOT.m
% Description：
% 该函数用于对数据块进行嵌入式编码
% filename：输入的数据块，大小要求为64*64的整型
% numOfBlocks：数据块的个数
% x：待存储码流的文件
% FLength：记录编码长度信息的文件
% FCtxt：记录上下文信息的文件
% x_new：存储码流后的文件
%%
% 初始化
sign=cell(numOfBlocks,1);       % 符号集
significant=cell(numOfBlocks,1);% 重要性集
refinement=cell(numOfBlocks,1); % 细化编码集
Block=cell(numOfBlocks,1);      % 码块
FCtxt=cell(numOfBlocks,1);
FLength=cell(numOfBlocks,1);

% 开始编码，对每个码块依次处理
for nb=1:numOfBlocks
    widthBlock=filename.width;
    heightBlock=filename.height;
    level=filename.level;
    subband=filename.subband;
    Block{nb}=filename.data; % data should be 64*64 and integer
    [~,x_new]=InitializeWrite(x);   % 对存储码流的文件进行初始化
    FCtxt{nb}=[];
    FLength{nb}=[];
    
    sign{nb}=zeros(heightBlock,widthBlock);
    significant{nb}=zeros(heightBlock,widthBlock);
    refinement{nb}=zeros(heightBlock,widthBlock);
    
    % 对符号集进行赋值，并将数据集全部取为正数
    [h,v]=find(Block{nb}<0);
    for i=1:length(h)
        sign{nb}(h(i),v(i))=1;
    end
    Block{nb}=abs(Block{nb});
    maximum=max(max(Block{nb}));
    
    % 通过位与操作获取位平面数
    numOfPlanes=1;
    for i=32:-1:1
        if bitand(maximum,bitshift(1,i-1))>0
            numOfPlanes=i;
            break;
        end
    end
    encoded=cell(numOfPlanes,1);    % 记录每个位置是否已编码

    % 创建位平面
    planeOfBits=cell(numOfPlanes,1);
    for i=1:numOfPlanes
        planeOfBits{i}=bitand(Block{nb},bitshift(1,i-1))>0;
    end
    
    % 开始编码
    FLength{nb}(end+1)=widthBlock;
    FLength{nb}(end+1)=heightBlock;
    FLength{nb}(end+1)=level;
    FLength{nb}(end+1)=subband;
    FLength{nb}(end+1)=numOfPlanes;
    
    for n=numOfPlanes:-1:1
        % 记录每个通道已扫描的个数
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
                        significant{nb},widthBlock,heightBlock,i,j);    % 获取邻居的重要性信息
                    if ~significant{nb}(i,j) && y                       % 判断自身是否重要
                        if planeOfBits{n}(i,j)
                            x_new=WriteBit(1,x_new);                    % 向文件中写入1
                            FCtxt{nb}(end+1)=Context(h,v,d,subband);    % 记录上下文信息
                            x_new=WriteBit(sign{nb}(i,j),x_new);        % 写入该位置的符号
                            [FCtxt{nb}(end+1),~,~]=ContextSign(h,v,significant{nb},sign{nb});   % 记录符号上下文信息
                            bitsGenProp=bitsGenProp+4;
                        else
                            x_new=WriteBit(0,x_new);                    % 向文件写入0
                            FCtxt{nb}(end+1)=Context(h,v,d,subband);    % 记录上下文信息
                            bitsGenProp=bitsGenProp+2;
                        end
                        encoded{n}(i,j)=1;                              % 标记为已编码
                        bitsPropagation=bitsPropagation+1;
                    end
                end
            end
        end
        
       %% start refinement
        for k=1:4:heightBlock
            for j=1:widthBlock
                for i=k:k+3
                    if ~encoded{n}(i,j) && significant{nb}(i,j)         % 判断是否已编码以及当前位置的重要性
                        x_new=WriteBit(planeOfBits{n}(i,j),x_new);      % 向文件写入当前所在位平面的数值
                        if refinement{nb}(i,j)
                            context=16;                                 % 上下文信息记为16
                            [y,~,~,~]=GetSignificantNeighbors(...
                                significant{nb},widthBlock,heightBlock,i,j);    % 获取邻居的重要性信息
                        else if y
                                context=15;                             % 上下文信息记为15
                            else
                                context=14;                             % 上下文信息记为14
                            end
                            refinement{nb}(i,j)=1;                      % 当前位置记为已通过细化编码扫描
                        end
                        FCtxt{nb}(end+1)=context;                       % 记录上下文信息
                        encoded{n}(i,j)=1;                              % 记为已编码
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
                                significant{nb},sign{nb});              % 记录符号上下文信息
                            significant{nb}(i,j)=1;                     % 该位置的重要性记为1
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
        % 记录各通道的编码个数
        FLength{nb}(end+1)=idivide(int32(bitsPropagation),int32(256));
        FLength{nb}(end+1)=mod(bitsPropagation,256);
        FLength{nb}(end+1)=idivide(int32(bitsRefinement),int32(256));
        FLength{nb}(end+1)=mod(bitsRefinement,256);
        FLength{nb}(end+1)=idivide(int32(bitsCleaning),int32(256));
        FLength{nb}(end+1)=mod(bitsCleaning,256);
    end
    x_new=EndWriting(x_new);    % 结束写入信息
end
end