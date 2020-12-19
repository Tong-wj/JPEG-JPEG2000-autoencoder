function [file,x_new]=DeEBCOT(numOfBlocks,x,FCtxt,FLength)
%% function DeEBCOT.m
% Description：
% 该函数用于对数据块进行嵌入式解码
% numOfBlocks：数据块的个数
% x：存储码流的文件
% FLength：记录编码长度信息的文件
% FCtxt：记录上下文信息的文件
% x_new：解码后的文件
% file：输出的数据块
%%
% 初始化
sign=cell(numOfBlocks,1);
significant=cell(numOfBlocks,1);
refinement=cell(numOfBlocks,1);
Block=cell(numOfBlocks,1);
file=cell(numOfBlocks,1);
numFL=1;    % 更新读取FLength元素的位置信息
numX=1;     % 更新读取x.file元素的位置信息
x_new=x;
for nb=1:numOfBlocks
    [~,x_new,numX]=InitializeReading(x_new,numX);
    widthBlock=FLength{nb}(numFL);
    numFL=numFL+1;
    heightBlock=FLength{nb}(numFL);
    numFL=numFL+1;
    level=FLength{nb}(numFL);
    numFL=numFL+1;
    subband=FLength{nb}(numFL);
    numFL=numFL+1;
    numOfPlanes=FLength{nb}(numFL);
    numFL=numFL+1;
    
    encoded=cell(numOfPlanes,1);
    planeOfBits=cell(numOfPlanes,1);
    sign{nb}=zeros(heightBlock,widthBlock);
    significant{nb}=zeros(heightBlock,widthBlock);
    refinement{nb}=zeros(heightBlock,widthBlock);
    
    %% 开始解码
    for n=numOfPlanes:-1:1
        bitsPropagation=0;
        bitsRefinement=0;
        bitsCleaning=0;
        encoded{n}=zeros(heightBlock,widthBlock);
         
        PartHigh=FLength{nb}(numFL);
        numFL=numFL+1;
        PartLow=FLength{nb}(numFL);
        numFL=numFL+1;
        
        PartHigh=FLength{nb}(numFL);
        numFL=numFL+1;
        PartLow=FLength{nb}(numFL);
        numFL=numFL+1;
        
        PartHigh=FLength{nb}(numFL);
        numFL=numFL+1;
        PartLow=FLength{nb}(numFL);
        numFL=numFL+1;
        
        % 开始解码
        %% start propagation
        for k=1:4:heightBlock
            for j=1:widthBlock
                for i=k:k+3
                    [y,~,~,~]=GetSignificantNeighbors(...
                        significant{nb},widthBlock,heightBlock,i,j);
                    if ~significant{nb}(i,j) && y
                        [planeOfBits{n}(i,j),x_new,numX]=ReadBit(x_new,numX);
                        if planeOfBits{n}(i,j)
                            [sign{nb}(i,j),x_new,numX]=ReadBit(x_new,numX);
                        end
                        encoded{n}(i,j)=1;
                        bitsPropagation=bitsPropagation+1;
                    end
                end
            end
        end
        
        %% start refinement
        for k=1:4:heightBlock
            for j=1:widthBlock
                for i=k:k+3
                    if ~encoded{n}(i,j) && significant{nb}(i,j)
                        [planeOfBits{n}(i,j),x_new,numX]=ReadBit(x_new,numX);
                        encoded{n}(i,j)=1;
                        bitsRefinement=bitsRefinement+1;
                    end
                end
            end
        end
        
        %% start cleaning
        for k=1:4:heightBlock
            for j=1:widthBlock
                for i=k:k+3
                    if ~encoded{n}(i,j)
                        [planeOfBits{n}(i,j),x_new,numX]=ReadBit(x_new,numX);
                        if planeOfBits{n}(i,j)
                            [sign{nb}(i,j),x_new,numX]=ReadBit(x_new,numX);
                            significant{nb}(i,j)=1;
                        end
                        bitsCleaning=bitsCleaning+1;
                    else
                        encoded{n}(i,j)=0;
                    end
                end
            end
        end
    end
        
    for i=1:heightBlock
        for j=1:widthBlock
            Block{nb}(i,j)=0;
            for n=numOfPlanes:-1:1
                if planeOfBits{n}(i,j)
                    Block{nb}(i,j)=Block{nb}(i,j)+int32(2^(n-1));
                end
            end
            if sign{nb}(i,j)
                Block{nb}(i,j)=(-1)*Block{nb}(i,j);
            end
        end
    end
    file{nb}=SaveBlock(widthBlock,heightBlock,level,subband,Block{nb});
end
end