function [ tile ] = split_4( RGB )
tile=cell(16,1);
t=1;
for i=1:4
    for j=1:4
      tile{t}.data=RGB(64*j-63:64*j,64*i-63:64*i,:);
      t=t+1;
    end
end
tile{1}.sub=0;
tile{1}.level=2;
tile{2}.sub=2;
tile{2}.level=2;
tile{3}.sub=2;
tile{3}.level=1;
tile{4}.sub=2;
tile{4}.level=1;
tile{5}.sub=1;
tile{5}.level=2;
tile{6}.sub=3;
tile{6}.level=2;
tile{7}.sub=2;
tile{7}.level=1;
tile{8}.sub=2;
tile{8}.level=1;
tile{9}.sub=1;
tile{9}.level=1;
tile{10}.sub=1;
tile{10}.level=1;
tile{11}.sub=3;
tile{11}.level=1;
tile{12}.sub=3;
tile{12}.level=1;
tile{13}.sub=1;
tile{13}.level=1;
tile{14}.sub=1;
tile{14}.level=1;
tile{15}.sub=3;
tile{15}.level=1;
tile{16}.sub=3;
tile{16}.level=1;
end

