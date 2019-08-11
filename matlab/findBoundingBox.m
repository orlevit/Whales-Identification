function [xy1,xy2,rect] = findBoundingBox(rgb)
%Find Bounding Box
%CALL: [xy1,xy2,rect] = findBoundingBox(rgb)
%Input:  rgb image
%Output: xy1  - Left Top corner
%        xy2  - Right Bottom corner
%        rect - [x1,y1,dx,dy] (For use with "rectangle" (outside the current function))
         

%Separate RGB to [R,G,B] Channels 
r=rgb(:,:,1);
g=rgb(:,:,2);
b=rgb(:,:,3);

%Convert to HSV
hsv=rgb2hsv(rgb);
H=hsv(:,:,1);
S=hsv(:,:,2);
imgGray=rgb2gray(rgb);
V  =imgGray; %Set V to Gray Image

%Multithresholding
th_S = multithresh(S,2);  %Saturation 
th_V = multithresh(V,2);  %Gray Image

%Tails usually have low saturation
bwS = S < th_S(1);
bwV = V < th_V(1);
bwS1  = imopen(bwS,ones(5)); %Open - morphology filter
bw = (bwV & bwS1); %Image Intersection

[y,x] = find(bw); %Get relevant pixels
siz = size(rgb);
[xy1,xy2,rect] = cunstructBoundingBox(x,y,siz);

%If B-Box doesn't exist - try high saturation
if isempty(xy1)    
    bwS = S > th_S(2); %high saturation
    bwS1  = imopen(bwS,ones(5)); %Open - morphology filter    
    bw = (bwV & bwS1); %Image Intersection     
    [y,x] = find(bw);  %Get relevant pixels
    [xy1,xy2,rect] = cunstructBoundingBox(x,y,siz);
end

return

% Drawing (for Debug)
% figure(1); imgray(rgb,1)
% figure(2); imgray(S,1); title('S')
% figure(3); imgray(V,1); title('V')
% figure(4); imgray(gr); title('gr')

%Gradient (currently not in use)
% gr = imgrad(V);
% th_gr = multithresh(gr,2);
% bwGR= gr > th_gr(2);
% bwGR1 = imopen(bwGR,ones(3));
% bwGR2 = imclose(bwGR1,ones(5));

return

%===================================
function [xy1,xy2,rect] = cunstructBoundingBox(x,y,siz)

x1=min(x); x2=max(x);
y1=min(y); y2=max(y);
dx=x2-x1+1;
dy=y2-y1+1;

xy1 = [x1,y1]; %Left Top corner
xy2 = [x2,y2]; %Right Bottom corner
rect = [x1,y1,dx,dy];

if ~isempty(xy1)    
    nx=siz(2); ny=siz(1);
    area = dx*dy;
    r = area/(nx*ny);
    if r < 0.1
        xy1= [];
        xy2= [];
        rect = [];
    end
end
