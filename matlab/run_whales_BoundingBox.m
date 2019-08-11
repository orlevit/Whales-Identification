function run_whales_BoundingBox

%Input Directory
dir_in  = '..\imgExamles';

%Output Directory
dir_out = '..\out';

%------------------------------------------------------------
if_plot        = 1;
if_save_images = 1;
%---------------------------------------

count = 0;
dd=dir(dir_in);

%k_vec = [35,38,60];
k_vec = [];
f_names = {};
xy1_xy2 = [];
for k=1:length(dd) %Run over all images in the Directory
    
    if ~isempty(k_vec)
        if ~any(k == k_vec)
            continue
        end
    end
    
    if ~dd(k).isdir
%         fprintf('%d,', k);
        fname = dd(k).name;
        
        fullName = fullfile(dir_in,fname);
        rgb = imread(fullName);
        siz = size(rgb);
        if length(siz)==2
            continue
        end
        
        r=rgb(:,:,1);
        g=rgb(:,:,2);
        b=rgb(:,:,3);
        
        %Treat RGB images only
        if isequal(r,g) && isequal(r,b)
            continue
        end

        %Find Bounding Box
        [xy1,xy2,rect] = findBoundingBox(rgb);
                       
        if ~isempty(xy1)
            %Fill Arrays 
            f_names{end+1,1}   = fname;     %Image name
            xy1_xy2(end+1,1:4) = [xy1,xy2]; %Bounding Box corners: [xy LeftTop, xy RightBottom]
            count = count+1;
        end
        
        if if_plot
            figure(1); imgray(rgb,1)
            if ~isempty(rect)
                h2=rectangle('position',rect,'EdgeColor','m');
                set(h2,'LineWidth',2)
            end
            drawnow
            
            if if_save_images
                ii = strfind(fname,'_');
                fname(ii) = ' ';
                ii = strfind(fname,'.');
                fname(ii) = ' ';
                
                %saveas(gcf,fname)
                %            saveas(gcf,[fname],'jpeg')
                ff = fullfile(dir_out,[num2str(k) '_' fname]);
                saveas(1,ff,'jpeg')
            end
        end
    end
    
    if rem(k,100)==0
        fprintf('%d,', k);
%         fprintf('\n');
    end
    
    if rem(k,2000)==0        
       fprintf('\n');
    end
    
end

fprintf('\n');
fprintf('No. of Processed Images: %d\n',count);
return
