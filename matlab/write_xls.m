%Construct XLS file

filename = 'BB_test.xlsx';
load res_test

% filename = 'BB_train.xlsx';
% load res_train

A = {'Name','x1', 'y1', 'x2', 'y2'};

B0 = f_names;
B1 = num2cell(xy1_xy2(:,1));
B2 = num2cell(xy1_xy2(:,2));
B3 = num2cell(xy1_xy2(:,3));
B4 = num2cell(xy1_xy2(:,4));

B = [B0,B1,B2,B3,B4];
C = [A;B];

%---------------------------------
sheet = 1;
xlRange = 'A1';
xlswrite(filename,C,sheet,xlRange)
