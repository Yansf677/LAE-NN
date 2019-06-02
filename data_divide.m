clc
clear
%% prepare training data
load TEdata.mat
all_x = [];
all_y = [];
train_x = [];
train_y = [];
index = [1 2 4 5 6 7 21]
class = size(index, 2)
for i = 1:class
    y = 0.*ones(480, class);
    y(:, i) = ones(480, 1);
    train_x = [train_x; TE_480(:, :, index(i))];
    train_y = [train_y; y];
end
train_x = train_x'; 
[train_x settings] = mapminmax(train_x, 0, 1);
train_x = train_x'; 
train_data = [train_x train_y];

%% prepare testing data
test_x = [];
for i = 1:class
    test_x = [test_x; TE_960(1:960, :, index(i))];
end
test_x = test_x';
test_x = mapminmax.apply(test_x, settings);
test_x = test_x';
clear class
clear d00
clear i
clear index
clear settings
clear TE_480
clear TE_960
clear y
clear train_y
clear all_x
clear all_y
clear val_y
save
