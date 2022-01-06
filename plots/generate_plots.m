
clc, clear all, close all
pwd = 'C:\Users\USERTEST\rl\PPO\';
file_path = fullfile(pwd, 'training\');
image_path = fullfile(pwd,'plots\');

data = readtable(fullfile(file_path, 'data'),'PreserveVariableNames',true);

mean=data.mean;

for i=1:5000

    t1=cell2mat(mean(i));
    t2=t1(2:end-1);
    new_mean(i)=str2num(t2);
end
%%
%std=data.std;
%ma=data.max;
%mi=data.min;
%epoch = 1:length(mean);

figure(1), hold on, grid on, box on
    %patch([epoch', fliplr(epoch')], [mi, fliplr(ma)],'b','FaceAlpha',.3);
    plot(new_mean, 'color',[0.9290 0.6940 0.1250] , 'LineWidth', 2);
    
    
    
    
    
    
    