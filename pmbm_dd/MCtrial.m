clc;clear

global Pd lfai slideWindow

for slideWindow = 3:6
    
    Pd = 0.9;lfai = 10;
    load(strcat('groundTruth3',num2str(100*Pd),num2str(lfai)));
    run main
    
    Pd = 0.7;lfai = 10;
    load(strcat('groundTruth3',num2str(100*Pd),num2str(lfai)));
    run main
    
    Pd = 0.9;lfai = 30;
    load(strcat('groundTruth3',num2str(100*Pd),num2str(lfai)));
    run main
    
    Pd = 0.7;lfai = 30;
    load(strcat('groundTruth3',num2str(100*Pd),num2str(lfai)));
    run main
    
end