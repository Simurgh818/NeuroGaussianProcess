

colNames = {'Stim','Xk0', 'Xk1'};

conversionTable = table(Stim, Xk0, Xk1, 'VariableNames',colNames);
fPath = 'C:\Users\Sina\OneDrive - Georgia Institute of Technology\DrGross\Eric\NeuroGaussianProcess';
fName = 'Mark_4sec_CA3PSD_ISO_freqamp_020619';
csv_fileName = [fName, '.csv'];
fileName = fullfile(fPath, csv_fileName);
writetable(conversionTable, fileName);
