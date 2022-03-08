

colNames = {'Stim','Xk0', 'Xk1'};

conversionTable = table(Stim, Xk0, Xk1, 'VariableNames',colNames);
fPath = 'C:\Users\sinad\OneDrive - Georgia Institute of Technology\DrGross\Eric\GaussianProcess';
fName = 'Mark_4sec_CA1PSD_ISO_freqamp_020619';
csv_fileName = [fName, '.csv'];
fileName = fullfile(fPath, csv_fileName);
writetable(conversionTable, fileName);
