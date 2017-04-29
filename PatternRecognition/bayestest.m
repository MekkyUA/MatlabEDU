dataClasses = {'male','male','male','male', 'female', 'female', 'female', 'female'};
dataSet = [6 180 12;5.92 190 11; 5.58 170 12; 5.92 165 10; 5 100 6; 5.5 150 8; 5.42 130 7; 5.75 150 9 ];
new = [6 30 8];
cl = Classifier;
bh = BayesHelper;
[baySet, classes, classesProps] = bh.getBayesianSet(dataSet, dataClasses);
classType = cl.bayesClassify(baySet, classes, classesProps, new);

