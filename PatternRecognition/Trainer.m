classdef Trainer < matlab.System
    % Trainer Add summary here
    
    % Pre-computed constants
    properties(Constant)
        imgSize = [320,320];
        objSize = [40,40];
    end

    methods(Access = public)
        function enhancedBinaryImg = imenhance(~, rawImgPath, noiseThreshold)
            % image read
            img = imread(rawImgPath);
            % resize to fixed w & h
            %img = imresize(img, self.imgSize);
            % Convert to Gray image
            if size(img,3)==3 %RGB image
                img = rgb2gray(img);
            end
            % threshold or convert to binary image
            img = imbinarize(img);
            img = ~img; %negative
            % Remove all object containing fewer than noiseThreshold pixels
            enhancedBinaryImg = bwareaopen(img, noiseThreshold);
        end
        
        function [imgObjects, rectPositions] = extractObjects(self, enhancedBinaryImg)  
            %Segment all object in the image based on 8 connectivity
            objects = bwconncomp(enhancedBinaryImg,8);
            %foreach extracted object
            %initialize imgObjects as cell array
            imgObjects = cell(objects.NumObjects, 1);
            rectPositions = cell(objects.NumObjects, 1);
            for obj=1:objects.NumObjects
                %get colored pixels indexes column
                coloredPixelsIdx = objects.PixelIdxList(1,obj);
                %create a black image with the same enhancedBinaryImg size
                objImg = zeros(size(enhancedBinaryImg));
                %draw the white obj pixel by pixel
                for i=1:numel(coloredPixelsIdx)
                    %color the specified pixel by index
                    objImg(coloredPixelsIdx{i}) = 1;
                end
                %imgObjects{obj} = objImg;
                
                %get the BoundingBox for the objImg
                s = regionprops(objImg, 'BoundingBox');
                %get the bounding rectangle
                rect = s.BoundingBox;
                %crop & resize the extracted object then add to imgObjects array
                imgObjects{obj} = imresize(imcrop(objImg, rect), self.objSize);
                rectPositions{obj} = rect;
                
                %debug
                %imshow(imgObjects{obj});
            end        
        end
        
        function Centroid = getCentroid(~, imgObject)
            [m,n] = size(imgObject);
            X_hist=sum(imgObject,1); 
            Y_hist=sum(imgObject,2); 
            X=1:n; Y=1:m;
            if sum(X_hist) == 0
                centX = 0;
            else
                centX=sum(X.*X_hist)/sum(X_hist); 
            end
            if sum(Y_hist) == 0
                centY = 0;
            else
                centY=sum(Y'.*Y_hist)/sum(Y_hist);
            end
            Centroid = [centX centY];
            %get centroid pixel index
            %roundedX = round(centX);
            %roundedY = round(centY);
            % create helper indexer matrix
            %idxerMat = reshape(1:m*n, [m n]);
            %Centroid = idxerMat(roundedX, roundedY);
        end
        
        function Medoid = getMedoid(self, imgObject)
            [~, medX] = self.extractMedoidRow(imgObject');
            [~, medY] = self.extractMedoidRow(imgObject);
            Medoid = [medX medY];
            %get medoid pixel index
            %[m,n] = size(imgObject);
            % create helper indexer matrix
            %idxerMat = reshape(1:m*n, [m n]);
            %Medoid = round(idxerMat(medX, medY)/2);
        end
        
        function Perimeter = getPerimeter(~, imgObject)
            I=zeros(size(imgObject)); 
            I(2:end-1,2:end-1)=1;
            Perimeter = sum(reshape(imgObject.*I,1,[]));
        end
        
        function Area = getArea(~, imgObject)
            Area = 0;
            for i=1:numel(imgObject)
                if(imgObject(i))
                    Area = Area + 1;
                end
            end
        end
        
        function [dataSet, dataSetClasses, rectPositions] = Train(self, dataClasses, imagePaths2D, noiseThreshold, blockSize)
            dataSetClasses = cell(0,1);
            rectPositions = cell(0,1);
            dataSet_Initialized = 0;
            for classIdx = 1 : numel(dataClasses)
                classImgsPaths = imagePaths2D{classIdx};
                for classImgPathIdx = 1 : numel(classImgsPaths)
                    curImgPath = classImgsPaths{classImgPathIdx};
                    enhancedBinImg = self.imenhance(curImgPath, noiseThreshold);
                    [imgObjs, imgObjsPositions] = self.extractObjects(enhancedBinImg);
                    rectPositions = vertcat(rectPositions, imgObjsPositions);
                    for objIdx = 1 : numel(imgObjs)
                        curObj = imgObjs{objIdx};
                        curObjSegms = self.segment(curObj, blockSize);
                        numOfFeatures = 11;
                        if ~dataSet_Initialized %initialize for first time only
                            dataSet = zeros(0, numel(curObjSegms)*numOfFeatures);
                            dataSet_Initialized = 1;
                        end
                        colRange = 1:numOfFeatures;
                        [m,~] = size(dataSet);
                        %foreach object segment
                        for segIdx = 1:numel(curObjSegms)
                            curObjSegm = curObjSegms{segIdx};
                            featureVector = zeros(1, numOfFeatures);
                            % get all features & add to featureVector
                            featureVector(1,1:2) = self.getCentroid(curObjSegm);
                            %featureVector(1,3:4) = self.getMedoid(curObjSegm);
                            featureVector(1,3:4) = [0 0];
                            featureVector(1,5) = self.getPerimeter(curObjSegm);
                            featureVector(1,6) = self.getArea(curObjSegm);
                            s = regionprops(curObjSegm,'Euler');
                            try
                                featureVector(1,7) = s.EulerNumber;
                            catch
                                featureVector(1,7) = 0;
                            end
                            s = regionprops(curObjSegm,'Extent');
                            try
                                featureVector(1,8) = s.Extent;
                            catch
                                featureVector(1,8) = 0;
                            end
                            s = regionprops(curObjSegm,'MajorAxisLength');
                            try
                                featureVector(1,9) = s.MajorAxisLength;                            
                            catch
                                featureVector(1,9) = 0;
                            end
                            s = regionprops(curObjSegm,'MinorAxisLength');
                            try
                                featureVector(1,10) = s.MinorAxisLength;
                            catch
                                featureVector(1,10) = 0;
                            end
                            s = regionprops(curObjSegm,'Orientation');
                            try
                                featureVector(1,11) = s.Orientation;
                            catch
                                featureVector(1,11) = 0;
                            end
                            % add featureVector to dataSet
                            if max(featureVector)
                                featureVector = abs(featureVector);
                                featureVector = featureVector/max(featureVector); %normalize
                            end
                            dataSet(m+1, colRange) = featureVector;
                            
                            colRange = colRange + numOfFeatures;
                        end
                        dataSetClasses{end+1,:} = dataClasses{classIdx};
                    end
                end              
            end
        end
        
        function imgSegments = segment(~, img, blockSize)
            [imgX,imgY] = size(img);
            blockX = blockSize(1);
            blockY = blockSize(2);
            imgSegments = cell(0, 1);
            if blockX <= imgX && blockY <= imgY
                x1 = 1; x2 = blockX;
                while x2 <= imgX
                    y1 = 1; y2 = blockY;
                    while y2 <= imgY
                        imgSegments{end+1} = img(x1:x2, y1:y2);
                        y1 = y2+1;
                        y2 = y2+blockY;
                        
                        if y1 <= imgY && y2 > imgY
                            y1 = y1-(y2-imgY);
                            y2 = imgY;
                        end
                        
                    end
                    x1 = x2+1;
                    x2 = x2+blockX;
                    
                    if x1 <= imgX && x2 > imgX
                        x1 = x1-(x2-imgX);
                        x2 = imgX;
                    end
                    
                end
            else
                imgSegments{1} = img;
            end
        end
        
        function [dataSet, dataSetClasses, rectPositions] = TrainHOG(self, dataClasses, imagePaths2D, noiseThreshold, CellSize)
            dataSetClasses = cell(0,1);
            rectPositions = cell(0,1);
            dataSet_Initialized = 0;
            for classIdx = 1 : numel(dataClasses)
                classImgsPaths = imagePaths2D{classIdx};
                for classImgPathIdx = 1 : numel(classImgsPaths)
                    curImgPath = classImgsPaths{classImgPathIdx};
                    enhancedBinImg = self.imenhance(curImgPath, noiseThreshold);
                    [imgObjs, imgObjsPositions] = self.extractObjects(enhancedBinImg);
                    rectPositions = vertcat(rectPositions, imgObjsPositions);
                    for objIdx = 1 : numel(imgObjs)
                        curObj = imgObjs{objIdx};
                        hogFeatures = extractHOGFeatures(curObj,'CellSize', CellSize);
                        if ~dataSet_Initialized %initialize for first time only
                            dataSet = zeros(0, numel(hogFeatures));
                            dataSet_Initialized = 1;
                        end
                        dataSet(end+1,:) = hogFeatures;
                        dataSetClasses{end+1,:} = dataClasses{classIdx};
                    end
                end              
            end
        end
    end
    
    methods(Access = private)
        function [MedoidVal, MedoidIdx] = extractMedoidRow(~, imgObject)
            [m,n] = size(imgObject);
            MedoidVal = zeros(1,n);
            lastRowDistance = 0;
            for r = 1:m
                rowDistance = 0;
                for c = 1:n
                    curEl = imgObject(r,c);
                    for internalRow = 1:m
                        rowDistance = rowDistance + abs(curEl - imgObject(internalRow, c));
                    end
                end

                if r == 1
                    MedoidVal = imgObject(1,:); % extract the first row
                    MedoidIdx = 1;
                    lastRowDistance = rowDistance; % save the last row distance
                else
                    if rowDistance < lastRowDistance
                        MedoidVal = imgObject(r,:); % extract the r's row
                        MedoidIdx = r;
                        lastRowDistance = rowDistance;
                    end
                end
            end
        end
    end
end
