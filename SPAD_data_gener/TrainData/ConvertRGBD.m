% The directory where you extracted the raw dataset.
%%
% 注意：
% 需要先运行intrinsic_texture-master下的demo.m文件，该程序才能运行，否则部分dll无法正确加载
clc;clear;close all
%%
addpath(genpath('./intrinsic_texture-master'));
addpath('nyu_utils');
datasetDir = './raw';

camera_params();
% get the scene names
scenes = ls(datasetDir);
scenes = regexp(cellstr(scenes), '(\s+|\n)', 'split');
scenes(1:2) = [];

xcoll =0; 

for ss = 1+xcoll:length(scenes)
    
    sceneName = cell2mat(scenes{ss});
    
    disp('starting!');

    % The name of the scene to demo.
    outdir = ['./processed/' sceneName];
    
    if exist(outdir, 'dir')==0 
        mkdir(outdir);
    end
    
    % The absolute directory of the 
    sceneDir = sprintf('%s/%s', datasetDir, sceneName);

    % Reads the list of frames.
    frameList = get_synched_frames(sceneDir);

    % Displays each pair of synchronized RGB and Depth frames.
    idx = 1 : 10 : numel(frameList); 
    
    for ii = 1:length(idx)
        
        collapse = []; % Data that will cause MATLAB to crash
        
        if ismember(idx(ii), collapse)
            disp('crash continue！')
            continue;
        end
        % check if already exists
        depth_out = sprintf('%s/depth_%04d.mat', outdir, idx(ii));
        albedo_out = sprintf('%s/albedo_%04d.mat', outdir, idx(ii));
        intensity_out = sprintf('%s/intensity_%04d.mat', outdir, idx(ii));
        dist_out = sprintf('%s/dist_%04d.mat',outdir, idx(ii));
        dist_out_hr = sprintf('%s/dist_hr_%04d.mat',outdir, idx(ii));

        if exist(albedo_out,'file') ...
                && exist(intensity_out,'file') && exist(dist_out,'file') ...
                && exist(dist_out_hr,'file')    
                disp('continuing');
                continue;
        end
        
        
        try
            imgRgb = imread([sceneDir '/' frameList(idx(ii)).rawRgbFilename]);
            imgDepthRaw = swapbytes(imread([sceneDir '/' frameList(idx(ii)).rawDepthFilename]));

            % Crop the images to include the areas where we have depth information.
            imgRgb = crop_image(imgRgb);
            imgDepthProj = project_depth_map(imgDepthRaw, imgRgb);
            imgDepthAbs = crop_image(imgDepthProj);
            imgDepthFilled = fill_depth_cross_bf(imgRgb, double(imgDepthAbs));
          
            % get distance from the depth image
            cx = cx_d - 41 + 1;
            cy = cy_d - 45 + 1;
            [xx,yy] = meshgrid(1:561, 1:427);
            X = (xx - cx) .* imgDepthFilled / fx_d;
            Y = (yy - cy) .* imgDepthFilled / fy_d;
            Z = imgDepthFilled;
            imgDist_hr = sqrt(X.^2 + Y.^2 + Z.^2);
           
            % estimate the albedo image and save the outputs
            I = im2double(imgRgb);
            I = imresize(I, [512, 512], 'bilinear');
            imgDepthFilled = imresize(imgDepthFilled, [512,512], 'bilinear');
            imgDist = imresize(imgDist_hr, [256,256], 'bilinear');
            imgDist_hr = imresize(imgDist_hr, [512,512], 'bilinear');
            intensity = rgb2gray(I);
            albedo = I;

            dist = imgDist;
            intensity = im2uint8(intensity);
            dist_hr = imgDist_hr;
            ConvertRGBDParsave(albedo_out, dist_out, intensity_out, dist_out_hr, albedo, dist, intensity, dist_hr)% 存储数据
             
        catch e
            fprintf(1,'ERROR: %s\n',e.identifier);
            fprintf(1,'%s',e.message);
            continue;
        end
    end
end
