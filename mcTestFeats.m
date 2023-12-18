%%
% Compute features for a set of video files from datasets
%
close all;
clear;

% add path
addpath(genpath('include'));

%%
% parameters
algo_name = 'RAPIQUE'; % algorithm name
data_name = 'mctest';  % dataset name
write_file = true;  % if true, save features on-the-fly
log_level = 0;  % 1=verbose, 0=quite

root_path = 'C:\\Users\\ryo20\\Videos\\research\\mctest';
data_path = 'C:\\Users\\ryo20\\Videos\\research\\mctest';

%%
% create temp dir to store decoded videos
video_tmp = 'C:\Users\ryo20\RAPIQUE-gRPC\Data\tmp';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = 'mos_files';
filelist_csv = fullfile(feat_path, [data_name,'.csv']);
filelist = readtable(filelist_csv);
num_videos = size(filelist,1);
out_path = 'feat_files';
if ~exist(out_path, 'dir'), mkdir(out_path); end
out_mat_name = fullfile(out_path, [data_name,'_',algo_name,'_new_feats.mat']);
feats_mat = [];
feats_mat_frames = cell(num_videos, 1);
%===================================================

% init deep learning models
minside = 512.0;
net = resnet50;
layer = 'avg_pool';

%% extract features
% parfor i = 1:num_videos % for parallel speedup
for i = 1:num_videos
    progressbar(i/num_videos) % Update figure
    video_name = fullfile(data_path, ['Video_', num2str(filelist.name(i)), '.mp4']);
    yuv_name = fullfile(video_tmp, [num2str(filelist.name(i)), '.yuv']);
    fprintf('\n\nComputing features for %d sequence: %s\n', i, video_name);

    % decode video and store in temp dir
    ffmpeg_path = 'C:\Users\ryo20\ffmpeg-master-latest-win64-gpl-shared\bin\'; % ffmpeg の実行可能ファイルがあるディレクトリを指定
    cmd = [ffmpeg_path, 'ffmpeg -loglevel error -y -i ', video_name, ...
       ' -pix_fmt yuv420p -vsync 0 ', yuv_name];
    system(cmd);

    

    % get video meta data
    width = filelist.width(i);
    height = filelist.height(i);
    framerate = round(filelist.framerate(i));

    % calculate video features
    tStart = tic;
    feats_frames = calc_RAPIQUE_features(yuv_name, width, height, ...
        framerate, minside, net, layer, log_level);
    fprintf('\nOverall %f seconds elapsed...', toc(tStart));
    % 
    feats_mat(i,:) = nanmean(feats_frames);
    feats_mat_frames{i} = feats_frames;
    % clear cache
    delete(yuv_name)

    if write_file
        save(out_mat_name, 'feats_mat');
%         save(out_mat_name, 'feats_mat', 'feats_mat_frames');
    end
end




