function cnn_eval_CRP()

%To be set:
GPUt = [2]; %number of GPU

%visualization
vis = 0;
data = {};
%%%%%%%%%%%%%%%% LSP %%%%%%%%%%%%%%%%

%Dataset
rootPth='/../data/LSP/';

datas='CRP';



load('/mnt/data0/goh4hi/MATLAB_files/matconvnet-deepReg/data/CRP-gray/CRP_annot.mat')
img_name = annot.img_name;
N_ims = size(img_name,1);
joints = annot.coords;
bounds = annot.bounds;
jointsAll = joints(:,1:2,:);

imgFmt=['../data/lsp_dataset/' 'images/im%.4d.jpg'];

%Dataset Frames
vaBeg=21001;
vaEnd=N_ims;


%Bounding Box shifts
v1=20;v2=10;v3=40;v4=20;

%%%%%%%%%%%%%%%% LSP %%%%%%%%%%%%%%%%

% run(fullfile(fileparts(mfilename('fullpath')),...
%     '..','matconvnet', 'matlab', 'vl_setupnn.m')) ;

%Dataset
datas='LSP';

%Camera
cam=1;

%Network model
opts.Net = '/mnt/data0/goh4hi/MATLAB_files/matconvnet-deepReg/data/CRP-gray/CRP-gray-baseline_tukeyloss_1/net-epoch-30.mat';

opts.write= sprintf('../data/%s/result_jo%d.mat',datas,cam) ;

%trained network
load(opts.Net);
% load('/mnt/data0/goh4hi/MATLAB_files/matconvnet-deepReg/net-epoch-37.mat')
% net = convert_DAG2Simple();
% net.layers{end+1} = struct('type','tukeyloss');
% net = fromSimpleNN_custom(net, 'canonicalNames', true);
net = dagnn.DagNN.loadobj(net);
% net = changeToSimple_hardcoded();
% net = fromSimpleNN_custom(net, 'canonicalNames', true);
% net.move('cpu');
net.move('gpu');
% Fixed parameters
ref_scale = [120,80]; %input to the network
layer='prediction'; %keypoints

% Dataset mean (Works fine to use 128)
% dataMean = repmat(128,1,1,3);
load('/mnt/data0/goh4hi/poseEstimation_torch/h5_files/pose/poseMeanImage_120.mat')

GPUon=0;
if numel(GPUt)>0
    GPUon=1;
end

% if GPUon
%     gpuDevice(GPUt);
%     net.move('gpu');
% else
%     net.move('cpu');
% end

%PCP evaluation
finalPCP=0;
parts = {'head','upper_arms_right','upper_arms_left','lower_arms_right','lower_arms_left',...
    'upper_legs_right','upper_legs_left','lower_legs_right','lower_legs_left','torso'};
% posture = hdf5read('/mnt/data0/goh4hi/poseEstimation_torch/trainingLogs/poseVal/sanity_LSP_05032017.h5','/pos');
% clc
for fr=vaBeg:vaEnd
    disp(fr);
    imProxy = strrep(['/mnt/data0/goh4hi/MATLAB_files/matconvnet-deepReg/data/CRP-gray/Images/' img_name{fr}],'\','/');
    img = rgb2gray(imread(imProxy));
    poseGT=double(jointsAll(:,:,fr)); %2D GT
    
    %ensure correct values
        poseGT(:,1) = max(1,poseGT(:,1));
        poseGT(:,1) = min(size(img,2),poseGT(:,1));
        poseGT(:,2) = max(1,poseGT(:,2));
        poseGT(:,2) = min(size(img,1),poseGT(:,2));
        
        %fit a bounding box
        x = max(bounds(fr,1),1);
        y = max(bounds(fr,2),1);
        wi = bounds(fr,3);
        if(x+wi> size(img,2))
            wi = size(img,2)-x;
        end
        hei = bounds(fr,4);
        if(y+hei> size(img,1))
            hei = size(img,1)-y;
        end
    
    %extend it
    x=round(x-v1);
    y=round(y-v2);
    wi=round(wi+v3);
    hei=round(hei+v4);
    
    % Check the bounding box
       if x<=0
            x=1;
            
        end
        
        if (x+wi)>size(img,2)
            wi=size(img,2)-x;
        end
        
        if y<=0
            y=1;
        end
        
        if (y+hei)>size(img,1)
            hei=size(img,1)-y;
        end
  
    %crop the image
    imgCrop = img(y:y+hei,x:x+wi,:);
    
    %120X80 - network input
    imt = imresize(imgCrop,ref_scale);
    
    %single format and mean subtraction
    im_ = single(imt);
    
    if GPUon
        im_ = gpuArray(im_);
    end
    
    im_ = bsxfun(@minus, im_, single(dataMean)); %subtract mean
%     data{end+1} = gather(im_);
    %evaluate the image
    net.mode='test';
    net.eval({'input', im_}) ;

    pose = squeeze(net.vars(net.getVarIndex(layer)).value);
%     data{end+1}=gather(pose);
    pose = gather(reshape(pose,[2 numel(pose)/2]))';
    pose(:,1) = (size(imgCrop,2).*pose(:,1))+ x - 1;
    pose(:,2) = (size(imgCrop,1).*pose(:,2))+ y - 1;
  
    if vis
        imshow(img); hold on;
        xp = pose(:,1);
        yp = pose(:,2);
        plot(xp,yp,'yx','MarkerSize',10);
        hold off; pause();
    end
    
    for key = 1:numel(parts)
        %PCP: loose / strict, threshold 0.5 (default)
        res(key) = get3DpcpJointRamaJoints(poseGT, pose, parts{key}, 0.5,'loose');
    end
    finalPCP = finalPCP + sum(res);
end
clc
disp(finalPCP/(numel(parts)*(vaEnd-vaBeg)));
% save('/mnt/data0/goh4hi/poseEstimation_torch/h5_files/pose/lsp_sanity_data.mat','data','-v7.3');
end

function v = get3DpcpJointRamaJoints(gt3D, pose3D, flg, alpha, type)

%error between 2 skeletons

%head
if strcmp(flg ,'head')
    v=getPCPscore(gt3D(14,:), gt3D(13,:), pose3D(14,:), pose3D(13,:),alpha, type);
end

%upper_arms_right
if strcmp(flg ,'upper_arms_right')
    v=getPCPscore(gt3D(9,:), gt3D(8,:), pose3D(9,:), pose3D(8,:),alpha, type);
end

%upper_arms_left
if strcmp(flg ,'upper_arms_left')
    v=getPCPscore(gt3D(10,:), gt3D(11,:), pose3D(10,:), pose3D(11,:),alpha, type);
end

%lower_arms_right
if strcmp(flg ,'lower_arms_right')
    v=getPCPscore(gt3D(7,:), gt3D(8,:), pose3D(7,:), pose3D(8,:),alpha, type);
end

%lower_arms_left
if strcmp(flg ,'lower_arms_left')
    v=getPCPscore(gt3D(11,:), gt3D(12,:), pose3D(11,:), pose3D(12,:),alpha, type);
end

%upper_legs_right
if strcmp(flg ,'upper_legs_right')
    v=getPCPscore(gt3D(3,:), gt3D(2,:), pose3D(3,:), pose3D(2,:),alpha, type);
end

%upper_legs_left
if strcmp(flg ,'upper_legs_left')
    v=getPCPscore(gt3D(4,:), gt3D(5,:), pose3D(4,:), pose3D(5,:),alpha, type);
end

%lower_legs_right
if strcmp(flg ,'lower_legs_right')
    v=getPCPscore(gt3D(2,:), gt3D(1,:), pose3D(2,:), pose3D(1,:),alpha, type);
end

%lower_legs_left
if strcmp(flg ,'lower_legs_left')
    v=getPCPscore(gt3D(5,:), gt3D(6,:), pose3D(5,:), pose3D(6,:),alpha, type);
end

%torso
if strcmp(flg ,'torso')
    %virtual joint
    joPrGT= gt3D(3,:) + ((gt3D(4,:)-gt3D(3,:))./2);
    joPr= pose3D(3,:) + ((pose3D(4,:)-pose3D(3,:))./2);
    v = getPCPscore(gt3D(13,:),joPrGT, pose3D(13,:), joPr,alpha, type);
end


end

function v = getPCPscore(jProxGT, jDistGT, jProxPose, jDistPose, alpha, type)

startDif = norm(jProxGT - jProxPose);
endDif   = norm(jDistGT - jDistPose);

len=norm(jProxGT-jDistGT);

if strcmp(type,'strict')
    %strict PCP
    pcp_scoreA = startDif;
    pcp_scoreB = endDif;
   
    if pcp_scoreA<= (alpha*len) && pcp_scoreB<= (alpha*len)
        v=1;
    else
        v=0;
    end
else
    %loose PCP
    pcp_score = (startDif + endDif)/2;
    if pcp_score<= (alpha*len)
        v=1;
    else
        v=0;
    end
end


end