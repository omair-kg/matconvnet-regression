function pcp_eval()

%To be set:
GPUt = [2]; %number of GPU

%visualization
vis = 0;
data = {};
%%%%%%%%%%%%%%%% LSP %%%%%%%%%%%%%%%%

%Dataset
rootPth='/../data/LSP/';

datas='LSP';

load(['../data/lsp_observer_centric/' 'jointsOC.mat']);
joints=permute(joints,[2 1 3]);
jointsAll = joints(:,1:2,:);
loadImgName=0;

imgFmt=['../data/lsp_dataset/' 'images/im%.4d.jpg'];

%Dataset Frames
vaBeg=1001;
vaEnd=2000;

%Bounding Box shifts
v1=20;v2=10;v3=40;v4=20;

%%%%%%%%%%%%%%%% LSP %%%%%%%%%%%%%%%%


% Fixed parameters
ref_scale = [120,80]; %input to the network



%PCP evaluation
finalPCP=0;
parts = {'head','upper_arms_right','upper_arms_left','lower_arms_right','lower_arms_left',...
    'upper_legs_right','upper_legs_left','lower_legs_right','lower_legs_left','torso'};
posture = hdf5read('/mnt/data1/goh4hi/PoseExperiments/Evaluation_pose/boost1_24_lsp.h5','/pos');
% posture = hdf5read('/mnt/data0/goh4hi/poseEstimation_torch/trainingLogs/poseVal/wgan_lsp_sanity.h5','/pos');
% posture = hdf5read('/mnt/data0/openshare/LSP/output2.hdf5','validation');

% clc
GT = zeros(14,2,1000);
Pred = zeros(14,2,1000);
indexer = 1;
for fr=vaBeg:vaEnd
    %disp(fr);
    
    img = imread(sprintf(imgFmt,fr)); %%%%%%NUMBERING%%%%%%
    poseGT=jointsAll(:,:,fr);
    
    %fit a bounding box
    x=min(poseGT(:,1));
    y=min(poseGT(:,2));
    wi=max(poseGT(:,1))-min(poseGT(:,1));
    hei=max(poseGT(:,2))-min(poseGT(:,2));
    
    %extend it
    x=round(x-v1);
    y=round(y-v2);
    wi=round(wi+v3);
    hei=round(hei+v4);
    
    % Check the bounding box
    x = max(1,x);
    y = max(1,y);
    wi  = min(size(img,2)-1,x+wi)-x;
    hei = min(size(img,1)-1,x+hei)-y;    
  
    %crop the image
    imgCrop = img(y:y+hei,x:x+wi,:);
    
    
    
    %single format and mean subtraction
    
    pose = posture(:,fr-1000);
    %pose = squeeze(net.vars(net.getVarIndex(layer)).value);
    %data{end+1}=gather(pose);
    pose = (reshape(pose,[2 numel(pose)/2]))';
    pose(:,1) = (size(imgCrop,2).*pose(:,1))+ x - 1;
    pose(:,2) = (size(imgCrop,1).*pose(:,2))+ y - 1;
  
    if vis
        imshow(img); hold on;
        xp = pose(:,1);
        yp = pose(:,2);
        plot(xp,yp,'yx','MarkerSize',10);
        hold off; pause();
    end
    GT(:,:,indexer) = poseGT;
    Pred(:,:,indexer) = pose;
    for key = 1:numel(parts)
        %PCP: loose / strict, threshold 0.5 (default)
        res(key) = get3DpcpJointRamaJoints(poseGT, pose, parts{key}, 0.5,'strict');
    end
    indexer = indexer + 1;
    finalPCP = finalPCP + sum(res);
end

disp(finalPCP/(numel(parts)*1000));
GT = permute(GT,[2 1 3]);
Pred = permute(Pred,[2 1 3]);
range = 0.5;
dist = getDistPCP(Pred,GT);
pcp = computePCP(dist,range);
genTablePCP(pcp,'LSP');
% save('/mnt/data0/goh4hi/poseEstimation_torch/h5_files/pose/lsp_netoutputGT','data','-v7.3');
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