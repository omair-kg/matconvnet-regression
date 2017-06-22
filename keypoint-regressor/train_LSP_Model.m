function train_LSP_Model(varargin)

run(fullfile(fileparts(mfilename('fullpath')),...
  '..','matconvnet', 'matlab', 'vl_setupnn.m')) ;

opts.datas='LSP';

opts.patchHei=120;
opts.patchWi=80;

opts.cam=1;%camera
opts.aug=40;%amount of augmentation

opts.batchSize = 256;
opts.numSubBatches = 1;
opts.numEpochs = 100 ;
opts.learningRate = 0.01;
opts.useBnorm = false ;
opts.prefetch = false ;

%GPU (leave it empty for training on CPU)
opts.gpus = [2];

opts.initNet=''; %pre-trained network
opts.outNode=28; %predicted-values
opts.inNode=3; %input channels
opts.lossFunc='tukeyloss'; %options: tukeyloss OR l2loss
opts.thrs=[];%not used
opts.refine=false;

%axis error plot (x,y)
opts.scbox=opts.patchWi*ones(opts.outNode,1);
opts.scbox(2:2:end)=opts.patchHei;

opts.expDir = sprintf('../data/%s/%s-base_baseline_%s_%d',opts.datas,opts.datas,opts.lossFunc,opts.cam) ;
opts.imdbPath = sprintf('../data/%s/%s-baseline_imdb%d.mat',opts.datas,opts.datas, opts.cam);%RAM image path

opts.DataMatTrain=sprintf('../data/%s/%s_imdbsT%daug%d.mat',opts.datas,opts.datas,opts.cam,opts.aug);
opts.DataMatVal=sprintf('../data/%s/%s_imdbsV%daug%d.mat',opts.datas,opts.datas,opts.cam,opts.aug);

%load network
%  load('/mnt/data0/goh4hi/MATLAB_files/matconvnet-deepReg/keypoint-regressor/matconvnet_0epoch.mat')
% opts.net =dagnn.DagNN.loadobj(net);%initializeRegNetwork(opts);
opts.net =initializeRegNetwork(opts);

%objectives
opts.derOutputs = {'objective',1} ;

opts = vl_argparse(opts, varargin);

%train
cnn_regressor_dag(opts);
