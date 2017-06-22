function net = changeToSimple_hardcoded()

opts.inNode = 3;
opts.outNode = 28;
opts.lossFunc='l2lossloss';
net_simple = initializeRegNetwork_forTransfer(opts);

% load('/mnt/data0/goh4hi/MATLAB_files/matconvnet-deepReg/data/CRP-gray/CRP-gray-baseline_tukeyloss_1/net-epoch-50.mat')
% load('/mnt/data0/goh4hi/MATLAB_files/matconvnet-deepReg/data/FLIC/FLIC-baseline_tukeyloss_1/net-epoch-21.mat')
%load('/mnt/data0/goh4hi/MATLAB_files/matconvnet-deepReg/data/LSP/LSP-baseline_l2loss_1_/net-epoch-10.mat')
opts.inNode = 3;
opts.outNode = 28;
opts.lossFunc='l2loss';
net = initializeRegNetwork(opts);
net = net.saveobj();
N_layers = length(net_simple.layers);
idx = 1;
for ii = 1 : N_layers
    layer = net_simple.layers{ii};
    type = layer.type;
    if strcmp(type , 'conv')
        layer.weights{1} = net.params(idx).value;
        idx = idx + 1;
        layer.weights{2} = net.params(idx).value;
        idx = idx + 1;
        net_simple.layers{ii} = layer;
    end
end
net = net_simple;
