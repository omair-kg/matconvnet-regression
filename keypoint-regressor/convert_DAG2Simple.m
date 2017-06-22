function converted_net = convert_DAG2Simple
opts.inNode = 3;
opts.outNode = 28;
opts.lossFunc='tukeyloss';
simple_net = initializeRegNetwork(opts);
simple_net = vl_simplenn_tidy(simple_net);
% load('E:\mycache\CNNs\reg\STN-2branches_commonStart\dataAug-net-epoch-100.mat')
% load('E:\mycache\CNNs\reg\Buffy-FLIC\net-epoch-100.mat')
% load('C:\Users\goh4hi\Desktop\net-epoch-50.mat')
load('/mnt/data0/goh4hi/MATLAB_files/matconvnet-deepReg/data/LSP/LSP-baseline_tukeyloss_1/net-epoch-40.mat')
net = dagnn.DagNN.loadobj(net);
% net = removeLayers_DeployTime_dag(net);
converted_net.layers = {};

n_layers = length(net.layers);

for ii = 1 : n_layers
    block = net.layers(ii).block;
    block_type = class(block);
    switch block_type
        case 'dagnn.Conv'
            converted_net.layers{end+1} = simple_net.layers{ii};
            filts = net.getParam(net.layers(ii).params{1});
            converted_net.layers{end}.weights{1} = filts.value;
            if (size(net.layers(ii).params,2) == 2)
                biases = net.getParam(net.layers(ii).params{2});
                converted_net.layers{end}.weights{2} = biases.value;
            end
        case 'dagnn.ReLU'
            converted_net.layers{end+1} = simple_net.layers{ii};
        case 'dagnn.Pooling'
            converted_net.layers{end+1} = simple_net.layers{ii};
        case 'dagnn.LRN'
            converted_net.layers{end+1} = simple_net.layers{ii};
        case 'dagnn.Sigmoid'
            converted_net.layers{end+1} = simple_net.layers{ii};
        otherwise
    end
end
converted_net = vl_simplenn_tidy(converted_net);
end

function net = removeLayers_DeployTime_dag(net)
names = thisKindOfLayer_dag(net , 'dagnn.DropOut');
for i = 1:numel(names)
    layer = net.layers(net.getLayerIndex(names{i})) ;
    net.removeLayer(names{i}) ;
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
end

net = dagMergeBatchNorm(net);
names = thisKindOfLayer_dag(net, 'dagnn.BatchNorm') ;
for i = 1:numel(names)
    layer = net.layers(net.getLayerIndex(names{i})) ;
    net.removeLayer(names{i}) ;
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
end
end

function layers = thisKindOfLayer_dag(net , type)
layers = [] ;
for l = 1:numel(net.layers)
    if isa(net.layers(l).block, type)
        layers{1,end+1} = net.layers(l).name ;
    end
end
end

function net =  dagMergeBatchNorm(net)
% -------------------------------------------------------------------------
names = thisKindOfLayer_dag(net, 'dagnn.BatchNorm') ;
for name = names
  name = char(name) ;
  layer = net.layers(net.getLayerIndex(name)) ;

  % merge into previous conv layer
  playerName = dagFindLayersWithOutput(net, layer.inputs{1}) ;
  playerName = playerName{1} ;
  playerIndex = net.getLayerIndex(playerName) ;
  player = net.layers(playerIndex);
  if ~isa(player.block, 'dagnn.Conv')
    error('Batch normalization cannot be merged as it is not preceded by a conv layer.') ;
  end

  % if the convolution layer does not have a bias,
  % recreate it to have one
  if ~player.block.hasBias
    block = player.block ;
    block.hasBias = true ;
    net.renameLayer(playerName, 'tmp') ;
    net.addLayer(playerName, ...
                 block, ...
                 player.inputs, ...
                 player.outputs, ...
                 {player.params{1}, sprintf('%s_b',playerName)}) ;
    net.removeLayer('tmp') ;
    playerIndex = net.getLayerIndex(playerName) ;
    player = net.layers(playerIndex) ;
    biases = net.getParamIndex(player.params{2}) ;
    net.params(biases).value = zeros(block.size(4), 1, 'single') ;
  end

  filters = net.getParamIndex(player.params{1}) ;
  biases = net.getParamIndex(player.params{2}) ;
  multipliers = net.getParamIndex(layer.params{1}) ;
  offsets = net.getParamIndex(layer.params{2}) ;
  moments = net.getParamIndex(layer.params{3}) ;

  [filtersValue, biasesValue] = mergeBatchNorm(...
    net.params(filters).value, ...
    net.params(biases).value, ...
    net.params(multipliers).value, ...
    net.params(offsets).value, ...
    net.params(moments).value) ;

  net.params(filters).value = filtersValue ;
  net.params(biases).value = biasesValue ;
end
end

function layers = dagFindLayersWithOutput(net, outVarName)
% -------------------------------------------------------------------------
layers = {} ;
for l = 1:numel(net.layers)
  if any(strcmp(net.layers(l).outputs, outVarName))
    layers{1,end+1} = net.layers(l).name ;
  end
end
end

% -------------------------------------------------------------------------
function [filters, biases] = mergeBatchNorm(filters, biases, multipliers, offsets, moments)
% -------------------------------------------------------------------------
% wk / sqrt(sigmak^2 + eps)
% bk - wk muk / sqrt(sigmak^2 + eps)
a = multipliers(:) ./ moments(:,2) ;
b = offsets(:) - moments(:,1) .* a ;
biases(:) = biases(:) + b(:) ;
sz = size(filters) ;
numFilters = sz(4) ;
filters = reshape(bsxfun(@times, reshape(filters, [], numFilters), a'), sz) ;
end

