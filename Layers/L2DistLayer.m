classdef L2DistLayer < dagnn.ElementWise

  properties (Transient)
  end
    
  methods
    function outputs = forward(obj, inputs, params)   
      descriptorLength = size(inputs{1},3);
      numSamples = size(inputs{1},4);
%       x1 = reshape(reshape(inputs{1},descriptorLength,numSamples)',1,numSamples,descriptorLength);  
%       x2 = reshape(reshape(inputs{2},descriptorLength,numSamples)',1,numSamples,descriptorLength);  
%       outputs{1} = reshape(vl_nnpdist(x1, x2, 2, 'noRoot',true),[1 1 1 128]);
      outputs{1} = vl_nnpdist(inputs{1}, inputs{2}, 2, 'noRoot',true);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
%         descriptorLength = size(inputs{1},3);
%         numSamples = size(inputs{1},4);
%         x1 = reshape(reshape(inputs{1},descriptorLength,numSamples)',1,numSamples,descriptorLength);  
%         x2 = reshape(reshape(inputs{2},descriptorLength,numSamples)',1,numSamples,descriptorLength);  
%         
%         [dx1, dx2] = vl_nnpdist(x1, x2, 2, reshape(derOutputs{1},[1 numSamples]), 'noRoot',true);
%         derInputs{1} = reshape(reshape(dx1,numSamples,descriptorLength)',1,1,descriptorLength,numSamples);
%         derInputs{2} = reshape(reshape(dx2,numSamples,descriptorLength)',1,1,descriptorLength,numSamples);
        [derInputs{1}, derInputs{2}] = vl_nnpdist(inputs{1}, inputs{2}, 2, derOutputs{1}, 'noRoot',true);
        derParams = {} ;
    end

    function reset(obj)
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = L2DistLayer(varargin)
      obj.load(varargin) ;
    end
  end
end
