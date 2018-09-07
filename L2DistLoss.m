classdef L2DistLoss < dagnn.ElementWise
  properties
      name = ''
      hingeThreshold = []
      opts = {}
  end

  properties (Transient)
    % order of the fields matter to the extractStats function !
    averagePositive = 0
    averageNegative = 0
    averageInformativeNegatives = 0   
    numAveragedPositive = 0   
    numAveragedNegative = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      positiveSamples = gather(inputs{2}) == 1;
      negativeSamples = ~positiveSamples;
      
      % Positive samples just need sum as we want to minimize the distance
      outputs{1} = sum(inputs{1}(:,:,:,positiveSamples));
      % Negative samples are evaluated using hinge loss function (vl_nnloss returns
      % the loss as sum) hence the -1 mul.
      outputs{2} = vl_nnloss(inputs{1}(:,:,:,negativeSamples), -1*inputs{2}(negativeSamples,:), [], 'loss', 'hinge','hingeThreshold',obj.hingeThreshold) ;
          
      InformativeNegatives = sum(inputs{1}(:,:,:,negativeSamples) < obj.hingeThreshold);
      
      outputs{3} = InformativeNegatives;
      
      n = obj.numAveragedPositive ;
      m = n + sum(positiveSamples) ;
      obj.averagePositive = (n * obj.averagePositive + gather(outputs{1})) / m ;
      obj.numAveragedPositive = m ;
      
      n = obj.numAveragedNegative ;
      m = n + sum(negativeSamples) ;
      obj.averageNegative = (n * obj.averageNegative + gather(outputs{2})) / m ;
      obj.averageInformativeNegatives = gather((n * obj.averageInformativeNegatives + InformativeNegatives) / m );
      obj.numAveragedNegative = m ;
              
%       obj.numAveraged = obj.numAveragedNegative + obj.numAveragedPositive;
%       obj.average = obj.averageNegative*(obj.numAveragedNegative/(obj.numAveraged)) +...
%                     obj.averagePositive*(obj.numAveragedPositive/(obj.numAveraged));
      
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      positiveSamples = gather(inputs{2}) == 1;
      negativeSamples = ~positiveSamples;
      
      derInputs{1} = zeros(size(inputs{1}),'like',inputs{1});
      derInputs{2} = [];
      
      derInputs{1}(:,:,:,positiveSamples) = 1*ones(size(inputs{1}(:,:,:,positiveSamples)))*derOutputs{1} ;
      derInputs{1}(:,:,:,negativeSamples) = vl_nnloss(inputs{1}(:,:,:,negativeSamples), -1*inputs{2}(negativeSamples,:), derOutputs{2}, 'loss', 'hinge','hingeThreshold',obj.hingeThreshold) ; 
      derParams = {} ;
      
%       nonInformativeNegatives = sum(inputs{1}(:,:,:,negativeSamples) < obj.hingeThreshold)/length(negativeSamples);
%       positiveNegativeRatio = mean(inputs{1}(:,:,:,positiveSamples))/mean(inputs{1}(:,:,:,negativeSamples));
%       if ((nonInformativeNegatives <= 0.3) && (positiveNegativeRatio <= 0.3))
%           obj.hingeThreshold = obj.hingeThreshold + 0.1*obj.hingeThreshold;
%       end    
    end
    

    function reset(obj)
%         obj.average = 0;
%         obj.numAveraged = 0;
        obj.averageInformativeNegatives = 0;
        obj.averagePositive = 0;
        obj.numAveragedPositive = 0;
        obj.averageNegative = 0;
        obj.numAveragedNegative = 0;
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

    function obj = L2DistLoss(varargin)
      obj.load(varargin) ;
      obj.hingeThreshold = obj.hingeThreshold;
      obj.name = obj.name;
    end
  end
end
