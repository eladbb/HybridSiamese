function dagNet = Hybrid_Siamese_Multiple_L2_init(varargin)
opts.initializeFrom = '';
opts = vl_argparse(opts, varargin) ;
import dagnn.*
% init the object:
dagNet = dagnn.DagNN();
lr = [.1 2] ;
% the DAG dagNetwork structure  
% symmetric Siamese branch 
% ------------------------------------------------------------------  
% 0st conv-relu-pool layer - left
cnv0_left_symmetric_siamese = dagnn.Conv('size', [5 5 1 32], 'pad', 2, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv0_left_symmetric_siamese', cnv0_left_symmetric_siamese , {'siamese_left_symmetric_input'}, {'cnv0_left_symmetric_siamese_x'}, {'cnv0_symmetric_siamese_f', 'cnv0_symmetric_siamese_b'});
dagNet.params(end-1).value = 0.01*randn([5 5 1 32],'single');
dagNet.params(end).value = zeros(32,1,'single');
relu0_left_symmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu0_left_symmetric_siamese', relu0_left_symmetric_siamese, {'cnv0_left_symmetric_siamese_x'}, {'relu0_left_symmetric_siamese_x'}, {}) ;
pool0_left_symmetric_siamese = dagnn.Pooling('method','max','poolSize',[3 3],'stride', 2,'pad', [0 1 0 1]);
dagNet.addLayer('pool0_left_symmetric_siamese',pool0_left_symmetric_siamese,{'relu0_left_symmetric_siamese_x'},{'pool0_left_symmetric_siamese_x'},{});
% 0st conv-relu-pool layer - right
cnv0_right_symmetric_siamese = dagnn.Conv('size', [5 5 1 32], 'pad', 2, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv0_right_symmetric_siamese', cnv0_right_symmetric_siamese , {'siamese_right_symmetric_input'}, {'cnv0_right_symmetric_siamese_x'}, {'cnv0_symmetric_siamese_f', 'cnv0_symmetric_siamese_b'});
dagNet.params(end-1).value = 0.01*randn([5 5 1 32],'single');
dagNet.params(end).value = zeros(32,1,'single');
relu0_right_symmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu0_right_symmetric_siamese', relu0_right_symmetric_siamese, {'cnv0_right_symmetric_siamese_x'}, {'relu0_right_symmetric_siamese_x'}, {}) ;
pool0_right_symmetric_siamese = dagnn.Pooling('method','max','poolSize',[3 3],'stride', 2,'pad', [0 1 0 1]);
dagNet.addLayer('pool0_right_symmetric_siamese',pool0_right_symmetric_siamese,{'relu0_right_symmetric_siamese_x'},{'pool0_right_symmetric_siamese_x'},{});


% 1st conv-relu-pool layer - left
cnv1_left_symmetric_siamese = dagnn.Conv('size', [5 5 32 64], 'pad', 2, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv1_left_symmetric_siamese', cnv1_left_symmetric_siamese , {'pool0_left_symmetric_siamese_x'}, {'cnv1_left_symmetric_siamese_x'}, {'cnv1_symmetric_siamese_f', 'cnv1_symmetric_siamese_b'});
dagNet.params(end-1).value = 0.01*randn([5 5 32 64],'single');
dagNet.params(end).value = zeros(64,1,'single');
relu1_left_symmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu1_left_symmetric_siamese', relu1_left_symmetric_siamese, {'cnv1_left_symmetric_siamese_x'}, {'relu1_left_symmetric_siamese_x'}, {}) ;
pool1_left_symmetric_siamese = dagnn.Pooling('method','max','poolSize',[3 3],'stride', 2,'pad', [0 1 0 1]);
dagNet.addLayer('pool1_left_symmetric_siamese',pool1_left_symmetric_siamese,{'relu1_left_symmetric_siamese_x'},{'pool1_left_symmetric_siamese_x'},{});
% 1st conv-relu-pool layer - right
cnv1_right_symmetric_siamese = dagnn.Conv('size', [5 5 32 64], 'pad', 2, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv1_right_symmetric_siamese', cnv1_right_symmetric_siamese , {'pool0_right_symmetric_siamese_x'}, {'cnv1_right_symmetric_siamese_x'}, {'cnv1_symmetric_siamese_f', 'cnv1_symmetric_siamese_b'});
dagNet.params(end-1).value = 0.01*randn([5 5 32 64],'single');
dagNet.params(end).value = zeros(64,1,'single');
relu1_right_symmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu1_right_symmetric_siamese', relu1_right_symmetric_siamese, {'cnv1_right_symmetric_siamese_x'}, {'relu1_right_symmetric_siamese_x'}, {}) ;
pool1_right_symmetric_siamese = dagnn.Pooling('method','max','poolSize',[3 3],'stride', 2,'pad', [0 1 0 1]);
dagNet.addLayer('pool1_right_symmetric_siamese',pool1_right_symmetric_siamese,{'relu1_right_symmetric_siamese_x'},{'pool1_right_symmetric_siamese_x'},{});

% 2st conv-relu-pool layer - left
cnv2_left_symmetric_siamese = dagnn.Conv('size', [3 3 64 128], 'pad', 1, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv2_left_symmetric_siamese', cnv2_left_symmetric_siamese , {'pool1_left_symmetric_siamese_x'}, {'cnv2_left_symmetric_siamese_x'}, {'cnv2_symmetric_siamese_f', 'cnv2_symmetric_siamese_b'});
dagNet.params(end-1).value = 0.01*randn([3 3 64 128],'single');
dagNet.params(end).value = zeros(128,1,'single');
relu2_left_symmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu2_left_symmetric_siamese', relu2_left_symmetric_siamese, {'cnv2_left_symmetric_siamese_x'}, {'relu2_left_symmetric_siamese_x'}, {}) ;
pool2_left_symmetric_siamese = dagnn.Pooling('method','max','poolSize',[3 3],'stride', 2,'pad', [0 1 0 1]);
dagNet.addLayer('pool2_left_symmetric_siamese',pool2_left_symmetric_siamese,{'relu2_left_symmetric_siamese_x'},{'pool2_left_symmetric_siamese_x'},{});
% 2st conv-relu-pool layer - right
cnv2_right_symmetric_siamese = dagnn.Conv('size', [3 3 64 128], 'pad', 1, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv2_right_symmetric_siamese', cnv2_right_symmetric_siamese , {'pool1_right_symmetric_siamese_x'}, {'cnv2_right_symmetric_siamese_x'}, {'cnv2_symmetric_siamese_f', 'cnv2_symmetric_siamese_b'});
dagNet.params(end-1).value = 0.01*randn([3 3 64 128],'single');
dagNet.params(end).value = zeros(128,1,'single');
relu2_right_symmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu2_right_symmetric_siamese', relu2_right_symmetric_siamese, {'cnv2_right_symmetric_siamese_x'}, {'relu2_right_symmetric_siamese_x'}, {}) ;
pool2_right_symmetric_siamese = dagnn.Pooling('method','max','poolSize',[3 3],'stride', 2,'pad', [0 1 0 1]);
dagNet.addLayer('pool2_right_symmetric_siamese',pool2_right_symmetric_siamese,{'relu2_right_symmetric_siamese_x'},{'pool2_right_symmetric_siamese_x'},{});

% 3st conv-relu-pool layer - left
cnv3_left_symmetric_siamese = dagnn.Conv('size', [3 3 128 256], 'pad', 0, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv3_left_symmetric_siamese', cnv3_left_symmetric_siamese , {'pool2_left_symmetric_siamese_x'}, {'cnv3_left_symmetric_siamese_x'}, {'cnv3_symmetric_siamese_f', 'cnv3_symmetric_siamese_b'});
dagNet.params(end-1).value = 0.05*randn([3 3 128 256],'single');
dagNet.params(end).value = zeros(256,1,'single');
relu3_left_symmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu3_left_symmetric_siamese', relu3_left_symmetric_siamese, {'cnv3_left_symmetric_siamese_x'}, {'relu3_left_symmetric_siamese_x'}, {}) ;
% 3st conv-relu-pool layer - right
cnv3_right_symmetric_siamese = dagnn.Conv('size', [3 3 128 256], 'pad', 0, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv3_right_symmetric_siamese', cnv3_right_symmetric_siamese , {'pool2_right_symmetric_siamese_x'}, {'cnv3_right_symmetric_siamese_x'}, {'cnv3_symmetric_siamese_f', 'cnv3_symmetric_siamese_b'});
dagNet.params(end-1).value = 0.05*randn([3 3 128 256],'single');
dagNet.params(end).value = zeros(256,1,'single');
relu3_right_symmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu3_right_symmetric_siamese', relu3_right_symmetric_siamese, {'cnv3_right_symmetric_siamese_x'}, {'relu3_right_symmetric_siamese_x'}, {}) ;

% 4st conv-relu-pool layer - left
cnv4_left_symmetric_siamese = dagnn.Conv('size', [3 3 256 256], 'pad', 0, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv4_left_symmetric_siamese', cnv4_left_symmetric_siamese , {'relu3_left_symmetric_siamese_x'}, {'cnv4_left_symmetric_siamese_x'}, {'cnv4_symmetric_siamese_f', 'cnv4_symmetric_siamese_b'});
dagNet.params(end-1).value = 0.05*randn([3 3 256 256],'single');
dagNet.params(end).value = zeros(256,1,'single');
relu4_left_symmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu4_left_symmetric_siamese', relu4_left_symmetric_siamese, {'cnv4_left_symmetric_siamese_x'}, {'relu4_left_symmetric_siamese_x'}, {}) ;
% 4st conv-relu-pool layer - right
cnv4_right_symmetric_siamese = dagnn.Conv('size', [3 3 256 256], 'pad', 0, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv4_right_symmetric_siamese', cnv4_right_symmetric_siamese , {'relu3_right_symmetric_siamese_x'}, {'cnv4_right_symmetric_siamese_x'}, {'cnv4_symmetric_siamese_f', 'cnv4_symmetric_siamese_b'});
dagNet.params(end-1).value = 0.05*randn([3 3 256 256],'single');
dagNet.params(end).value = zeros(256,1,'single');
relu4_right_symmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu4_right_symmetric_siamese', relu4_right_symmetric_siamese, {'cnv4_right_symmetric_siamese_x'}, {'relu4_right_symmetric_siamese_x'}, {}) ;

% 5st conv-relu-pool layer - left
cnv5_left_symmetric_siamese = dagnn.Conv('size', [4 4 256 128], 'pad', 0, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv5_left_symmetric_siamese', cnv5_left_symmetric_siamese , {'relu4_left_symmetric_siamese_x'}, {'cnv5_left_symmetric_siamese_x'}, {'cnv5_symmetric_siamese_f', 'cnv5_symmetric_siamese_b'});
dagNet.params(end-1).value = 0.05*randn([4 4 256 128],'single');
dagNet.params(end).value = zeros(128,1,'single');
% 5st conv-relu-pool layer - right
cnv5_right_symmetric_siamese = dagnn.Conv('size', [4 4 256 128], 'pad', 0, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv5_right_symmetric_siamese', cnv5_right_symmetric_siamese , {'relu4_right_symmetric_siamese_x'}, {'cnv5_right_symmetric_siamese_x'}, {'cnv5_symmetric_siamese_f', 'cnv5_symmetric_siamese_b'});
dagNet.params(end-1).value = 0.05*randn([4 4 256 128],'single');
dagNet.params(end).value = zeros(128,1,'single');

% Normalize features to unit vector - left
L2Normalization_Symmetric_Left = L2NormalizationLayer();
dagNet.addLayer('L2Normalization_Symmetric_Left',L2Normalization_Symmetric_Left,{'cnv5_left_symmetric_siamese_x'},{'left_symmetric_Normalized_x'},{});

% Normalize features to unit vector - right
L2Normalization_Symmetric_Right = L2NormalizationLayer();
dagNet.addLayer('L2Normalization_Symmetric_Right',L2Normalization_Symmetric_Right,{'cnv5_right_symmetric_siamese_x'},{'right_symmetric_Normalized_x'},{});


l2Dist_Symmetric_Final = L2DistLayer();
dagNet.addLayer('l2Dist_Symmetric_Final',l2Dist_Symmetric_Final,{'left_symmetric_Normalized_x','right_symmetric_Normalized_x'},{'l2distSymmetricOutput'},{});

symmetric_L2Dist_HingeLossLayer = L2DistLoss('name','symmetric_L2Dist_HingeLossLayer','hingeThreshold',1);
dagNet.addLayer('symmetric_L2Dist_HingeLossLayer', symmetric_L2Dist_HingeLossLayer, {'l2distSymmetricOutput','labels'}, {'SymmetricPos','SymmetricNeg','SymmetricInfoNeg'}, {});

% ------------------------------------------------------------------
% asymmetric Siamese branch 
% ------------------------------------------------------------------  
% 0st conv-relu-pool layer - left
cnv0_left_asymmetric_siamese = dagnn.Conv('size', [5 5 1 32], 'pad', 2, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv0_left_asymmetric_siamese', cnv0_left_asymmetric_siamese , {'siamese_left_Asymmetric_input'}, {'cnv0_left_asymmetric_siamese_x'}, {'cnv0_left_asymmetric_siamese_f', 'cnv0_left_asymmetric_siamese_b'});
dagNet.params(end-1).value = 0.01*randn([5 5 1 32],'single');
dagNet.params(end).value = zeros(32,1,'single');
relu0_left_asymmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu0_left_asymmetric_siamese', relu0_left_asymmetric_siamese, {'cnv0_left_asymmetric_siamese_x'}, {'relu0_left_asymmetric_siamese_x'}, {}) ;
pool0_left_asymmetric_siamese = dagnn.Pooling('method','max','poolSize',[3 3],'stride', 2,'pad', [0 1 0 1]);
dagNet.addLayer('pool0_left_asymmetric_siamese',pool0_left_asymmetric_siamese,{'relu0_left_asymmetric_siamese_x'},{'pool0_left_asymmetric_siamese_x'},{});
% 0st conv-relu-pool layer - right
cnv0_right_asymmetric_siamese = dagnn.Conv('size', [5 5 1 32], 'pad', 2, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv0_right_asymmetric_siamese', cnv0_right_asymmetric_siamese , {'siamese_right_Asymmetric_input'}, {'cnv0_right_asymmetric_siamese_x'}, {'cnv0_right_asymmetric_siamese_f', 'cnv0_right_asymmetric_siamese_b'});
dagNet.params(end-1).value = dagNet.params(end-3).value;%0.01*randn([5 5 1 32],'single');
dagNet.params(end).value = zeros(32,1,'single');
relu0_right_asymmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu0_right_asymmetric_siamese', relu0_right_asymmetric_siamese, {'cnv0_right_asymmetric_siamese_x'}, {'relu0_right_asymmetric_siamese_x'}, {}) ;
pool0_right_asymmetric_siamese = dagnn.Pooling('method','max','poolSize',[3 3],'stride', 2,'pad', [0 1 0 1]);
dagNet.addLayer('pool0_right_asymmetric_siamese',pool0_right_asymmetric_siamese,{'relu0_right_asymmetric_siamese_x'},{'pool0_right_asymmetric_siamese_x'},{});

% 1st conv-relu-pool layer - left
cnv1_left_asymmetric_siamese = dagnn.Conv('size', [5 5 32 64], 'pad', 2, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv1_left_asymmetric_siamese', cnv1_left_asymmetric_siamese , {'pool0_left_asymmetric_siamese_x'}, {'cnv1_left_asymmetric_siamese_x'}, {'cnv1_left_asymmetric_siamese_f', 'cnv1_left_asymmetric_siamese_b'});
dagNet.params(end-1).value = 0.01*randn([5 5 32 64],'single');
dagNet.params(end).value = zeros(64,1,'single');
relu1_left_asymmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu1_left_asymmetric_siamese', relu1_left_asymmetric_siamese, {'cnv1_left_asymmetric_siamese_x'}, {'relu1_left_asymmetric_siamese_x'}, {}) ;
pool1_left_asymmetric_siamese = dagnn.Pooling('method','max','poolSize',[3 3],'stride', 2,'pad', [0 1 0 1]);
dagNet.addLayer('pool1_left_asymmetric_siamese',pool1_left_asymmetric_siamese,{'relu1_left_asymmetric_siamese_x'},{'pool1_left_asymmetric_siamese_x'},{});
% 1st conv-relu-pool layer - right
cnv1_right_asymmetric_siamese = dagnn.Conv('size', [5 5 32 64], 'pad', 2, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv1_right_asymmetric_siamese', cnv1_right_asymmetric_siamese , {'pool0_right_asymmetric_siamese_x'}, {'cnv1_right_asymmetric_siamese_x'}, {'cnv1_right_asymmetric_siamese_f', 'cnv1_right_asymmetric_siamese_b'});
dagNet.params(end-1).value = dagNet.params(end-3).value;%0.01*randn([5 5 32 64],'single');
dagNet.params(end).value = zeros(64,1,'single');
relu1_right_asymmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu1_right_asymmetric_siamese', relu1_right_asymmetric_siamese, {'cnv1_right_asymmetric_siamese_x'}, {'relu1_right_asymmetric_siamese_x'}, {}) ;
pool1_right_asymmetric_siamese = dagnn.Pooling('method','max','poolSize',[3 3],'stride', 2,'pad', [0 1 0 1]);
dagNet.addLayer('pool1_right_asymmetric_siamese',pool1_right_asymmetric_siamese,{'relu1_right_asymmetric_siamese_x'},{'pool1_right_asymmetric_siamese_x'},{});

% 2st conv-relu-pool layer - left
cnv2_left_asymmetric_siamese = dagnn.Conv('size', [3 3 64 128], 'pad', 1, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv2_left_asymmetric_siamese', cnv2_left_asymmetric_siamese , {'pool1_left_asymmetric_siamese_x'}, {'cnv2_left_asymmetric_siamese_x'}, {'cnv2_left_asymmetric_siamese_f', 'cnv2_left_asymmetric_siamese_b'});
dagNet.params(end-1).value = 0.01*randn([3 3 64 128],'single');
dagNet.params(end).value = zeros(128,1,'single');
relu2_left_asymmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu2_left_asymmetric_siamese', relu2_left_asymmetric_siamese, {'cnv2_left_asymmetric_siamese_x'}, {'relu2_left_asymmetric_siamese_x'}, {}) ;
pool2_left_asymmetric_siamese = dagnn.Pooling('method','max','poolSize',[3 3],'stride', 2,'pad', [0 1 0 1]);
dagNet.addLayer('pool2_left_asymmetric_siamese',pool2_left_asymmetric_siamese,{'relu2_left_asymmetric_siamese_x'},{'pool2_left_asymmetric_siamese_x'},{});
% 2st conv-relu-pool layer - right
cnv2_right_asymmetric_siamese = dagnn.Conv('size', [3 3 64 128], 'pad', 1, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv2_right_asymmetric_siamese', cnv2_right_asymmetric_siamese , {'pool1_right_asymmetric_siamese_x'}, {'cnv2_right_asymmetric_siamese_x'}, {'cnv2_right_asymmetric_siamese_f', 'cnv2_right_asymmetric_siamese_b'});
dagNet.params(end-1).value = dagNet.params(end-3).value;%0.01*randn([3 3 64 128],'single');
dagNet.params(end).value = zeros(128,1,'single');
relu2_right_asymmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu2_right_asymmetric_siamese', relu2_right_asymmetric_siamese, {'cnv2_right_asymmetric_siamese_x'}, {'relu2_right_asymmetric_siamese_x'}, {}) ;
pool2_right_asymmetric_siamese = dagnn.Pooling('method','max','poolSize',[3 3],'stride', 2,'pad', [0 1 0 1]);
dagNet.addLayer('pool2_right_asymmetric_siamese',pool2_right_asymmetric_siamese,{'relu2_right_asymmetric_siamese_x'},{'pool2_right_asymmetric_siamese_x'},{});

% 3st conv-relu-pool layer - left
cnv3_left_asymmetric_siamese = dagnn.Conv('size', [3 3 128 256], 'pad', 0, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv3_left_asymmetric_siamese', cnv3_left_asymmetric_siamese , {'pool2_left_asymmetric_siamese_x'}, {'cnv3_left_asymmetric_siamese_x'}, {'cnv3_left_asymmetric_siamese_f', 'cnv3_left_asymmetric_siamese_b'});
dagNet.params(end-1).value = 0.05*randn([3 3 128 256],'single');
dagNet.params(end).value = zeros(256,1,'single');
relu3_left_asymmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu3_left_asymmetric_siamese', relu3_left_asymmetric_siamese, {'cnv3_left_asymmetric_siamese_x'}, {'relu3_left_asymmetric_siamese_x'}, {}) ;
% 3st conv-relu-pool layer - right
cnv3_right_asymmetric_siamese = dagnn.Conv('size', [3 3 128 256], 'pad', 0, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv3_right_asymmetric_siamese', cnv3_right_asymmetric_siamese , {'pool2_right_asymmetric_siamese_x'}, {'cnv3_right_asymmetric_siamese_x'}, {'cnv3_right_asymmetric_siamese_f', 'cnv3_right_asymmetric_siamese_b'});
dagNet.params(end-1).value = dagNet.params(end-3).value;%0.05*randn([3 3 128 256],'single');
dagNet.params(end).value = zeros(256,1,'single');
relu3_right_asymmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu3_right_asymmetric_siamese', relu3_right_asymmetric_siamese, {'cnv3_right_asymmetric_siamese_x'}, {'relu3_right_asymmetric_siamese_x'}, {}) ;

% 4st conv-relu-pool layer - left
cnv4_left_asymmetric_siamese = dagnn.Conv('size', [3 3 256 256], 'pad', 0, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv4_left_asymmetric_siamese', cnv4_left_asymmetric_siamese , {'relu3_left_asymmetric_siamese_x'}, {'cnv4_left_asymmetric_siamese_x'}, {'cnv4_left_asymmetric_siamese_f', 'cnv4_left_asymmetric_siamese_b'});
dagNet.params(end-1).value = 0.05*randn([3 3 256 256],'single');
dagNet.params(end).value = zeros(256,1,'single');
relu4_left_asymmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu4_left_asymmetric_siamese', relu4_left_asymmetric_siamese, {'cnv4_left_asymmetric_siamese_x'}, {'relu4_left_asymmetric_siamese_x'}, {}) ;
% 4st conv-relu-pool layer - right
cnv4_right_asymmetric_siamese = dagnn.Conv('size', [3 3 256 256], 'pad', 0, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv4_right_asymmetric_siamese', cnv4_right_asymmetric_siamese , {'relu3_right_asymmetric_siamese_x'}, {'cnv4_right_asymmetric_siamese_x'}, {'cnv4_right_asymmetric_siamese_f', 'cnv4_right_asymmetric_siamese_b'});
dagNet.params(end-1).value = dagNet.params(end-3).value;%0.05*randn([3 3 256 256],'single');
dagNet.params(end).value = zeros(256,1,'single');
relu4_right_asymmetric_siamese = dagnn.ReLU() ;
dagNet.addLayer('relu4_right_asymmetric_siamese', relu4_right_asymmetric_siamese, {'cnv4_right_asymmetric_siamese_x'}, {'relu4_right_asymmetric_siamese_x'}, {}) ;

% 5st conv-relu-pool layer - left
cnv5_left_asymmetric_siamese = dagnn.Conv('size', [4 4 256 128], 'pad', 0, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv5_left_asymmetric_siamese', cnv5_left_asymmetric_siamese , {'relu4_left_asymmetric_siamese_x'}, {'cnv5_left_asymmetric_siamese_x'}, {'cnv5_left_asymmetric_siamese_f', 'cnv5_left_asymmetric_siamese_b'});
dagNet.params(end-1).value = 0.05*randn([4 4 256 128],'single');
dagNet.params(end).value = zeros(128,1,'single');
% 5st conv-relu-pool layer - right
cnv5_right_asymmetric_siamese = dagnn.Conv('size', [4 4 256 128], 'pad', 0, 'stride', 1, 'hasBias',true);
dagNet.addLayer('cnv5_right_asymmetric_siamese', cnv5_right_asymmetric_siamese , {'relu4_right_asymmetric_siamese_x'}, {'cnv5_right_asymmetric_siamese_x'}, {'cnv5_right_asymmetric_siamese_f', 'cnv5_right_asymmetric_siamese_b'});
dagNet.params(end-1).value = dagNet.params(end-3).value;%0.05*randn([4 4 256 128],'single');
dagNet.params(end).value = zeros(128,1,'single');

% Normalize features to unit vector - left
L2Normalization_Asymmetric_Left = L2NormalizationLayer();
dagNet.addLayer('L2Normalization_Asymmetric_Left',L2Normalization_Asymmetric_Left,{'cnv5_left_asymmetric_siamese_x'},{'left_asymmetric_Normalized_x'},{});

% Normalize features to unit vector - right
L2Normalization_Asymmetric_Right = L2NormalizationLayer();
dagNet.addLayer('L2Normalization_Asymmetric_Right',L2Normalization_Asymmetric_Right,{'cnv5_right_asymmetric_siamese_x'},{'right_asymmetric_Normalized_x'},{});


l2Dist_Asymmetric_Final = L2DistLayer();
dagNet.addLayer('l2Dist_Asymmetric_Final',l2Dist_Asymmetric_Final,{'left_asymmetric_Normalized_x','right_asymmetric_Normalized_x'},{'l2distAsymmetricOutput'},{});
asymmetric_L2Dist_HingeLossLayer = L2DistLoss('name','asymmetric_L2Dist_HingeLossLayer','hingeThreshold',1);
dagNet.addLayer('asymmetric_L2Dist_HingeLossLayer', asymmetric_L2Dist_HingeLossLayer, {'l2distAsymmetricOutput','labels'}, {'AsymmetricPos','AsymmetricNeg','AsymmetricInfoNeg'}, {});


% %%%%%%%%Symmetric + Asymmetric%%%%%%%%%%%%%%%%

LeftSymmetricAsymmericConcatLayer = dagnn.Concat('dim',3);
dagNet.addLayer('LeftSymmetricAsymmericConcatLayer',LeftSymmetricAsymmericConcatLayer,{'cnv5_left_symmetric_siamese_x','cnv5_left_asymmetric_siamese_x'},{'LeftConcatFeatures'},{});
rightSymmetricAsymmericConcatLayer = dagnn.Concat('dim',3);
dagNet.addLayer('rightSymmetricAsymmericConcatLayer',rightSymmetricAsymmericConcatLayer,{'cnv5_right_symmetric_siamese_x','cnv5_right_asymmetric_siamese_x'},{'RightConcatFeatures'},{});

% final left - Lower features dimensionality from 256 to 128
final_left_siamese = dagnn.Conv('size', [1 1 256 128], 'pad', 0, 'stride', 1, 'hasBias',true);
dagNet.addLayer('final_left_siamese', final_left_siamese , {'LeftConcatFeatures'}, {'final_left_siamese_x'}, {'final_siamese_f', 'final_siamese_b'});
dagNet.params(end-1).value = 0.05*randn([1 1 256 128],'single');
dagNet.params(end).value = zeros(128,1,'single');

% Normalize features to unit vector - left
L2Normalization_Symmetric_Asymmetric_Left = L2NormalizationLayer();
dagNet.addLayer('L2Normalization_Symmetric_Asymmetric_Left',L2Normalization_Symmetric_Asymmetric_Left,{'final_left_siamese_x'},{'left_Normalized_x'},{});

% final right - Lower features dimensionality from 256 to 128
final_right_siamese = dagnn.Conv('size', [1 1 256 128], 'pad', 0, 'stride', 1, 'hasBias',true);
dagNet.addLayer('final_right_siamese', final_right_siamese , {'RightConcatFeatures'}, {'final_right_siamese_x'}, {'final_siamese_f', 'final_siamese_b'});
% dagNet.params(end-1).value = dagNet.params(end-3).value;%0.05*randn([1 1 256 128],'single');
% dagNet.params(end).value = zeros(128,1,'single');

% Normalize features to unit vector - right
L2Normalization_Symmetric_Asymmetric_Right = L2NormalizationLayer();
dagNet.addLayer('L2Normalization_Symmetric_Asymmetric_Right',L2Normalization_Symmetric_Asymmetric_Right,{'final_right_siamese_x'},{'right_Normalized_x'},{});

l2Dist_Final = L2DistLayer();
dagNet.addLayer('l2Dist_Final',l2Dist_Final,{'left_Normalized_x','right_Normalized_x'},{'l2distOutput'},{});

SymmetricAsymmetric_L2Dist_HingeLossLayer = L2DistLoss('name','SymmetricAsymmetric_L2Dist_HingeLossLayer','hingeThreshold',1);
dagNet.addLayer('SymmetricAsymmetric_L2Dist_HingeLossLayer', SymmetricAsymmetric_L2Dist_HingeLossLayer, {'l2distOutput','labels'}, {'HybridPos','HybridNeg','HybridInfoNeg'}, {});
 
if isempty(opts.initializeFrom)
    for i = 1:2:length(dagNet.params)
        dagNet.params(i).learningRate = lr(1);
        dagNet.params(i+1).learningRate = lr(2);
        dagNet.params(i+1).weightDecay = 0;
    end
else
    load(opts.initializeFrom,'net');
    for i = 1:length(net.layers)
        if isequal(net.layers{i}.type,'conv')
            for j = 1:length(dagNet.params)
                if isequal(size(net.layers{i}.weights{1}),size(dagNet.params(j).value))
                    dagNet.params(j).value = net.layers{i}.weights{1};
                    dagNet.params(j).name
                    if isequal(size(net.layers{i}.weights{2}),size(dagNet.params(j+1).value'))
                        dagNet.params(j+1).name
                        dagNet.params(j+1).value = net.layers{i}.weights{2}';
                    end
                end
            end
        end
    end
end

% Meta parameters
dagNet.meta.inputSize = [64 64 1] ;
dagNet.meta.trainOpts.learningRate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,5)] ;
dagNet.meta.trainOpts.weightDecay = 0.0001 ;
dagNet.meta.trainOpts.batchSize = 128 ;
dagNet.meta.trainOpts.numEpochs = numel(dagNet.meta.trainOpts.learningRate) ;


