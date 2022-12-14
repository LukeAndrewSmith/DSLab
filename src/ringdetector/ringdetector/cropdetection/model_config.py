from tty import CFLAG
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN

##see: https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py
##TODO(2): model_config refactoring
##TODO(2): config argparse
##TODO(2): rotated rect prediction: https://github.com/facebookresearch/detectron2/issues/21, 
##see https://colab.research.google.com/drive/1JXKl48u1fxC35bBryKlQVyQf8tp-DUpE?usp=sharing for a possible solution


def generate_config(output_dir, dataset_train, dataset_test):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

    ## ====Dataset====
    cfg.DATASETS.TRAIN = dataset_train
    cfg.DATASETS.TEST = dataset_test

    ## ====FORMACM1====
    cfg.MODEL.DEVICE = "cpu"

    ## ====Dataloader====
    # Number of data loading threads
    cfg.DATALOADER.NUM_WORKERS = 4
    # # If True, each batch should contain only images for which the aspect ratio
    # # is compatible. This groups portrait images together, and landscape images
    # # are not batched with portrait images.
    # cfg.DATALOADER.ASPECT_RATIO_GROUPING = True
    # # Options: TrainingSampler, RepeatFactorTrainingSampler
    # cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
    # # Repeat threshold for RepeatFactorTrainingSampler
    # cfg.DATALOADER.REPEAT_THRESHOLD = 0.0
    # # Tf True, when working on datasets that have instance annotations, the
    # # training dataloader will filter out images without associated annotations
    # cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

    ## ====Model:Anchors====TODO(1)
    # # The generator can be any name in the ANCHOR_GENERATOR registry
    # cfg.MODEL.ANCHOR_GENERATOR.NAME = "DefaultAnchorGenerator"
    
    # # Anchor sizes (i.e. sqrt of area) in absolute pixels w.r.t. the network input.
    # # Format: list[list[float]]. SIZES[i] specifies the list of sizes to use for
    # # IN_FEATURES[i]; len(SIZES) must be equal to len(IN_FEATURES) or 1.
    # # When len(SIZES) == 1, SIZES[0] is used for all IN_FEATURES.
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[256, 384, 512]]
    # # Anchor aspect ratios. For each area given in `SIZES`, anchors with different aspect
    # # ratios are generated by an anchor generator.
    # # Format: list[list[float]]. ASPECT_RATIOS[i] specifies the list of aspect ratios (H/W)
    # # to use for IN_FEATURES[i]; len(ASPECT_RATIOS) == len(IN_FEATURES) must be true,
    # # or len(ASPECT_RATIOS) == 1 is true and aspect ratio list ASPECT_RATIOS[0] is used
    # # for all IN_FEATURES.
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1/75, 1/80, 1/85, 1/90, 1/100, 1/125]]##NOTE: Input core size: (250*20000)*0.15 = 37.5*3000
    
    # # Anchor angles.
    # # list[list[float]], the angle in degrees, for each input feature map.
    # # ANGLES[i] specifies the list of angles for IN_FEATURES[i].
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]
    
    # # Relative offset between the center of the first anchor and the top-left corner of the image
    # # Value has to be in [0, 1). Recommend to use 0.5, which means half stride.
    # # The value is not expected to affect model accuracy.
    cfg.MODEL.ANCHOR_GENERATOR.OFFSET = 0.0


    ## ====Model:RPN Network====
    # cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"  # used by RPN_HEAD_REGISTRY
    
    # # Remove RPN anchors that go outside the image by BOUNDARY_THRESH pixels
    # # Set to -1 or a large value, e.g. 100000, to disable pruning anchors
    # cfg.MODEL.RPN.BOUNDARY_THRESH = -1

    #  # IOU overlap ratios [BG_IOU_THRESHOLD, FG_IOU_THRESHOLD]
    # # Minimum overlap required between an anchor and ground-truth box for the
    # # (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
    # # ==> positive RPN example: 1)
    # # Maximum overlap allowed between an anchor and ground-truth box for the
    # # (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
    # # ==> negative RPN example: 0)
    # # Anchors with overlap in between (BG_IOU_THRESHOLD <= IoU < FG_IOU_THRESHOLD)
    # # are ignored (-1)
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
    cfg.MODEL.RPN.IOU_LABELS = [0, -1, 1]
    
    # Number of regions per image used to train RPN
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512
    
    # # Target fraction of foreground (positive) examples per RPN minibatch
    # cfg.MODEL.RPN.POSITIVE_FRACTION = 0.5
    
    # # Options are: "smooth_l1", "giou", "diou", "ciou"
    # cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"
    # cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1.0
    # # Weights on (dx, dy, dw, dh) for normalizing RPN anchor regression targets
    # cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    # # The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
    # cfg.MODEL.RPN.SMOOTH_L1_BETA = 0.0
    # cfg.MODEL.RPN.LOSS_WEIGHT = 1.0

    # # Number of top scoring RPN proposals to keep before applying NMS
    # # When FPN is used, this is *per FPN level* (not total)
    # cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    # cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    
    # # Number of top scoring RPN proposals to keep after applying NMS
    # # When FPN is used, this limit is applied per level and then again to the union
    # # of proposals from all levels
    # # NOTE: When FPN is used, the meaning of this config is different from Detectron1.
    # # It means per-batch topk in Detectron1, but per-image topk here.
    # # See the "find_top_rpn_proposals" function for details.
    # cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    # cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    
    # # NMS threshold used on RPN proposals
    cfg.MODEL.RPN.NMS_THRESH = 0.0
   
    # # Set this to -1 to use the same number of output channels as input channels.
    # cfg.MODEL.RPN.CONV_DIMS = [-1]
    

    ## ====Model:ROI====
    # cfg.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # only has one class.

    # # Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
    # # Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    
    # RoI minibatch size *per image* (number of regions of interest [ROIs]) during training
    # Total number of RoIs per training minibatch = ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
    # E.g., a common configuration is: 512 * 16 = 8192
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    
    # # Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.30

    # Only used on test mode
    # Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
    # balance obtaining high recall with not having too many low precision
    # detections that will slow down inference post processing steps (like NMS)
    # A default threshold of 0.0 increases AP by ~0.2-0.3 but significantly slows down
    # inference.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    
    # # Overlap threshold used for non-maximum suppression (suppress boxes with
    # # IoU >= this threshold)
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.0


    ## ====Model:BoxHead====
    # # C4 don't use head name option
    # # Options for non-C4 models: FastRCNNConvFCHead,
    # cfg.MODEL.ROI_BOX_HEAD.NAME = ""
    
    # # Options are: "smooth_l1", "giou", "diou", "ciou"
    # cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "smooth_l1"
    # # The final scaling coefficient on the box regression loss, used to balance the magnitude of its
    # # gradients with other losses in the model. See also `MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT`.
    # cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 1.0
    # # Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
    # # These are empirically chosen to approximately lead to unit variance targets
    # cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
    # # The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
    # cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.0
    # cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
    # cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
    # # Type of pooling operation applied to the incoming feature map for each RoI
    # cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"

    # cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 0
    # # Hidden layer dimension for FC layers in the RoI box head
    # cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 1024
    # cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 0
    # # Channel dimension for Conv layers in the RoI box head
    # cfg.MODEL.ROI_BOX_HEAD.CONV_DIM = 256
    # # Normalization method for the convolution layers.
    # # Options: "" (no norm), "GN", "SyncBN".
    # cfg.MODEL.ROI_BOX_HEAD.NORM = ""
    # # Whether to use class agnostic for bbox regression
    # cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = False
    # # If true, RoI heads use bounding boxes predicted by the box head rather than proposal boxes.
    # cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = False

    # # Federated loss can be used to improve the training of LVIS
    # cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
    # # Sigmoid cross entrophy is used with federated loss
    # cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    # # The power value applied to image_count when calcualting frequency weight
    # cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER = 0.5
    # # Number of classes to keep in total
    # cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 50


    ## ====Solver====TODO(2)
    # # Options: WarmupMultiStepLR, WarmupCosineLR.
    # # See detectron2/solver/build.py for definition.
    # cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

    cfg.SOLVER.MAX_ITER = 5000

    cfg.SOLVER.BASE_LR = 0.0025
    # # The end lr, only used by WarmupCosineLR
    # cfg.SOLVER.BASE_LR_END = 0.0
    # cfg.SOLVER.MOMENTUM = 0.9
    # cfg.SOLVER.NESTEROV = False
    # cfg.SOLVER.WEIGHT_DECAY = 0.0001
    # # The weight decay that's applied to parameters of normalization layers
    # # (typically the affine transformation)
    # cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    # cfg.SOLVER.GAMMA = 0.1
    # The iteration number to decrease learning rate by GAMMA.
    cfg.SOLVER.STEPS = [1000,2000,3000,4000,4500]
    # cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    # cfg.SOLVER.WARMUP_ITERS = 1000
    # cfg.SOLVER.WARMUP_METHOD = "linear"

    # Save a checkpoint after every this number of iterations
    cfg.SOLVER.CHECKPOINT_PERIOD = 500

    # Number of images per batch across all machines. This is also the number
    # of training images per step (i.e. per iteration). If we use 16 GPUs
    # and IMS_PER_BATCH = 32, each GPU will see 2 images per batch.
    # May be adjusted automatically if REFERENCE_WORLD_SIZE is set.
    cfg.SOLVER.IMS_PER_BATCH = 1

    # # The reference number of workers (GPUs) this config is meant to train with.
    # # It takes no effect when set to 0.
    # # With a non-zero value, it will be used by DefaultTrainer to compute a desired
    # # per-worker batch size, and then scale the other related configs (total batch size,
    # # learning rate, etc) to match the per-worker batch size.
    # # See documentation of `DefaultTrainer.auto_scale_workers` for details:
    # cfg.SOLVER.REFERENCE_WORLD_SIZE = 0

    # # Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
    # # biases. This is not useful (at least for recent models). You should avoid
    # # changing these and they exist only to reproduce Detectron v1 training if
    # # desired.
    # cfg.SOLVER.BIAS_LR_FACTOR = 1.0
    # cfg.SOLVER.WEIGHT_DECAY_BIAS = None  # None means following WEIGHT_DECAY

    # Gradient clipping
    # cfg.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
    # # Type of gradient clipping, currently 2 values are supported:
    # # - "value": the absolute values of elements of each gradients are clipped
    # # - "norm": the norm of the gradient for each parameter is clipped thus
    # #   affecting all elements in the parameter
    # cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    # # Maximum absolute value used for clipping gradients
    # cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    # # Floating point number p for L-p norm to be used with the "norm"
    # # gradient clipping type; for L-inf, please specify .inf
    # cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

    # # Enable automatic mixed precision for training
    # # Note that this does not change model's inference behavior.
    # # To use AMP in inference, run inference under autocast()
    # cfg.SOLVER.AMP = CN({"ENABLED": False})


    ## ===Test===
    # The period (in terms of steps) to evaluate the model during training.
    # Set to 0 to disable.
    cfg.TEST.EVAL_PERIOD = 100
    # Maximum number of detections to return per image during inference (100 is
    # based on the limit established for the COCO dataset).
    cfg.TEST.DETECTIONS_PER_IMAGE = 50

    # cfg.TEST.AUG = CN({"ENABLED": False})
    # cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
    # cfg.TEST.AUG.MAX_SIZE = 4000
    # cfg.TEST.AUG.FLIP = True

    # cfg.TEST.PRECISE_BN = CN({"ENABLED": False})
    # cfg.TEST.PRECISE_BN.NUM_ITER = 200


    ## ===Misc===
    # Directory where output files are written
    cfg.OUTPUT_DIR = output_dir
    
    # Set seed to negative to fully randomize everything.
    # Set seed to positive to use a fixed seed. Note that a fixed seed increases
    # reproducibility but does not guarantee fully deterministic behavior.
    # Disabling all parallelism further increases reproducibility.
    cfg.SEED = 12
    
    # # Benchmark different cudnn algorithms.
    # # If input images have very different sizes, this option will have large overhead
    # # for about 10k iterations. It usually hurts total time, but can benefit for certain models.
    # # If input images have the same or similar sizes, benchmark is often helpful.
    # cfg.CUDNN_BENCHMARK = False
    
    # The period (in terms of steps) for minibatch visualization at train time.
    # Set to 0 to disable.
    cfg.VIS_PERIOD = 100

    return cfg