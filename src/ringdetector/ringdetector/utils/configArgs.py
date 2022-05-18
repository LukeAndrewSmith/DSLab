import numpy as np
from ringdetector.Paths import CROP_MODEL

def getArgs(parser):
    """ Arg handler for Core Analysis
    """
    
    ###########################################################################
    # Logging and Wands
    ###########################################################################
    parser.add_argument('-wb', '--wandb', dest='wb', action='store_true')
    parser.add_argument('-wbn', dest='wbname', default="", type=str, 
        help="Name of wandb run.")

    ###########################################################################
    # Dataset
    ########################################################################### 
    #NOTE: placeholder for a potential future arg for datasets, right now
    # everything is hard coded for inner.
    #parser.add_argument('-dataset', type=str, choices=['inner'],
    #    default='inner', help="Which dataset to load.")
    parser.add_argument("-sample", type=str, default="",
        help="Specify sampleName, e.g. RUEP03WW, to run only on one sample.")
    parser.add_argument("-n", type=int, default=None,
        help="Run on n samples.")
    
    ###########################################################################
    # Ring Processor
    ########################################################################### 
    #TODO: experiment with args, pick best default
    parser.add_argument("-imReadType", type=str, choices=['grayscale', 'hsv'],
        default='hsv', help="RingDetection: grayscale or hsv value component before processing")

    parser.add_argument("-denoiseH", type=int, default=10, 
        help="RingDetection: h arg for cv2.fastNlMeansDenoising")
    parser.add_argument("-denoiseTemplateWindowSize", type=int, default=7, 
        help="RingDetection: templateWindowSize arg for cv2.fastNlMeansDenoising")
    parser.add_argument("-denoiseSearchWindowSize", type=int, default=21, 
        help="RingDetection: searchWindowSize arg for cv2.fastNlMeansDenoising")

    parser.add_argument("-cannyMin", type=int, default=50, 
        help="RingDetection: Minimum threshold (threshold1 arg) for cv2.Canny")
    parser.add_argument("-cannyMax", type=int, default=100, 
        help="RingDetection: Maximum threshold (threshold2 arg) for cv2.Canny")
    
    parser.add_argument("-rightEdgeMethod", type=str, choices=['simple', 'angle', 'counts'], default='simple', 
        help="RingDetection: 'Keep right edge' heuristic method for choosing of shapes to apply to")

    parser.add_argument("-invertedEdgeWindowSize", type=int, default=25, 
        help="RingDetection: Number of pixels left and right of an edge to use to determine the average color right/left of a shape")
    
    parser.add_argument("-mergeShapes1Ball", type=tuple, default=(10,5), 
        help="RingDetection: Distance between shape tips allowed for merging")
    parser.add_argument("-mergeShapes1Angle", type=float, default=np.pi/4, 
        help="RingDetection: Angle between shapes allowed for merging")
    
    parser.add_argument("-mergeShapes2Ball", type=tuple, default=(20,20), 
        help="RingDetection: Distance between shape tips allowed for merging")
    parser.add_argument("-mergeShapes2Angle", type=float, default=np.pi/4, 
        help="RingDetection: Angle between shapes allowed for merging")
    
    parser.add_argument("-filterLengthImgProportion", type=float, default=0.5, 
        help="RingDetection: Propotion of image height under which a shape of lower length is discarded")
    
    parser.add_argument("-filterRegressionAnglesAngleThreshold", type=float, default=np.pi/4, 
        help="RingDetection: Angle for which a difference between angles of shape regressions is considered an anomaly")
    

    ###########################################################################
    # Edge Processor and Edges
    ########################################################################### 
    parser.add_argument("-minedgelen", type=int, default=80, 
        help="EdgeProcessor: Minimum threshold (threshold1 arg) for cv2.Canny")
    parser.add_argument("-edgemodel", type=str, choices=["linear"],
        default="linear", help="Edge: model type for edge fitting.")
    
    cfg = parser.parse_args()

    ###########################################################################
    # Additional logic
    ###########################################################################
    # turn on wandb automatically if running on entire dataset
    #if not cfg.sample:
    #    cfg.wb = True

    return cfg


def getCropDetectionArgs(parser):
    parser.add_argument("--mode", "-m", dest="mode", default="pred", choices=["train", "eval", "pred"],
                        type=str, help="mode to run the object detection in: train eval or prediction")
    parser.add_argument("--dataMode", "-tm", dest="dataMode", choices=["inner", "outer", "outerInner"],
                        default="inner", type=str, help="inner or outer rectangle to detect?")
    parser.add_argument("-split", dest="split", help="What split to predict on if mode pred or eval is chosen",
                        choices=["train", "val", "test"], type=str)
    # TODO add vis mode and an actual inference mode
    parser.add_argument("--k-pred", "-k", dest="k", default=5, type=int, help="amount of visualizations of "
                                                                              "predictions made by the pred mode")
    parser.add_argument("--num-gpus", dest="num-gpus", type=int)
    parser.add_argument("--cracks", dest="cracks", action='store_true', default=False, help="if set: also includes " \
                                                                                           "cracks and gaps in the data")

    #### INFERENCE #####
    parser.add_argument("--modelPath", dest="modelPath", type=str, help="absolute path to the model that you want to "
                                                                        "evaluate or predict with",
                                                                        default=CROP_MODEL)

    parser.add_argument("--imgPath", dest="imgPath", type=str, help="relative path of the image you want to predict on, based from the data folder."
                                                                    "example: images/KunL11-20.jpg")

    parser.add_argument("--csvPath", dest="csvPath", type=str, help="absolute path to the csv file that contains the core information",
                        default = None)
                        
    
  

    cfg = parser.parse_args()

    return cfg
