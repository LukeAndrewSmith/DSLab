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
    # Image Processor
    ########################################################################### 
    #TODO: experiment with args, pick best default
    parser.add_argument("-ipread", type=str, choices=['grayscale', 'hsv'],
        default='hsv', help="ImageProcessor: grayscale or hsv value component before processing")
    parser.add_argument("-denoiseh", type=int, default=10, 
        help="ImageProcessor: h arg for cv2.fastNlMeansDenoising")
    parser.add_argument("-denoisetempwind", type=int, default=7, 
        help="ImageProcessor: templateWindowSize arg for cv2.fastNlMeansDenoising")
    parser.add_argument("-denoisesearchwind", type=int, default=21, 
        help="ImageProcessor: searchWindowSize arg for cv2.fastNlMeansDenoising")
    parser.add_argument("-ipgrad", type=str, default="canny", 
        choices=["canny", "sobel"], 
        help="ImageProcessor: gradient-based edge detection method")
    parser.add_argument("-sobelksize", type=int, default=3, 
        help="ImageProcessor: Ksize arg for cv2.Sobel")
    parser.add_argument("-cannymin", type=int, default=50, 
        help="ImageProcessor: Minimum threshold (threshold1 arg) for cv2.Canny")
    parser.add_argument("-cannymax", type=int, default=100, 
        help="ImageProcessor: Maximum threshold (threshold2 arg) for cv2.Canny")

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
    parser.add_argument("--mode", "-m", dest="mode", default="train", choices=["train", "eval", "pred"],
                        type=str, help="mode to run the object detection in: train eval or prediction")
    parser.add_argument("--dataMode", "-tm", dest="dataMode", choices=["inner", "outer", "outerInner"],
                        default="inner", type=str, help="inner or outer rectangle to detect?")
    parser.add_argument("--modelPath", dest="modelPath", type=str, help="absolute path to the model that you want to "
                                                                        "evaluate or predict with")
    parser.add_argument("-split", dest="split", help="What split to predict on if mode pred or eval is chosen",
                        choices=["train", "val", "test"], type=str)
    # TODO add vis mode and an actual inference mode
    parser.add_argument("--k-pred", "-k", dest="k", default=5, type=int, help="amount of visualizations of "
                                                                              "predictions made by the pred mode")
    parser.add_argument("--num-gpus", dest="num-gpus", type=int)
    parser.add_argument("--cracks", dest="cracks", action='store_true', default=False, help="if set: also includes " \
                                                                                           "cracks and gaps in the data")

    cfg = parser.parse_args()

    return cfg
