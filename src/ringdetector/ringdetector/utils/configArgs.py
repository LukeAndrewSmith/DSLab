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
        default='hsv', help="EdgeDetection: grayscale or hsv value component before processing")
    parser.add_argument("-denoiseh", type=int, default=10, 
        help="EdgeDetection: h arg for cv2.fastNlMeansDenoising")
    parser.add_argument("-denoisetempwind", type=int, default=7, 
        help="EdgeDetection: templateWindowSize arg for cv2.fastNlMeansDenoising")
    parser.add_argument("-denoisesearchwind", type=int, default=21, 
        help="EdgeDetection: searchWindowSize arg for cv2.fastNlMeansDenoising")
    parser.add_argument("-ipgrad", type=str, default="canny", 
        choices=["canny", "sobel"], 
        help="EdgeDetection: gradient-based edge detection method")
    parser.add_argument("-sobelksize", type=int, default=3, 
        help="EdgeDetection: Ksize arg for cv2.Sobel")
    parser.add_argument("-cannymin", type=int, default=50, 
        help="EdgeDetection: Minimum threshold (threshold1 arg) for cv2.Canny")
    parser.add_argument("-cannymax", type=int, default=100, 
        help="EdgeDetection: Maximum threshold (threshold2 arg) for cv2.Canny")

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
