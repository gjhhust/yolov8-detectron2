from detectron2.config import LazyCall as L
from omegaconf import OmegaConf
from utils.get_optimizer import get_optimizer
from .datasets.builder import build_train_loader, build_test_loader, build_evaluator
# from .models.model_utils import get_model
from .models.yolo import DetectionModel

# build dataloader
dataloader = OmegaConf.create()
dataloader.train = L(build_train_loader)(
    batch_size = 1,
    num_workers = 1,
    is_distributed = False,
    no_aug  = False,
    data_dir = '/data1/jiahaoguo/dataset/gaode_3_coco',
    image_dir = "images",
    json_file = "debug.json",
    input_size = (640, 640),
    degrees = 1.0,
    translate = 0.1,
    scale = (0.1, 0.6),
    shear = 1.0,
    perspective = 0.0,
    enable_mixup = True,
)
dataloader.test = L(build_test_loader)(
    data_dir = '/data1/jiahaoguo/dataset/gaode_3_coco',
    image_dir = "images",
    json_file = "test.json",
    test_size = (640, 640),
    infer_batch = 1,
    test_num_workers  = 4,
    max_dets = 50
)
dataloader.evaluator = L(build_evaluator)(
    output_folder  = None,
    max_dets = 50
)
 
# build model
model = L(DetectionModel)(
    cfg='yolov8s.yaml',  
    ch=3,  
    nc=1,  
    cls_idx = [0,], 
    conf=0.001, 
    iou=0.7, 
    agnostic=False, 
    multi_label=False, 
    num_max_dets=100, 
    verbose=True
)
 
# build optimizer
optimizer = L(get_optimizer)(
    batch_size = 16, 
    basic_lr_per_img = 0.01 / 64.0, 
    model = None, 
    momentum = 0.937, 
    weight_decay = 5e-4, 
    warmup_epochs = 1, 
    warmup_lr_start = 0 
)

# build LR
lr_cfg = dict(
    train_batch_size = 16, 
    basic_lr_per_img = 0.01 / 64.0, 
    scheduler_name = "yoloxwarmcos", 
    iters_per_epoch = 1769, 
    max_eps = 80, 
    num_warmup_eps = 1,
    warmup_lr_start = 0,
    no_aug_eps = 10,
    min_lr_ratio = 0.01
)

# bs = 16,  1eps = 1769 iter   
# build trainer
train = dict(
    output_dir="./yolov8_s",
    init_checkpoint="",
    max_iter = 1769 * 80 ,
    start_iter = 0,
    seed = 0,
    random_size = (20, 36), 
    amp=dict(enabled=True),  # options for Automatic Mixed Precision
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
    ),
    # model ema
    model_ema = dict(
        enabled=True,
        use_ema_weights_for_eval_only = True,
        decay = 0.9998,
        device = "cuda",
        after_backward = False
    ),
    checkpointer=dict(period=1769 * 2, max_to_keep=5),  # options for PeriodicCheckpointer
    eval_period = 2,
    log_period = 20,
    device="cuda"
)


    




