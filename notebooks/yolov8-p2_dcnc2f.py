import shutil
import os
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from ultralytics.utils.loss_logger import log_loss_config
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    dataset_path = 'D:/deep-learning/datasets/cctsdb2021-full'
    yolo_repo = ".."
    project_name = "YOLOv8s Traffic Detection"
    output_path = f"./runs/detect/{project_name}"
    yolo_mod_path = f'{yolo_repo}/custom'
    ultralytics_path = "D:/deep-learning/ultralytics/ultralytics"

    yolo_path = f'{yolo_repo}/cfg/yolov8s-dcnc2f.yaml'
    model_name = "YOLOv8s + DCN-C2f"


    # conv.py -- ultralytics/nn/modules/
    shutil.copy(f'{yolo_mod_path}/nn/modules/conv.py', 
                f'{ultralytics_path}/nn/modules/conv.py')

    # block.py -- ultralytics/nn/modules/
    shutil.copy(f'{yolo_mod_path}/nn/modules/block.py', 
                f'{ultralytics_path}/nn/modules/block.py')

    # __init__.py -- ultralytics/nn/modules/
    shutil.copy(f'{yolo_mod_path}/nn/modules/__init__.py', 
                f'{ultralytics_path}/nn/modules/__init__.py')

    # spconv.py -- ultralytics/nn/
    shutil.copy(f'{yolo_mod_path}/nn/modules/spconv.py',
                f'{ultralytics_path}/nn/modules/spconv.py')

    # dcnc2f.py -- ultralytics/nn/
    shutil.copy(f'{yolo_mod_path}/nn/modules/dcn.py',
                f'{ultralytics_path}/nn/modules/dcn.py')

    # tasks.py -- ultralytics/nn/
    shutil.copy(f'{yolo_mod_path}/nn/tasks.py', 
                f'{ultralytics_path}/nn/tasks.py')

    # loss.py -- ultralytics/utils/
    shutil.copy(f'{yolo_mod_path}/utils/loss.py',
                f'{ultralytics_path}/utils/loss.py')

    # loss_logger.py -- ultralytics/utils/
    shutil.copy(f'{yolo_mod_path}/utils/loss_logger.py',
                f'{ultralytics_path}/utils/loss_logger.py')

    SETTINGS['wandb'] = True
    SETTINGS['tensorboard'] = True

    configs = {
        "data": dataset_path,
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "amp": True,
        "device": "cuda:0",
        "optimizer": 'SGD',
        "lr0": 0.01,
        "workers": 4,
        "patience": 15,
        "val": True,
        "project": project_name
    }

    os.environ['YOLO_IOU_LOSS']        = 'CIoU'
    os.environ['YOLO_IOU_MONOTONOUS']  = 'none'

    # from wandb.integration.ultralytics import add_wandb_callback
    # with wandb.init(project=configs["project"], name=model_name, job_type="train") as run:
    model = YOLO(yolo_path, task='detect', verbose=True)
    model.add_callback('on_pretrain_routine_end', log_loss_config)

    model_results = model.train(
        data    = configs["data"],
        epochs  = configs["epochs"],
        imgsz   = configs["imgsz"],
        batch   = configs["batch"], 
        device  = configs["device"],
        amp     = configs["amp"],
        workers      = configs["workers"],
        optimizer    = configs["optimizer"],
        lr0          = configs["lr0"],
        project      = configs["project"],
        name         = f"{model_name}/train",
        exist_ok     = True,
        save_period  = 10,
        verbose      = True,
        patience     = configs["patience"],
        val          = configs["val"]
    )
