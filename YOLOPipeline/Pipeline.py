from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
from YOLOPipeline.Profile import Profile
import logging
import logging.config
import os
from pathlib import Path
import numpy as np
from SharedFunctions.utils.general import (
    LOGGER,
    increment_path,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from SharedFunctions.utils.metrics import box_iou, ap_per_class
from SharedFunctions.utils.plots import output_to_target, plot_images
from PerfMonitor.PerfMonitor import PerformaceMonitor



class Pipeline:
    def __init__(self, dataloader, model, num_classes, device="cpu", enable_profile=True, half=True, plots=True, save_dir=None):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        tqdm_nar_format = "{l_bar}{bar:10}{r_bar}" 
        tqdm_desc = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
        self.pbar = tqdm(dataloader, desc=tqdm_desc, bar_format=tqdm_nar_format) 
        self.enable_profile = enable_profile

        self.half = half
        self.logger = self.set_logging()
        self.plots = plots
        self.save_dir = save_dir

        self.preprocessing_profiler = Profile(device=device)
        self.postprocessing_profiler = Profile(device=device)
        self.inference_profiler = Profile(device=device)
        self.metrics_profiler = Profile(device=device)
        self.num_classes = num_classes
        self.seen = 0
        project=Path(os.getcwd(), "runs/run/")
        

        self.save_dir = increment_path(project, exist_ok=False)  
        self.perf_monitor = PerformaceMonitor(self.save_dir)


    def run(self):
        iouv = torch.linspace(0.5, 0.95, 10, device=self.device)  
        niou = iouv.numel()
        jdict, stats, ap, ap_class = [], [], [], []
        names = self.model.names if hasattr(self.model, "names") else self.model.module.names  
        self.model.eval()
        
        for batch_i, (im, targets, paths, shapes) in enumerate(self.pbar):
            with self.preprocessing_profiler:
                im, targets = self.preprocess_data(im, targets)
                nb, _, height, width = im.shape  

            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                    with record_function("model_inference"):
                        with self.inference_profiler:
                            print(im.shape)
                            preds = self.model(im)
            self.perf_monitor.accumulate_cpu_runtime(prof)


            with self.postprocessing_profiler:
                preds = self.postprocessing(targets, preds, width, height)
                
                # Metrics
                for si, pred in enumerate(preds):
                    labels = targets[targets[:, 0] == si, 1:]
                    nl, npr = labels.shape[0], pred.shape[0]  
                    path, shape = Path(paths[si]), shapes[si][0]
                    correct = torch.zeros(npr, niou, dtype=torch.bool, device=self.device)  
                    self.seen += 1

                    if npr == 0:
                        if nl:
                            stats.append((correct, *torch.zeros((2, 0), device=self.device), labels[:, 0]))
                        continue

                    # Predictions
                    predn = pred.clone()
                    scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1]) 

                    # Evaluate
                    if nl:
                        tbox = xywh2xyxy(labels[:, 1:5])  
                        scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  
                        labelsn = torch.cat((labels[:, 0:1], tbox), 1)  
                        correct = self.process_batch(predn, labelsn, iouv)
                    stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  


                # Plot images
                if self.plots and batch_i < 2 and self.save_dir!= None:
                    plot_images(im, targets, paths, self.save_dir / f"val_batch{batch_i}_labels.jpg", names)  # labels
                    plot_images(im, output_to_target(preds), paths, self.save_dir / f"val_batch{batch_i}_pred.jpg", names)  # pred
        
        with self.metrics_profiler:
            # Compute metrics
            stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  
            if len(stats) and stats[0].any():
                tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=self.plots, save_dir=self.save_dir, names=names)
                ap50, ap = ap[:, 0], ap.mean(1)  
                mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(int), minlength=self.num_classes)  
                
            pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format

            for i, c in enumerate(ap_class):
                LOGGER.info(pf % (names[c], self.seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        self.perf_monitor.add_profilers(self.preprocessing_profiler, self.postprocessing_profiler, self.inference_profiler, self.metrics_profiler)
        self.perf_monitor.generate_reports()



    def postprocessing(self, targets, preds, width, height):

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=self.device)  # to pixels
        lb = []  
        preds = non_max_suppression(
            preds, 0.001, 0.6, labels=lb, multi_label=True, agnostic=False, max_det=300
        )
        return preds


    def process_batch(self, detections, labels, iouv):
        """
        Return correct prediction matrix.

        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(iouv)):
            x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

    def preprocess_data(self, im, targets):
        im = im.to(self.device, non_blocking=True)
        targets = targets.to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        return im, targets
    
    def set_logging(self, name="inference", verbose=True):
        # sets up logging for the given name
        rank = int(os.getenv("RANK", -1))  # rank in world for Multi-GPU trainings
        level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
        logging.config.dictConfig(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {name: {"format": "%(message)s"}},
                "handlers": {
                    name: {
                        "class": "logging.StreamHandler",
                        "formatter": name,
                        "level": level,
                    }
                },
                "loggers": {
                    name: {
                        "level": level,
                        "handlers": [name],
                        "propagate": False,
                    }
                },
            }
        )
        return logging.getLogger(name) 
    