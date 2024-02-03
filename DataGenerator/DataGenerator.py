from SharedFunctions.utils.general import (
    check_dataset,
    check_yaml,
    colorstr,
)

from SharedFunctions.utils.dataloaders import create_dataloader
import json
import os

class DataLoaderGenerator:
    def __init__(self, json_dir="config.json"):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(current_directory, json_dir), "r") as json_file:
            loaded_config = json.load(json_file)
            self.data = check_yaml(loaded_config["data"])
            self.batch_size = loaded_config["batch_size"]
            self.imgsz = loaded_config["imgsz"]
            self.task = loaded_config["task"] 
            self.workers = loaded_config["workers"]
            self.single_cls = loaded_config["single_cls"]
            self.stride = loaded_config["stride"]
            self.pt = loaded_config["pt"]

    def run(self):
        data = check_dataset(self.data)
        pad, rect = (0.0, False) if self.task == "speed" else (0.5, self.pt)  # square inference for benchmarks
        task = self.task if self.task in ("train", "val", "test") else "val"  # path to train/val/test images
        return create_dataloader(
            data[task],
            self.imgsz,
            self.batch_size,
            self.stride,
            self.single_cls,
            pad=pad,
            rect=rect,
            workers=self.workers,
            prefix=colorstr(f"{task}: "),
        )[0], data["nc"]