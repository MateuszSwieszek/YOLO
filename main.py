from DataGenerator.DataGenerator import DataLoaderGenerator
from YOLOPipeline.Pipeline import Pipeline
import torch

def main():
    data_gen, num_classes = DataLoaderGenerator().run()
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device="cpu")
    print(model.model.device)
    Pipeline(data_gen, model, num_classes, "cpu", save_dir="runs/").run()


if __name__ == '__main__':
    main()