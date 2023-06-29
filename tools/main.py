import os
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description='Yolov8 main')


parser.add_argument('--run',
                    default='detect',
                    type=str,
                    help='run train or detect or export')
args = parser.parse_args()

def train(config_file_path, num_class, pretrain=None):
    if pretrain is None:
        yolo = YOLO("yolov8n.yaml", num_class=num_class)
    else:
        yolo = YOLO(pretrain)
    yolo.train(cfg=config_file_path)


def detect(model_path, source):
    yolo = YOLO(model_path)
    source_par_dir = os.path.dirname(source)
    source_name = os.path.basename(source)
    yolo.predict(source, save=True, project=source_par_dir, name='det_'+source_name)


def export(model_path):
    yolo = YOLO(model_path)
    yolo.export(format='onnx')


if __name__ == '__main__':
    # ######################## config options ###########################
    project_name = ''
    version = 0
    num_class = 2

    ####################### # run train ## ##############################
    if args.run == 'train':
        config_file_path = f'Projects/{project_name}/configs/v{version}.yaml'
        pretrain = None
        assert os.path.exists(config_file_path)
        train(config_file_path, num_class, pretrain)


    ####################### # run train ## ##############################
    if args.run == 'detect':
        model_path = f'Projects/{project_name}/log/v{version}/train/weights/best.pt'
        source = ''
        assert os.path.exists(model_path)
        detect(model_path, source)

    ####################### # run train ## ##############################
    if args.run == 'export':
        model_path = f'Projects/{project_name}/log/v{version}/train/weights/best.pt'
        assert os.path.exists(model_path)
        export(model_path)