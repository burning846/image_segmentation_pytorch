import os
import yaml
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
import logging
from datetime import datetime
import json


from datasets.face_det import FaceDetectionDataset
from utils.model_utils import load_model
from utils.criterion_utils import build_criterion, calculate_loss, calculate_accuracy_detection
from utils.image_processing_utils import build_preproc
from utils.optimizer_utils import build_optimizer
from utils.scheduler_utils import build_scheduler
from models.prior_box import PriorBox
from utils.data_utils import build_dataset


class Trainer(object):
    def __init__(self, config_path) -> None:
        self.config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

        # 信息记录
        self.experiment_name = self.config['experiment']['experiment_name']
        self.version = self.config['experiment']['version']
        self.model_name = self.config['model']['name']
        self.checkpoint_dir = self.config['train']['checkpoint_dir']
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_{self.version}_{current_time}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 创建一个文件处理器，将日志输出到文件
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.checkpoint_dir, 'train.log'))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info(json.dumps(self.config))
        self.logger.info(self.config['experiment']['description'])

        # 也可以写到utils里
        self.train_dataset, self.train_dataloader = build_dataset(self.config, context='train')
        self.val_dataset, self.val_dataloader = build_dataset(self.config, context='val')
        self.test_dataset, self.test_dataloader = build_dataset(self.config, context='test')

        self.device = self.config['environment']['device']
        self.model = load_model(self.config).to(self.device)
        self.criterion, criterion_params = build_criterion(self.config, self.device)
        self.optimizer = build_optimizer(self.config, self.model, criterion_params)
        self.scheduler = build_scheduler(self.config, self.optimizer)

        self.save_frequency = self.config['train']['save_frequency']

    def train(self, verbose=False):
        self.model.train()
        min_loss = np.inf
        for epoch in range(self.config['train']['epochs']):
            progress_bar = tqdm(enumerate(self.train_dataloader))
            total_loss = 0
            for i, (inputs, labels) in progress_bar:
                # 前向传播
                inputs = {k:v.to(self.device) for k, v in inputs.items()}
                labels = {k:v.to(self.device) for k, v in labels.items()}

                outputs = self.model(inputs)
                loss, loss_detail = calculate_loss(outputs, labels, self.criterion)
                total_loss += loss.item()

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 打印训练信息
                if verbose:
                    loss_info = ' '.join([f"{name}: {value.item():.4f}" for name, value in loss_detail.items()])
                    progress_bar.set_description(f'Epoch [{epoch+1}/{self.config["train"]["epochs"]}], {loss_info}, Loss: {total_loss/(i+1):.4f}')
                
                self.logger.info(f'Train -- Epoch [{epoch+1}/{self.config["train"]["epochs"]}], Loss: {total_loss / (i+1):.4f}')

            # 更新学习率
            self.scheduler.step()

            # 验证模型
            avg_loss = self.validate(verbose=verbose)
            self.logger.info(f'Val -- Epoch [{epoch+1}/{self.config["train"]["epochs"]}], Loss: {avg_loss:.4f}')
            if avg_loss < min_loss:
                min_loss = avg_loss
                self.save_model(os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pth"))

            if epoch % self.save_frequency == 0:
                self.save_model(os.path.join(self.checkpoint_dir, f"{self.model_name}_{epoch}.pth"))

    def validate(self, verbose=False):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            progress_bar = tqdm(enumerate(self.val_dataloader))
            for i, (inputs, labels) in progress_bar:
                inputs = {k:v.to(self.device) for k, v in inputs.items()}
                labels = {k:v.to(self.device) for k, v in labels.items()}

                outputs = self.model(inputs)

                loss, loss_detail = calculate_loss(outputs, labels, self.criterion)
                
                total_loss += loss.item()

            avg_loss = total_loss / len(self.val_dataloader)
            if verbose:
                print(f'Validation Loss: {avg_loss:.4f}')

        return avg_loss

    def test(self, verbose=False):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            progress_bar = tqdm(enumerate(self.test_dataloader))
            for i, (inputs, labels) in progress_bar:
                inputs = {k:v.to(self.device) for k, v in inputs.items()}
                labels = {k:v.to(self.device) for k, v in labels.items()}
                labels['anchor'] = self.prior_box.data.to(self.device)

                outputs = self.model(inputs)
                loss, loss_detail = calculate_loss(outputs, labels, self.criterion)
                total_loss += loss.item()

            avg_loss = total_loss / len(self.test_dataloader)
            if verbose:
                print(f'Test Loss: {avg_loss:.4f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

if __name__ == '__main__':
    trainer = Trainer('configs/train.yaml')
    trainer.train(verbose=True)