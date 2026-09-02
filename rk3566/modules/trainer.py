"""
Trainer
=======
标准 PyTorch 训练循环: train / validate / test + checkpoint 保存。

支持 CrossEntropyLoss + Adam/SGD, 所有超参通过 YAML 配置。
每个 epoch 输出: Epoch / Train Loss / Train Acc / Val Loss / Val Acc / LR
保存 validation accuracy 最高的 checkpoint。
"""
import os
import time
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: nn.Module, cfg: Dict[str, Any],
                 device: str = 'cuda', seed: int = 42):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.model.to(device)

        # 优化器
        opt_name = cfg.get('optimizer', 'adam').lower()
        lr = cfg.get('learning_rate', 0.001)
        weight_decay = cfg.get('weight_decay', 0.0)
        if opt_name == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(),
                                              lr=lr, weight_decay=weight_decay)
        elif opt_name == 'sgd':
            momentum = cfg.get('momentum', 0.9)
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                             momentum=momentum,
                                             weight_decay=weight_decay)
        else:
            raise ValueError(f'不支持的优化器: {opt_name}')

        self.criterion = nn.CrossEntropyLoss()

        # 学习率调度: 余弦退火 (可选)
        self.scheduler = None
        if cfg.get('scheduler'):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg['epochs'])

        self.epochs = cfg.get('epochs', 30)
        self.batch_size = cfg.get('batch_size', 128)
        self.num_workers = cfg.get('num_workers', 2)
        self.seed = seed
        self.best_val_acc = 0.0
        self.history: Dict[str, list] = {
            'epoch': [], 'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'lr': []}

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        return {'loss': total_loss / total, 'acc': correct / total}

    # ------------------------------------------------------------------
    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            checkpoint_dir: str, model_id: str = 'model',
            verbose: bool = True) -> float:
        """完整训练流程, 返回 best validation accuracy。"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f'{model_id}_best.pt')
        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)
            lr = self.optimizer.param_groups[0]['lr']

            if self.scheduler is not None:
                self.scheduler.step()

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['lr'].append(lr)

            if val_metrics['acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['acc']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'config': self.cfg,
                }, ckpt_path)

            if verbose:
                print(f'  Epoch {epoch:3d}/{self.epochs} | '
                      f'Train Loss {train_metrics["loss"]:.4f} | '
                      f'Train Acc {train_metrics["acc"]:.4f} | '
                      f'Val Loss {val_metrics["loss"]:.4f} | '
                      f'Val Acc {val_metrics["acc"]:.4f} | '
                      f'LR {lr:.2e}')

        self.training_time = time.time() - t0
        return self.best_val_acc

    # ------------------------------------------------------------------
    def evaluate(self, test_loader: DataLoader,
                 checkpoint_path: Optional[str] = None) -> Dict[str, float]:
        """测试集评估。checkpoint_path 指定时加载 best 权重。"""
        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])

        metrics = self._run_epoch(test_loader, train=False)
        return {'test_loss': metrics['loss'], 'test_acc': metrics['acc']}

    # ------------------------------------------------------------------
    def load_best(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        return ckpt.get('best_val_acc', 0.0)
