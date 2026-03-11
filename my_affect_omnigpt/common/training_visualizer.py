"""
训练可视化工具 - 自动保存学习率和Loss曲线图
集成到训练流程中，无需额外启动监控脚本
"""

import os
import matplotlib
matplotlib.use('Agg')  # 非GUI后端
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class TrainingVisualizer:
    """训练可视化器 - 记录并绘制学习率和Loss曲线"""
    
    def __init__(self, output_dir, enabled=True):
        """
        Args:
            output_dir (str): 输出目录
            enabled (bool): 是否启用可视化
        """
        self.enabled = enabled
        if not self.enabled:
            return
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.steps = []
        self.lrs = []
        self.losses = []
        self.epochs = []
        
        print(f"📊 Training Visualizer enabled: {self.output_dir}")
    
    def add_scalar(self, epoch, step, lr, loss):
        """
        添加一条训练记录
        
        Args:
            epoch (int): 当前epoch
            step (int): 当前step（epoch内的步数）
            lr (float): 学习率
            loss (float): 损失值
        """
        if not self.enabled:
            return
        
        self.epochs.append(epoch)
        self.steps.append(len(self.steps))  # 全局步数
        self.lrs.append(lr)
        self.losses.append(loss)
    
    def plot_and_save(self, suffix=''):
        """
        绘制并保存曲线图
        
        Args:
            suffix (str): 文件名后缀（如'_epoch10'）
        """
        if not self.enabled or len(self.steps) == 0:
            return
        
        # 设置样式
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Progress - {len(self.steps)} steps recorded', 
                     fontsize=16, fontweight='bold')
        
        # 1. 学习率曲线（线性）
        ax1 = axes[0, 0]
        ax1.plot(self.steps, self.lrs, linewidth=1.5, alpha=0.9, color='#3498db')
        ax1.set_xlabel('Steps', fontsize=12)
        ax1.set_ylabel('Learning Rate', fontsize=12)
        ax1.set_title('Learning Rate vs Steps (Linear Scale)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.ticklabel_format(axis='x', style='plain')
        
        # 添加warmup和衰减阶段标注
        if len(self.lrs) > 10:
            max_lr_idx = np.argmax(self.lrs)
            ax1.axvline(x=self.steps[max_lr_idx], color='red', linestyle='--', 
                       alpha=0.5, label=f'Peak LR (step {self.steps[max_lr_idx]})')
            ax1.legend(fontsize=10)
        
        # 2. 学习率曲线（对数）
        ax2 = axes[0, 1]
        ax2.plot(self.steps, self.lrs, linewidth=1.5, alpha=0.9, color='#e67e22')
        ax2.set_xlabel('Steps', fontsize=12)
        ax2.set_ylabel('Learning Rate (log scale)', fontsize=12)
        ax2.set_title('Learning Rate vs Steps (Log Scale)', fontsize=13, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.ticklabel_format(axis='x', style='plain')
        
        # 3. Loss曲线
        ax3 = axes[1, 0]
        # 原始loss（半透明）
        ax3.plot(self.steps, self.losses, linewidth=0.5, alpha=0.3, color='gray', label='Raw Loss')
        
        # 平滑loss
        if len(self.losses) > 50:
            window = min(100, len(self.losses) // 10)
            if window > 1:
                smoothed_loss = np.convolve(self.losses, np.ones(window)/window, mode='valid')
                smoothed_steps = self.steps[window-1:]
                ax3.plot(smoothed_steps, smoothed_loss, linewidth=2.5, color='#e74c3c', 
                        label=f'Smoothed (window={window})')
        
        ax3.set_xlabel('Steps', fontsize=12)
        ax3.set_ylabel('Loss', fontsize=12)
        ax3.set_title('Loss vs Steps', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(fontsize=10)
        ax3.ticklabel_format(axis='x', style='plain')
        
        # 添加最低loss标注
        min_loss_idx = np.argmin(self.losses)
        ax3.plot(self.steps[min_loss_idx], self.losses[min_loss_idx], 'r*', 
                markersize=15, label=f'Min Loss: {self.losses[min_loss_idx]:.4f}')
        ax3.legend(fontsize=10)
        
        # 4. 每个Epoch的平均Loss
        ax4 = axes[1, 1]
        if len(self.epochs) > 0:
            unique_epochs = sorted(set(self.epochs))
            epoch_losses = []
            epoch_stds = []
            
            for ep in unique_epochs:
                epoch_mask = [i for i, e in enumerate(self.epochs) if e == ep]
                losses_in_epoch = [self.losses[i] for i in epoch_mask]
                avg_loss = np.mean(losses_in_epoch)
                std_loss = np.std(losses_in_epoch)
                epoch_losses.append(avg_loss)
                epoch_stds.append(std_loss)
            
            # 绘制平均loss
            ax4.plot(unique_epochs, epoch_losses, marker='o', linewidth=2.5, 
                    markersize=8, color='#9b59b6', label='Average Loss')
            
            # 添加标准差阴影
            if len(epoch_losses) > 1:
                ax4.fill_between(unique_epochs, 
                                np.array(epoch_losses) - np.array(epoch_stds),
                                np.array(epoch_losses) + np.array(epoch_stds),
                                alpha=0.2, color='#9b59b6')
            
            ax4.set_xlabel('Epoch', fontsize=12)
            ax4.set_ylabel('Average Loss', fontsize=12)
            ax4.set_title('Average Loss per Epoch', fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3, linestyle='--')
            ax4.legend(fontsize=10)
            
            # 设置整数刻度
            if len(unique_epochs) < 20:
                ax4.set_xticks(unique_epochs)
        
        plt.tight_layout()
        
        # 保存标准分辨率
        output_path = self.output_dir / f'training_curves{suffix}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training curves saved: {output_path}")
        
        # 保存高分辨率版本（每5个epoch保存一次，避免文件过多）
        if suffix == '' or 'final' in suffix.lower() or (self.epochs and self.epochs[-1] % 5 == 0):
            output_path_hd = self.output_dir / f'training_curves_hd{suffix}.png'
            
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle(f'Training Progress (High Resolution) - {len(self.steps)} steps', 
                        fontsize=20, fontweight='bold')
            
            # 重新绘制（高分辨率）
            # 1. 学习率（线性）
            ax1 = axes[0, 0]
            ax1.plot(self.steps, self.lrs, linewidth=2, alpha=0.9, color='#3498db')
            ax1.set_xlabel('Steps', fontsize=14)
            ax1.set_ylabel('Learning Rate', fontsize=14)
            ax1.set_title('Learning Rate vs Steps (Linear Scale)', fontsize=16, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            if len(self.lrs) > 10:
                max_lr_idx = np.argmax(self.lrs)
                ax1.axvline(x=self.steps[max_lr_idx], color='red', linestyle='--', 
                           alpha=0.5, label=f'Peak LR (step {self.steps[max_lr_idx]})')
                ax1.legend(fontsize=12)
            
            # 2. 学习率（对数）
            ax2 = axes[0, 1]
            ax2.plot(self.steps, self.lrs, linewidth=2, alpha=0.9, color='#e67e22')
            ax2.set_xlabel('Steps', fontsize=14)
            ax2.set_ylabel('Learning Rate (log scale)', fontsize=14)
            ax2.set_title('Learning Rate vs Steps (Log Scale)', fontsize=16, fontweight='bold')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            # 3. Loss
            ax3 = axes[1, 0]
            ax3.plot(self.steps, self.losses, linewidth=0.8, alpha=0.3, color='gray', label='Raw Loss')
            if len(self.losses) > 50:
                window = min(100, len(self.losses) // 10)
                if window > 1:
                    smoothed_loss = np.convolve(self.losses, np.ones(window)/window, mode='valid')
                    smoothed_steps = self.steps[window-1:]
                    ax3.plot(smoothed_steps, smoothed_loss, linewidth=3, color='#e74c3c', 
                            label=f'Smoothed (window={window})')
            ax3.set_xlabel('Steps', fontsize=14)
            ax3.set_ylabel('Loss', fontsize=14)
            ax3.set_title('Loss vs Steps', fontsize=16, fontweight='bold')
            ax3.grid(True, alpha=0.3, linestyle='--')
            min_loss_idx = np.argmin(self.losses)
            ax3.plot(self.steps[min_loss_idx], self.losses[min_loss_idx], 'r*', 
                    markersize=20, label=f'Min Loss: {self.losses[min_loss_idx]:.4f}')
            ax3.legend(fontsize=12)
            
            # 4. Epoch Loss
            ax4 = axes[1, 1]
            if len(self.epochs) > 0:
                unique_epochs = sorted(set(self.epochs))
                epoch_losses = []
                epoch_stds = []
                for ep in unique_epochs:
                    epoch_mask = [i for i, e in enumerate(self.epochs) if e == ep]
                    losses_in_epoch = [self.losses[i] for i in epoch_mask]
                    epoch_losses.append(np.mean(losses_in_epoch))
                    epoch_stds.append(np.std(losses_in_epoch))
                
                ax4.plot(unique_epochs, epoch_losses, marker='o', linewidth=3, 
                        markersize=10, color='#9b59b6', label='Average Loss')
                if len(epoch_losses) > 1:
                    ax4.fill_between(unique_epochs, 
                                    np.array(epoch_losses) - np.array(epoch_stds),
                                    np.array(epoch_losses) + np.array(epoch_stds),
                                    alpha=0.2, color='#9b59b6')
                ax4.set_xlabel('Epoch', fontsize=14)
                ax4.set_ylabel('Average Loss', fontsize=14)
                ax4.set_title('Average Loss per Epoch', fontsize=16, fontweight='bold')
                ax4.grid(True, alpha=0.3, linestyle='--')
                ax4.legend(fontsize=12)
                if len(unique_epochs) < 20:
                    ax4.set_xticks(unique_epochs)
            
            plt.tight_layout()
            plt.savefig(output_path_hd, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ High-res curves saved: {output_path_hd}")
    
    def print_statistics(self):
        """打印训练统计信息"""
        if not self.enabled or len(self.steps) == 0:
            return
        
        print("\n" + "="*70)
        print("📊 Training Statistics")
        print("="*70)
        print(f"  Total Steps:          {len(self.steps):,}")
        print(f"  Current Epoch:        {self.epochs[-1]}")
        print(f"  Current Learning Rate: {self.lrs[-1]:.2e}")
        print(f"  Latest Loss:          {self.losses[-1]:.6f}")
        
        if len(self.losses) > 100:
            recent_loss = np.mean(self.losses[-100:])
            print(f"  Recent 100 Avg Loss:  {recent_loss:.6f}")
        
        min_loss = min(self.losses)
        min_loss_step = self.steps[self.losses.index(min_loss)]
        print(f"  Best Loss:            {min_loss:.6f} (Step {min_loss_step})")
        print(f"  Max Learning Rate:    {max(self.lrs):.2e}")
        print(f"  Min Learning Rate:    {min(self.lrs):.2e}")
        print("="*70 + "\n")
    
    def save_data(self, suffix=''):
        """保存原始数据为numpy文件"""
        if not self.enabled or len(self.steps) == 0:
            return
        
        data_file = self.output_dir / f'training_data{suffix}.npz'
        np.savez(data_file,
                 steps=np.array(self.steps),
                 epochs=np.array(self.epochs),
                 lrs=np.array(self.lrs),
                 losses=np.array(self.losses))
        print(f"💾 Training data saved: {data_file}")
    
    def load_data(self, data_file):
        """加载之前保存的数据（用于恢复训练）"""
        if not self.enabled:
            return
        
        data_file = Path(data_file)
        if not data_file.exists():
            print(f"⚠️  Data file not found: {data_file}")
            return
        
        data = np.load(data_file)
        self.steps = data['steps'].tolist()
        self.epochs = data['epochs'].tolist()
        self.lrs = data['lrs'].tolist()
        self.losses = data['losses'].tolist()
        
        print(f"✅ Loaded {len(self.steps)} training records from {data_file}")
