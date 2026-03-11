"""
TensorBoard Logger - 用于可视化训练过程
"""

import os
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """TensorBoard日志记录器"""
    
    def __init__(self, log_dir, enabled=True):
        """
        Args:
            log_dir (str): TensorBoard日志目录
            enabled (bool): 是否启用TensorBoard
        """
        self.enabled = enabled
        self.writer = None
        
        if self.enabled:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"✅ TensorBoard logger initialized: {log_dir}")
            print(f"   Run: tensorboard --logdir {log_dir}")
    
    def add_scalar(self, tag, value, step):
        """记录标量值"""
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def add_scalars(self, main_tag, tag_scalar_dict, step):
        """记录多个标量值"""
        if self.enabled and self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def add_histogram(self, tag, values, step):
        """记录直方图"""
        if self.enabled and self.writer is not None:
            self.writer.add_histogram(tag, values, step)
    
    def add_text(self, tag, text, step):
        """记录文本"""
        if self.enabled and self.writer is not None:
            self.writer.add_text(tag, text, step)
    
    def flush(self):
        """刷新缓冲"""
        if self.enabled and self.writer is not None:
            self.writer.flush()
    
    def close(self):
        """关闭writer"""
        if self.enabled and self.writer is not None:
            self.writer.close()
            print("✅ TensorBoard logger closed")
