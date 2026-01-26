import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


def inverse_transform_targets(scaler, data, target_indices):
    """对目标变量进行逆标准化"""
    mean = scaler.mean_[target_indices]
    scale = scaler.scale_[target_indices]
    if isinstance(data, torch.Tensor):
        mean_t = torch.tensor(mean, device=data.device, dtype=data.dtype)
        scale_t = torch.tensor(scale, device=data.device, dtype=data.dtype)
        shape = (1,) * (data.ndim - 1) + (-1,)
        mean_t = mean_t.reshape(shape)
        scale_t = scale_t.reshape(shape)
        return data * scale_t + mean_t
    shape = (1,) * (data.ndim - 1) + (-1,)
    mean = mean.reshape(shape)
    scale = scale.reshape(shape)
    return data * scale + mean


def plot_predictions(y_true, y_pred, save_path='predictions.png'):
    """绘制预测结果"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='True', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Prediction Results')
    plt.savefig(save_path)
    plt.close()


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
