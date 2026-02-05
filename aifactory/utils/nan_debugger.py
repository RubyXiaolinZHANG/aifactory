import torch
import torch.nn as nn
import os
import json
import datetime
import pickle
import numpy as np


class NanDebugger:
    def __init__(self, model, optimizer, save_dir='./debug_snapshots'):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self._snapshot_dir = NanDebugger

    @property
    def snapshot_dir(self):
        return self._snapshot_dir

    def check_and_save(self, loss, iteration, additional_info=None):
        """Check if loss is NaN, and save the complete state if so"""
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"⚠️  NaN/Inf detected at iteration {iteration}! Saving debug snapshot...")
            self.save_snapshot(iteration, loss, additional_info)
            return True
        return False

    def save_snapshot(self, iteration, loss=None, additional_info=None):
        """Save complete debugging snapshot"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = os.path.join(self.save_dir, f"nan_debug_{timestamp}_iter{iteration}")
        self._snapshot_dir = snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)

        # 1. Save model parameters
        model_path = os.path.join(snapshot_dir, "model_state.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_structure': str(self.model),
            'model_config': self._get_model_config()
        }, model_path)

        # 2. Save optimizer state
        optimizer_path = os.path.join(snapshot_dir, "optimizer_state.pth")
        torch.save(self.optimizer.state_dict(), optimizer_path)

        # 3. Save gradient information
        self._save_gradient_info(snapshot_dir)

        # 4. Save layer statistics
        self._save_layer_statistics(snapshot_dir)

        # 5. Save metadata
        if isinstance(loss, dict):
            loss = {name: float(val.item()) for name, val in loss.items()}
        elif isinstance(loss, torch.Tensor):
            loss = float(loss.item())
        elif isinstance(loss, (tuple, dict)):
            loss = [float(val.item()) if isinstance(val, torch.Tensor) else float(val) for val in loss]
        elif loss is None:
            pass
        else:
            raise ValueError("do not support loss of type: {}".format(type(loss)))
        metadata = {
            'timestamp': timestamp,
            'iteration': iteration,
            'loss_value': loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'additional_info': additional_info or {},
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device': str(next(self.model.parameters()).device)
        }

        with open(os.path.join(snapshot_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # 6. Save current Python environment information
        self._save_environment_info(snapshot_dir)

        print(f"✅ Debug snapshot saved to: {snapshot_dir}")
        return snapshot_dir

    def _save_gradient_info(self, save_dir):
        """Save gradient information for each layer"""
        grad_info = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                grad_info[name] = {
                    'mean': float(grad.mean().item()),
                    'std': float(grad.std().item()),
                    'max': float(grad.max().item()),
                    'min': float(grad.min().item()),
                    'has_nan': bool(torch.isnan(grad).any()),
                    'has_inf': bool(torch.isinf(grad).any())
                }

        with open(os.path.join(save_dir, 'gradients.json'), 'w') as f:
            json.dump(grad_info, f, indent=2)

        # Save gradient histogram data
        self._save_gradient_histograms(save_dir)

    def _save_layer_statistics(self, save_dir):
        """Save parameter statistics for each layer"""
        stats = {}
        for name, param in self.model.named_parameters():
            stats[name] = {
                'data_mean': float(param.data.mean().item()),
                'data_std': float(param.data.std().item()),
                'data_max': float(param.data.max().item()),
                'data_min': float(param.data.min().item()),
                'has_nan': bool(torch.isnan(param.data).any()),
                'has_inf': bool(torch.isinf(param.data).any()),
                'shape': list(param.data.shape)
            }

        with open(os.path.join(save_dir, 'layer_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

    def _get_model_config(self):
        """Get model configuration information"""
        config = {}
        for name, module in self.model.named_modules():
            if name:  # Skip empty name (root module)
                config[name] = str(module)
        return config

    def _save_gradient_histograms(self, save_dir):
        """Save gradient histogram data (for subsequent visualization)"""
        hist_data = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_flat = param.grad.data.cpu().numpy().flatten()
                try:
                    hist, bins = np.histogram(grad_flat, bins=50)
                    hist_data[name] = {
                        'hist': hist.tolist(),
                        'bins': bins.tolist()
                    }
                except:
                    hist_data[name] = {
                        'hist': 0,
                        'bins': 0
                    }

        with open(os.path.join(save_dir, 'gradient_hists.json'), 'w') as f:
            json.dump(hist_data, f, indent=2)

    def _save_environment_info(self, save_dir):
        """Save environment information"""
        import sys
        import platform

        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'command': ' '.join(sys.argv)
        }

        with open(os.path.join(save_dir, 'environment.json'), 'w') as f:
            json.dump(env_info, f, indent=2)


