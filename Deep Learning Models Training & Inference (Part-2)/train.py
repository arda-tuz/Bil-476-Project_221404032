import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class Trainer:
    """
    Comprehensive training framework for time series models.
    
    Features:
    - RMSLE loss function (competition metric)
    - Early stopping with patience
    - Learning rate scheduling
    - Detailed metrics tracking
    - Visualization of training progress
    - Model checkpointing
    """
    
    def __init__(self, model, device='cuda', model_dir='./saved_models/model'):
        self.model = model.to(device)
        self.device = device
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        self.checkpoint_dir = self.model_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.plots_dir = self.model_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        self.logs_dir = self.model_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        
        # Training history tracking
        self.train_losses = []
        self.val_losses = []
        self.train_metrics_history = []
        self.val_metrics_history = []
        self.best_val_loss = float('inf')
        self.best_val_rmsle = float('inf')
        
    def rmsle_loss(self, predictions, targets):
        """
        Root Mean Squared Logarithmic Error - the competition evaluation metric.
        
        RMSLE is particularly suitable for sales forecasting because:
        1. It treats percentage errors equally across different scales
        2. It penalizes under-prediction more than over-prediction
        3. It handles zero values gracefully with log1p
        """
        # Ensure non-negative predictions
        predictions = torch.clamp(predictions, min=0)
        
        # Calculate log1p (log(1 + x)) for numerical stability
        log_pred = torch.log1p(predictions)
        log_true = torch.log1p(targets)
        
        # RMSLE calculation
        rmsle = torch.sqrt(torch.mean((log_pred - log_true) ** 2))
        
        return rmsle
    
    def calculate_metrics(self, predictions, targets):
        """Calculate comprehensive metrics for evaluation"""
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Ensure non-negative predictions
        predictions = np.clip(predictions, 0, None)
        
        # RMSLE
        log_pred = np.log1p(predictions)
        log_true = np.log1p(targets)
        rmsle = np.sqrt(np.mean((log_pred - log_true) ** 2))
        
        # R² score
        r2 = r2_score(targets.flatten(), predictions.flatten())
        
        # Mean Absolute Error
        mae = np.mean(np.abs(predictions - targets))
        
        # Mean Squared Error
        mse = np.mean((predictions - targets) ** 2)
        
        return {'rmsle': rmsle, 'r2': r2, 'mae': mae, 'mse': mse, 'rmse': np.sqrt(mse)}
    
    def train_epoch(self, model, train_loader, optimizer, criterion='rmsle'):
        """
        Train for one epoch.
        
        Training process:
        1. Forward pass through model
        2. Calculate loss (RMSLE or MSE)
        3. Backward pass to compute gradients
        4. Gradient clipping to prevent exploding gradients
        5. Optimizer step to update weights
        """
        model.train()
        total_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch in progress_bar:
            # Move batch to device
            batch = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            
            # Forward pass
            predictions = model(batch)
            targets = batch['sales_target']
            
            # Calculate loss
            if criterion == 'rmsle':
                loss = self.rmsle_loss(predictions, targets)
            elif criterion == 'mse':
                loss = nn.MSELoss()(predictions, targets)
            else:
                raise ValueError(f"Unknown criterion: {criterion}")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Store for metrics calculation
            all_predictions.append(predictions.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        train_metrics = self.calculate_detailed_metrics(all_predictions, all_targets)
        
        return total_loss / num_batches, train_metrics
    
    def evaluate(self, model, val_loader, criterion='rmsle'):
        """Evaluate model on validation/test set without gradient computation"""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                # Move batch to device
                batch = {
                    k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                
                # Forward pass
                predictions = model(batch)
                targets = batch['sales_target']
                
                # Calculate loss
                if criterion == 'rmsle':
                    loss = self.rmsle_loss(predictions, targets)
                elif criterion == 'mse':
                    loss = nn.MSELoss()(predictions, targets)
                else:
                    raise ValueError(f"Unknown criterion: {criterion}")
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store for metrics
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Calculate detailed metrics
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        metrics = self.calculate_detailed_metrics(all_predictions, all_targets)
        
        return total_loss / num_batches, metrics
    
    def calculate_detailed_metrics(self, predictions, targets):
        """
        Calculate metrics for different time periods.
        
        The competition uses:
        - Public leaderboard: First 5 days
        - Private leaderboard: Days 6-16
        
        We calculate metrics for both to understand model performance.
        """
        # Overall metrics
        overall_metrics = self.calculate_metrics(predictions, targets)
        
        # Public period (days 1-5)
        public_metrics = self.calculate_metrics(predictions[:, :5], targets[:, :5])
        
        # Private period (days 6-16)
        private_metrics = self.calculate_metrics(predictions[:, 5:], targets[:, 5:])
        
        # Per-day RMSLE
        per_day_rmsle = []
        for day in range(predictions.shape[1]):
            day_rmsle = np.sqrt(np.mean((np.log1p(predictions[:, day]) - np.log1p(targets[:, day]))**2))
            per_day_rmsle.append(day_rmsle)
        
        metrics = {
            'overall_rmsle': overall_metrics['rmsle'],
            'overall_r2': overall_metrics['r2'],
            'overall_mae': overall_metrics['mae'],
            'overall_rmse': overall_metrics['rmse'],
            'public_rmsle': public_metrics['rmsle'],
            'public_r2': public_metrics['r2'],
            'public_mae': public_metrics['mae'],
            'private_rmsle': private_metrics['rmsle'],
            'private_r2': private_metrics['r2'],
            'private_mae': private_metrics['mae'],
            'mean_prediction': predictions.mean(),
            'mean_target': targets.mean(),
            'std_prediction': predictions.std(),
            'std_target': targets.std(),
            'per_day_rmsle': per_day_rmsle
        }
        
        return metrics
    
    def plot_training_history(self):
        """Create comprehensive training visualization plots"""
        epochs = range(1, len(self.train_losses) + 1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training History', fontsize=16)
        
        # Plot 1: Loss curves
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: RMSLE
        train_rmsle = [m['overall_rmsle'] for m in self.train_metrics_history]
        val_rmsle = [m['overall_rmsle'] for m in self.val_metrics_history]
        axes[0, 1].plot(epochs, train_rmsle, 'b-', label='Train RMSLE')
        axes[0, 1].plot(epochs, val_rmsle, 'r-', label='Val RMSLE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RMSLE')
        axes[0, 1].set_title('RMSLE over Epochs')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: R² Score
        train_r2 = [m['overall_r2'] for m in self.train_metrics_history]
        val_r2 = [m['overall_r2'] for m in self.val_metrics_history]
        axes[0, 2].plot(epochs, train_r2, 'b-', label='Train R²')
        axes[0, 2].plot(epochs, val_r2, 'r-', label='Val R²')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('R² Score')
        axes[0, 2].set_title('R² Score over Epochs')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Plot 4: Public vs Private RMSLE
        val_public_rmsle = [m['public_rmsle'] for m in self.val_metrics_history]
        val_private_rmsle = [m['private_rmsle'] for m in self.val_metrics_history]
        axes[1, 0].plot(epochs, val_public_rmsle, 'g-', label='Val Public RMSLE')
        axes[1, 0].plot(epochs, val_private_rmsle, 'm-', label='Val Private RMSLE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSLE')
        axes[1, 0].set_title('Public vs Private RMSLE (Validation)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 5: MAE
        train_mae = [m['overall_mae'] for m in self.train_metrics_history]
        val_mae = [m['overall_mae'] for m in self.val_metrics_history]
        axes[1, 1].plot(epochs, train_mae, 'b-', label='Train MAE')
        axes[1, 1].plot(epochs, val_mae, 'r-', label='Val MAE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('Mean Absolute Error over Epochs')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Plot 6: Mean predictions vs targets
        train_mean_pred = [m['mean_prediction'] for m in self.train_metrics_history]
        train_mean_target = [m['mean_target'] for m in self.train_metrics_history]
        val_mean_pred = [m['mean_prediction'] for m in self.val_metrics_history]
        val_mean_target = [m['mean_target'] for m in self.val_metrics_history]
        
        axes[1, 2].plot(epochs, train_mean_pred, 'b-', label='Train Mean Pred')
        axes[1, 2].plot(epochs, train_mean_target, 'b--', label='Train Mean Target')
        axes[1, 2].plot(epochs, val_mean_pred, 'r-', label='Val Mean Pred')
        axes[1, 2].plot(epochs, val_mean_target, 'r--', label='Val Mean Target')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Mean Value')
        axes[1, 2].set_title('Mean Predictions vs Targets')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_history.png', dpi=150)
        plt.close()
        
        # Plot per-day RMSLE for the last epoch
        if self.val_metrics_history:
            fig, ax = plt.subplots(figsize=(10, 6))
            last_val_metrics = self.val_metrics_history[-1]
            days = range(1, len(last_val_metrics['per_day_rmsle']) + 1)
            ax.plot(days, last_val_metrics['per_day_rmsle'], 'o-')
            ax.axvline(x=5.5, color='red', linestyle='--', alpha=0.5, label='Public/Private Split')
            ax.set_xlabel('Day')
            ax.set_ylabel('RMSLE')
            ax.set_title('Per-Day RMSLE (Last Epoch - Validation)')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'per_day_rmsle.png', dpi=150)
            plt.close()
    
    def save_training_logs(self):
        """Save all training logs to JSON files for later analysis"""
        # Convert numpy scalars to Python floats for JSON serialization
        serializable_history = {
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses':   [float(x) for x in self.val_losses],
            'train_metrics': [
                {k: ([float(i) for i in v] if isinstance(v, list) else float(v))
                for k, v in epoch.items()}
                for epoch in self.train_metrics_history
            ],
            'val_metrics': [
                {k: ([float(i) for i in v] if isinstance(v, list) else float(v))
                for k, v in epoch.items()}
                for epoch in self.val_metrics_history
            ],
            'best_val_rmsle': float(self.best_val_rmsle),
            'best_val_loss':  float(self.best_val_loss)
        }
        
        # Save full training history
        with open(self.logs_dir / 'training_history.json', 'w') as f:
            json.dump(serializable_history, f, indent=2)

        # Create epoch-by-epoch summary
        epoch_summary = []
        for idx in range(len(self.train_losses)):
            summary = {
                'epoch': idx + 1,
                'train_loss': float(self.train_losses[idx]),
                'val_loss':   float(self.val_losses[idx]),
                'train_rmsle': float(self.train_metrics_history[idx]['overall_rmsle']),
                'val_rmsle':   float(self.val_metrics_history[idx]['overall_rmsle']),
                'train_r2':    float(self.train_metrics_history[idx]['overall_r2']),
                'val_r2':      float(self.val_metrics_history[idx]['overall_r2']),
                'val_public_rmsle':  float(self.val_metrics_history[idx]['public_rmsle']),
                'val_private_rmsle': float(self.val_metrics_history[idx]['private_rmsle'])
            }
            epoch_summary.append(summary)

        # Save epoch summary
        with open(self.logs_dir / 'epoch_summary.json', 'w') as f:
            json.dump(epoch_summary, f, indent=2)
    
    def train(self, train_loader, val_loader, num_epochs=10, lr=0.001, 
              criterion='rmsle', patience=5, hyperparams=None):
        """
        Main training loop with early stopping and learning rate scheduling.
        
        Training strategy:
        1. Train for specified epochs
        2. Monitor validation RMSLE
        3. Reduce learning rate when validation plateaus
        4. Stop early if no improvement for 'patience' epochs
        5. Save best model based on validation RMSLE
        """
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience//2, 
                                    factor=0.5)
        
        # Early stopping counter
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss, train_metrics = self.train_epoch(self.model, train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            self.train_metrics_history.append(train_metrics)
            
            # Validation phase
            val_loss, val_metrics = self.evaluate(self.model, val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_metrics_history.append(val_metrics)
            
            # Update learning rate based on validation RMSLE
            scheduler.step(val_metrics['overall_rmsle'])
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train RMSLE: {train_metrics['overall_rmsle']:.4f}, Train R²: {train_metrics['overall_r2']:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val RMSLE: {val_metrics['overall_rmsle']:.4f} (Public: {val_metrics['public_rmsle']:.4f}, Private: {val_metrics['private_rmsle']:.4f})")
            print(f"Val R²: {val_metrics['overall_r2']:.4f} (Public: {val_metrics['public_r2']:.4f}, Private: {val_metrics['private_r2']:.4f})")
            
            # Save best model based on validation RMSLE
            if val_metrics['overall_rmsle'] < self.best_val_rmsle:
                self.best_val_rmsle = val_metrics['overall_rmsle']
                self.save_checkpoint(epoch, val_loss, val_metrics, hyperparams)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save final plots and logs
        self.plot_training_history()
        self.save_training_logs()
        
        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, epoch, val_loss, val_metrics, hyperparams=None):
        """Save model checkpoint with descriptive filename"""
        # Create descriptive filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if hyperparams is not None:
            model_name = (f"{hyperparams['model_type']}_"
                         f"lr{hyperparams['lr']}_"
                         f"dim{hyperparams['latent_dim']}_"
                         f"drop{hyperparams['dropout']}_"
                         f"rmsle{val_metrics['overall_rmsle']:.4f}_"
                         f"{timestamp}.pth")
        else:
            model_name = f"model_rmsle{val_metrics['overall_rmsle']:.4f}_{timestamp}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'hyperparams': hyperparams,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_rmsle': self.best_val_rmsle,
            'timestamp': timestamp
        }
        
        checkpoint_path = self.checkpoint_dir / model_name
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Also save as 'best_model.pth' for easy access
        best_path = self.checkpoint_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)
        
        # Save metrics to JSON
        serializable_val_metrics = {}
        for k, v in val_metrics.items():
            if isinstance(v, list):
                serializable_val_metrics[k] = [float(x) for x in v]
            else:
                serializable_val_metrics[k] = float(v)

        metrics_data = {
            'model_name': model_name,
            'epoch': epoch,
            'val_metrics': serializable_val_metrics,
            'hyperparams': hyperparams,
            'timestamp': timestamp
        }
        
        metrics_path = self.logs_dir / f'best_model_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path=None):
        """Load model checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return checkpoint


def make_predictions(model, test_loader, device='cuda'):
    """Generate predictions for test set"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Generating predictions'):
            # Move batch to device
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            
            # Forward pass
            predictions = model(batch)
            
            # Store predictions
            all_predictions.append(predictions.cpu().numpy())
    
    # Concatenate all predictions
    all_predictions = np.vstack(all_predictions)
    
    return all_predictions


def evaluate_on_test_set(model, test_loader, device='cuda', save_dir=None):
    """Evaluate model on test set and return comprehensive metrics"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating on test set'):
            # Move batch to device
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            
            # Forward pass
            predictions = model(batch)
            targets = batch['sales_target']
            
            # Store predictions and targets
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics
    trainer = Trainer(model, device)  # Create temporary trainer for metrics
    metrics = trainer.calculate_detailed_metrics(all_predictions, all_targets)
    
    # Save visualizations if directory provided
    if save_dir:
        save_dir = Path(save_dir)
        plots_dir = save_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Plot predictions vs actual for random samples
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Randomly select 4 samples
        indices = np.random.choice(len(all_predictions), 4, replace=False)
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            days = np.arange(16)
            
            ax.plot(days, all_targets[idx], 'o-', label='Actual', linewidth=2)
            ax.plot(days, all_predictions[idx], 's-', label='Predicted', linewidth=2)
            ax.axvline(x=4.5, color='red', linestyle='--', alpha=0.5, label='Public/Private Split')
            
            sample_rmsle = np.sqrt(np.mean((np.log1p(all_predictions[idx]) - np.log1p(all_targets[idx]))**2))
            ax.set_xlabel('Day')
            ax.set_ylabel('Sales')
            ax.set_title(f'Sample {idx}: RMSLE = {sample_rmsle:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'test_predictions_samples.png', dpi=150)
        plt.close()
        
        # Save test metrics
        logs_dir = save_dir / 'logs'
        logs_dir.mkdir(exist_ok=True)
        
        # Convert to JSON-serializable format
        test_metrics_clean = {}
        for k, v in metrics.items():
            if isinstance(v, list):
                test_metrics_clean[k] = [float(i) for i in v]
            else:
                test_metrics_clean[k] = float(v)
            
        with open(logs_dir / 'test_metrics.json', 'w') as f:
            json.dump(test_metrics_clean, f, indent=2)
    
    return metrics, all_predictions, all_targets