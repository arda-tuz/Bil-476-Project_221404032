import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules with updated names
from models import create_model
from train import Trainer, evaluate_on_test_set
from dataset import create_dataloaders_with_test
from data_utils import DataPivotProcessor


class ModelEvaluator:
    """
    Comprehensive model evaluation framework.
    
    This class handles:
    1. Loading saved models and evaluating on test sets
    2. Creating leaderboards to compare different models
    3. Visualizing predictions and error patterns
    4. Analyzing model performance across different dimensions
    """
    
    def __init__(self, data_dir='./', saved_models_dir='./saved_models', device='cuda'):
        self.data_dir = Path(data_dir)
        self.saved_models_dir = Path(saved_models_dir)
        self.device = device
        self.results_dir = self.saved_models_dir / 'evaluation_results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Load data once for all evaluations
        print("Loading data...")
        processor = DataPivotProcessor(data_dir)
        self.pivoted_data, self.static_df, self.metadata = processor.load_pivoted_data()
        self.feature_info = processor.get_feature_info()
        
    def find_model_checkpoint(self, model_path):
        """
        Find the checkpoint file for a given model path.
        Handles both direct .pth files and model directories.
        """
        model_path = Path(model_path)
        
        # Direct .pth file
        if model_path.suffix == '.pth':
            return model_path
        
        # Model directory - look for best_model.pth
        if model_path.is_dir():
            checkpoint_path = model_path / 'checkpoints' / 'best_model.pth'
            if checkpoint_path.exists():
                return checkpoint_path
            
            # Search for any .pth file in checkpoints
            checkpoints_dir = model_path / 'checkpoints'
            if checkpoints_dir.exists():
                pth_files = list(checkpoints_dir.glob('*.pth'))
                if pth_files:
                    # Prefer 'best' in filename
                    for pth_file in pth_files:
                        if 'best' in pth_file.name:
                            return pth_file
                    return pth_files[0]
        
        raise FileNotFoundError(f"Could not find checkpoint file in {model_path}")
        
    def evaluate_single_model(self, model_path, verbose=True):
        """
        Evaluate a single saved model on the test set.
        
        Process:
        1. Load model checkpoint
        2. Recreate model architecture with saved hyperparameters
        3. Load weights
        4. Create test dataloader
        5. Run evaluation and collect metrics
        """
        model_path = Path(model_path)
        
        # Find checkpoint file
        checkpoint_path = self.find_model_checkpoint(model_path)
        
        if verbose:
            print(f"\nEvaluating model: {model_path.name}")
            print(f"Checkpoint: {checkpoint_path.name}")
            print("-" * 60)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract hyperparameters
        hyperparams = checkpoint.get('hyperparams', {})
        model_type = hyperparams.get('model_type', 'cnn')
        
        # Create model with saved hyperparameters
        model = create_model(
            model_type,
            self.feature_info,
            timesteps=hyperparams.get('timesteps', 200),
            latent_dim=hyperparams.get('latent_dim', 32),
            dropout=hyperparams.get('dropout', 0.25)
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        # Create test dataloader
        _, _, test_loader = create_dataloaders_with_test(
            self.pivoted_data, self.static_df, self.metadata,
            batch_size=hyperparams.get('batch_size', 32),
            timesteps=hyperparams.get('timesteps', 200),
            num_workers=4
        )
        
        # Evaluate on test set
        model_dir = model_path if model_path.is_dir() else model_path.parent.parent
        metrics, predictions, targets = evaluate_on_test_set(
            model, test_loader, self.device, save_dir=model_dir
        )
        
        # Add model metadata to metrics
        metrics['model_name'] = model_path.name if model_path.is_dir() else model_path.parent.parent.name
        metrics['model_type'] = model_type
        metrics['hyperparams'] = hyperparams
        metrics['val_metrics'] = checkpoint.get('val_metrics', {})
        metrics['model_path'] = str(model_path)
        
        if verbose:
            print(f"Test RMSLE: {metrics['overall_rmsle']:.4f}")
            print(f"Test R²: {metrics['overall_r2']:.4f}")
            print(f"Test RMSLE (Public): {metrics['public_rmsle']:.4f}")
            print(f"Test RMSLE (Private): {metrics['private_rmsle']:.4f}")
        
        return metrics, predictions, targets
    
    def evaluate_all_models(self):
        """
        Evaluate all saved models in the saved_models directory.
        Creates a comprehensive comparison of all trained models.
        """
        # Find all model directories
        model_dirs = [d for d in self.saved_models_dir.iterdir() if d.is_dir() and d.name != 'evaluation_results']
        
        if not model_dirs:
            print("No saved models found!")
            return None
        
        print(f"Found {len(model_dirs)} saved models")
        
        results = []
        
        # Evaluate each model
        for model_dir in tqdm(model_dirs, desc="Evaluating models"):
            try:
                metrics, _, _ = self.evaluate_single_model(model_dir, verbose=False)
                results.append(metrics)
            except Exception as e:
                print(f"Error evaluating {model_dir.name}: {e}")
                continue
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f'evaluation_results_{timestamp}.json'
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (np.float32, np.float64)):
                    serializable_result[key] = float(value)
                elif isinstance(value, dict):
                    cleaned = {}
                    for k, v in value.items():
                        if isinstance(v, (np.generic,)):
                            cleaned[k] = float(v)
                        elif isinstance(v, list):
                            cleaned[k] = [float(x) for x in v]
                        else:
                            cleaned[k] = v
                    serializable_result[key] = cleaned
                elif isinstance(value, list):
                    serializable_result[key] = [float(i) for i in value]
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Also save as latest for easy access
        latest_file = self.results_dir / 'evaluation_results_latest.json'
        with open(latest_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return results
    
    def create_leaderboard(self, results=None, save_csv=True):
        """
        Create a leaderboard comparing all evaluated models.
        
        Displays:
        - Ranking by test RMSLE
        - Model architecture comparison
        - Hyperparameter impact analysis
        - Comparison with Kaggle benchmark
        """
        if results is None:
            # Load latest results
            latest_file = self.results_dir / 'evaluation_results_latest.json'
            if latest_file.exists():
                with open(latest_file, 'r') as f:
                    results = json.load(f)
            else:
                print("No evaluation results found. Run evaluate_all_models() first.")
                return None
        
        # Create DataFrame for analysis
        rows = []
        for result in results:
            row = {
                'Model': result['model_name'],
                'Type': result['model_type'],
                'Test RMSLE': result['overall_rmsle'],
                'Test R²': result['overall_r2'],
                'Public RMSLE': result['public_rmsle'],
                'Private RMSLE': result['private_rmsle'],
                'Val RMSLE': result['val_metrics'].get('overall_rmsle', np.nan),
                'Learning Rate': result['hyperparams'].get('lr', np.nan),
                'Latent Dim': result['hyperparams'].get('latent_dim', np.nan),
                'Dropout': result['hyperparams'].get('dropout', np.nan),
                'Batch Size': result['hyperparams'].get('batch_size', np.nan)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by test RMSLE (lower is better)
        df = df.sort_values('Test RMSLE')
        
        # Display leaderboard
        print("\n" + "="*100)
        print("MODEL LEADERBOARD (Sorted by Test RMSLE)")
        print("="*100)
        print("\nTop 10 Models:")
        print(df.head(10).to_string(index=False, float_format='%.4f'))
        
        # Save to CSV
        if save_csv:
            csv_file = self.results_dir / 'model_leaderboard.csv'
            df.to_csv(csv_file, index=False)
            print(f"\nLeaderboard saved to: {csv_file}")
        
        # Compare with Kaggle benchmark
        print("\n" + "-"*60)
        print("COMPARISON WITH KAGGLE LEADERBOARD")
        print("-"*60)
        print("5th Place Solution RMSLE: ~0.514")
        print(f"Our Best Model RMSLE: {df['Test RMSLE'].min():.4f}")
        print(f"Difference: {df['Test RMSLE'].min() - 0.514:.4f}")
        
        # Performance by model type
        print("\n" + "-"*60)
        print("PERFORMANCE BY MODEL TYPE")
        print("-"*60)
        type_summary = df.groupby('Type')['Test RMSLE'].agg(['mean', 'min', 'max', 'count'])
        print(type_summary.to_string(float_format='%.4f'))
        
        return df
    
    def plot_predictions(self, model_path, num_samples=5):
        """
        Visualize model predictions vs actual values.
        Shows both public (days 1-5) and private (days 6-16) performance.
        """
        # Evaluate model
        metrics, predictions, targets = self.evaluate_single_model(model_path)
        
        # Get model directory
        model_path = Path(model_path)
        model_dir = model_path if model_path.is_dir() else model_path.parent.parent
        plots_dir = model_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Create visualization
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        # Randomly select samples
        indices = np.random.choice(len(predictions), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            days = np.arange(16)
            
            # Plot actual vs predicted
            ax.plot(days, targets[idx], 'o-', label='Actual', linewidth=2, markersize=8)
            ax.plot(days, predictions[idx], 's-', label='Predicted', linewidth=2, markersize=6)
            
            # Mark public/private split
            ax.axvline(x=4.5, color='red', linestyle='--', alpha=0.5, 
                      label='Public/Private Split')
            
            # Calculate sample metrics
            sample_rmsle = self._calculate_sample_rmsle(predictions[idx], targets[idx])
            public_rmsle = self._calculate_sample_rmsle(predictions[idx][:5], targets[idx][:5])
            private_rmsle = self._calculate_sample_rmsle(predictions[idx][5:], targets[idx][5:])
            
            ax.set_xlabel('Day')
            ax.set_ylabel('Sales')
            ax.set_title(f'Sample {idx}: RMSLE = {sample_rmsle:.4f} (Public: {public_rmsle:.4f}, Private: {private_rmsle:.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = plots_dir / f'predictions_samples_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_file, dpi=150)
        plt.close()
        
        print(f"Predictions plot saved to: {plot_file}")
    
    def _calculate_sample_rmsle(self, pred, true):
        """Calculate RMSLE for a single sample"""
        pred = np.clip(pred, 0, None)
        return np.sqrt(np.mean((np.log1p(pred) - np.log1p(true))**2))
    
    def analyze_errors(self, model_path):
        """
        Comprehensive error analysis across different dimensions.
        
        Analyzes:
        - Error distribution
        - Per-day performance
        - Public vs private split performance
        - Error patterns and correlations
        """
        # Evaluate model
        metrics, predictions, targets = self.evaluate_single_model(model_path)
        
        # Get model directory
        model_path = Path(model_path)
        model_dir = model_path if model_path.is_dir() else model_path.parent.parent
        plots_dir = model_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        logs_dir = model_dir / 'logs'
        logs_dir.mkdir(exist_ok=True)
        
        # Calculate errors
        errors = np.log1p(predictions) - np.log1p(targets)
        abs_errors = np.abs(errors)
        
        # Error statistics
        error_stats = {
            'mean_absolute_error': float(np.mean(abs_errors)),
            'std_errors': float(np.std(errors)),
            'max_absolute_error': float(np.max(abs_errors)),
            'percentile_95_error': float(np.percentile(abs_errors, 95)),
            'percentile_99_error': float(np.percentile(abs_errors, 99))
        }
        
        print("\n" + "="*60)
        print("ERROR ANALYSIS")
        print("="*60)
        
        print(f"\nError Statistics:")
        for key, value in error_stats.items():
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        
        # Per-day analysis
        per_day_stats = []
        print("\nRMSLE by Day:")
        for day in range(16):
            day_rmsle = np.sqrt(np.mean((np.log1p(predictions[:, day]) - 
                                        np.log1p(targets[:, day]))**2))
            day_mae = np.mean(np.abs(predictions[:, day] - targets[:, day]))
            print(f"  Day {day+1}: RMSLE={day_rmsle:.4f}, MAE={day_mae:.4f}")
            per_day_stats.append({
                'day': day + 1,
                'rmsle': float(day_rmsle),
                'mae': float(day_mae)
            })
        
        # Save error analysis
        error_analysis = {
            'error_statistics': error_stats,
            'per_day_statistics': per_day_stats,
            'model_name': str(model_path.name),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(logs_dir / 'error_analysis.json', 'w') as f:
            json.dump(error_analysis, f, indent=2)
        
        # Create error visualization plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Error Analysis', fontsize=16)
        
        # Plot 1: Error distribution
        axes[0, 0].hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Log Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution')
        
        # Plot 2: RMSLE by day
        day_errors = [np.sqrt(np.mean((np.log1p(predictions[:, d]) - 
                                      np.log1p(targets[:, d]))**2)) 
                      for d in range(16)]
        axes[0, 1].plot(range(1, 17), day_errors, 'o-', linewidth=2, markersize=8)
        axes[0, 1].axvline(x=5.5, color='red', linestyle='--', alpha=0.5, label='Public/Private Split')
        axes[0, 1].set_xlabel('Day')
        axes[0, 1].set_ylabel('RMSLE')
        axes[0, 1].set_title('RMSLE by Day')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Predicted vs Actual scatter
        sample_size = min(5000, predictions.size)
        sample_indices = np.random.choice(predictions.size, sample_size, replace=False)
        axes[0, 2].scatter(targets.flatten()[sample_indices], 
                          predictions.flatten()[sample_indices], 
                          alpha=0.3, s=10)
        axes[0, 2].plot([0, targets.max()], [0, targets.max()], 'r--', label='Perfect Prediction')
        axes[0, 2].set_xlabel('Actual Sales')
        axes[0, 2].set_ylabel('Predicted Sales')
        axes[0, 2].set_title('Predicted vs Actual')
        axes[0, 2].legend()
        
        # Plot 4: Residuals vs Predicted
        axes[1, 0].scatter(predictions.flatten()[sample_indices], 
                          errors.flatten()[sample_indices], 
                          alpha=0.3, s=10)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Sales (log scale)')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals vs Predicted')
        
        # Plot 5: Error percentiles by day
        percentiles = [25, 50, 75, 90, 95]
        for p in percentiles:
            day_percentiles = [np.percentile(abs_errors[:, d], p) for d in range(16)]
            axes[1, 1].plot(range(1, 17), day_percentiles, '-', label=f'{p}th percentile')
        axes[1, 1].set_xlabel('Day')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title('Error Percentiles by Day')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Plot 6: Public vs Private performance
        public_errors = errors[:, :5].flatten()
        private_errors = errors[:, 5:].flatten()
        
        axes[1, 2].boxplot([public_errors, private_errors], 
                          labels=['Public (Days 1-5)', 'Private (Days 6-16)'])
        axes[1, 2].set_ylabel('Log Error')
        axes[1, 2].set_title('Public vs Private Error Distribution')
        axes[1, 2].grid(True, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = plots_dir / f'error_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_file, dpi=150)
        plt.close()
        
        print(f"\nError analysis plot saved to: {plot_file}")
        print(f"Error analysis data saved to: {logs_dir / 'error_analysis.json'}")


def main():
    """Main evaluation function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate saved models')
    parser.add_argument('--data_dir', type=str, default='./',
                       help='Directory containing the data')
    parser.add_argument('--saved_models_dir', type=str, default='./saved_models',
                       help='Directory containing saved models')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Specific model to evaluate')
    parser.add_argument('--evaluate_all', action='store_true',
                       help='Evaluate all saved models')
    parser.add_argument('--plot', action='store_true',
                       help='Plot predictions for the model')
    parser.add_argument('--analyze_errors', action='store_true',
                       help='Perform error analysis')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.data_dir, args.saved_models_dir)
    
    if args.evaluate_all:
        # Evaluate all models and create leaderboard
        results = evaluator.evaluate_all_models()
        if results:
            evaluator.create_leaderboard(results)
    
    elif args.model_path:
        # Evaluate specific model
        metrics, _, _ = evaluator.evaluate_single_model(args.model_path)
        
        if args.plot:
            evaluator.plot_predictions(args.model_path, num_samples=5)
        
        if args.analyze_errors:
            evaluator.analyze_errors(args.model_path)
    
    else:
        # Just create leaderboard from existing results
        evaluator.create_leaderboard()


if __name__ == "__main__":
    main()