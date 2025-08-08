import argparse
import torch
import numpy as np
import pandas as pd
import random
from pathlib import Path
import json
from datetime import datetime, timedelta

# Import all project modules with updated names
from data_utils import DataPivotProcessor, prepare_data_for_training
from dataset import create_dataloaders, create_dataloaders_with_test, inspect_batch
from models import create_model
from train import Trainer, make_predictions, evaluate_on_test_set
from hyperparameter_tuning import run_hyperparameter_tuning
from evaluate import ModelEvaluator


def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries.
    Ensures consistent results across runs with the same seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model_directory(model_type, hyperparams, base_dir='./saved_models'):
    """
    Create a unique directory for each training run.
    Directory name includes model type, hyperparameters, and timestamp.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create descriptive directory name with key hyperparameters
    dir_name = (f"{model_type}_"
                f"lr{hyperparams['lr']}_"
                f"dim{hyperparams['latent_dim']}_"
                f"drop{hyperparams['dropout']}_"
                f"bs{hyperparams['batch_size']}_"
                f"{timestamp}")
    
    model_dir = Path(base_dir) / dir_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    return model_dir


def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Configure device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Data Preparation Pipeline
    if args.prepare_data:
        """
        This step performs the critical data transformation:
        1. Loads feature-engineered CSV files
        2. Creates label encoders for categorical variables
        3. Pivots data from long to wide format for efficient access
        4. Saves pivoted data to disk for fast loading
        """
        print("Preparing feature-engineered data...")
        processor = prepare_data_for_training(args.data_dir)
        feature_info = processor.get_feature_info()
        
        print("\nFeature Summary:")
        print(f"- Numerical features: {len(feature_info['numerical_features'])}")
        print(f"- Categorical features: {len(feature_info['categorical_features'])}")
        print(f"- Cyclical features: {len(feature_info['cyclical_features'])}")
        print(f"- Interaction features: {len(feature_info['interaction_features'])}")
        print(f"- Label encoders: {len(feature_info['label_encoder_sizes'])}")
        print("\nData preparation complete!")
        return
    
    # Step 2: Hyperparameter Tuning
    if args.tune:
        """
        Automated hyperparameter optimization using Optuna.
        Tests different combinations of learning rate, latent dimension, dropout, etc.
        """
        print(f"\nStarting hyperparameter tuning for {args.model_type} model...")
        best_result = run_hyperparameter_tuning(
            args.model_type,
            args.data_dir,
            max_epochs=args.tune_epochs,
            patience=args.tune_patience
        )
        print("\nTuning complete! Best hyperparameters saved.")
        return
    
    # Step 3: Model Evaluation
    if args.evaluate_all or args.evaluate_model:
        """
        Evaluate saved models on test set and create leaderboard.
        Includes error analysis and prediction visualization.
        """
        evaluator = ModelEvaluator(args.data_dir, args.saved_models_dir or './saved_models', device)
        
        if args.evaluate_all:
            print("\nEvaluating all saved models...")
            results = evaluator.evaluate_all_models()
            if results:
                evaluator.create_leaderboard(results)
        
        elif args.evaluate_model:
            print(f"\nEvaluating model: {args.evaluate_model}")
            metrics, _, _ = evaluator.evaluate_single_model(args.evaluate_model)
            
            if args.plot_predictions:
                evaluator.plot_predictions(args.evaluate_model, num_samples=5)
            
            if args.analyze_errors:
                evaluator.analyze_errors(args.evaluate_model)
        
        return
    
    # Step 4: Load Pivoted Data
    print("\nLoading pivoted feature-engineered data...")
    processor = DataPivotProcessor(args.data_dir)
    pivoted_data, static_df, metadata = processor.load_pivoted_data()
    feature_info = processor.get_feature_info()
    
    print(f"\nData loaded successfully!")
    print(f"Number of features: {len(pivoted_data)}")
    print(f"Number of store-family combinations: {metadata['n_combinations']}")
    
    # Step 5: Data Inspection (optional)
    if args.inspect_data:
        print("\nInspecting data structure...")
        print("\nPivoted feature shapes:")
        for feat_name, feat_data in list(pivoted_data.items())[:5]:
            print(f"  - {feat_name}: {feat_data.shape}")
        
        print("\nStatic features:")
        print(static_df.head())
    
    # Step 6: Create Data Loaders
    print("\nCreating data loaders...")
    
    # Create train/val/test splits with proper temporal separation
    train_loader, val_loader, test_loader = create_dataloaders_with_test(
        pivoted_data, static_df, metadata,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        num_workers=args.num_workers
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Inspect batch structure
    if args.inspect_data:
        print("\nInspecting a batch:")
        batch = inspect_batch(train_loader)
    
    # Step 7: Create Model
    print(f"\nCreating {args.model_type} model...")
    
    # Print model description based on type
    if args.model_type == 'cnn':
        print("Model: Temporal CNN with dilated convolutions (one-shot prediction)")
    elif args.model_type == 'seq2seq':
        print("Model: Autoregressive Encoder-Decoder GRU (sequential prediction)")
    elif args.model_type == 'multi-step-seq2seq':
        print("Model: One-Shot LSTM-based model (parallel prediction)")
    
    # Model-specific parameters
    model_kwargs = {
        'latent_dim': args.latent_dim,
        'dropout': args.dropout
    }
    
    # Instantiate model with appropriate architecture
    model = create_model(
        args.model_type,
        feature_info,
        timesteps=args.timesteps,
        **model_kwargs
    )
    
    # Display model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Display architecture if verbose
    if args.verbose:
        print("\nModel Architecture:")
        print(model)
    
    # Validate data pipeline
    if args.train or args.validate_data:
        print("\nValidating data pipeline...")
        from debug_utils import validate_dataloader, debug_model_forward
        
        # Validate data loader
        batch = validate_dataloader(train_loader, feature_info)
        
        # Test forward pass
        print("\nTesting model forward pass...")
        debug_model_forward(model, batch, device)
        
        if args.validate_data and not args.train:
            print("\nValidation complete. Exiting.")
            return
    
    # Step 8: Model Training
    if args.train:
        """
        Main training loop with:
        - RMSLE loss (competition metric)
        - Early stopping
        - Learning rate scheduling
        - Comprehensive logging and visualization
        """
        print("\nStarting training...")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"Batch size: {args.batch_size}")
        print(f"Loss criterion: {args.criterion}")
        
        # Create hyperparameters dictionary
        hyperparams = {
            'model_type': args.model_type,
            'timesteps': args.timesteps,
            'latent_dim': args.latent_dim,
            'dropout': args.dropout,
            'batch_size': args.batch_size,
            'lr': args.lr
        }
        
        # Create unique directory for this training run
        model_dir = create_model_directory(args.model_type, hyperparams, args.saved_models_dir)
        print(f"\nModel directory: {model_dir}")
        
        # Initialize trainer
        trainer = Trainer(model, device=device, model_dir=str(model_dir))
        
        # Log experiment configuration
        experiment_config = {
            **hyperparams,
            'epochs': args.epochs,
            'criterion': args.criterion,
            'feature_count': len(pivoted_data),
            'static_features': len(static_df.columns),
            'timestamp': datetime.now().isoformat(),
            'model_directory': str(model_dir),
            'seed': args.seed
        }
        
        config_path = model_dir / 'experiment_config.json'
        with open(config_path, 'w') as f:
            json.dump(experiment_config, f, indent=2)
        
        # Train model
        train_losses, val_losses = trainer.train(
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            criterion=args.criterion,
            patience=args.patience,
            hyperparams=hyperparams
        )
        
        print("\nTraining complete!")
        print(f"Best validation RMSLE: {trainer.best_val_rmsle:.4f}")
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics, test_predictions, test_targets = evaluate_on_test_set(
            model, test_loader, device, save_dir=model_dir
        )
        
        print("\nTest Set Results:")
        print(f"Test RMSLE: {test_metrics['overall_rmsle']:.4f}")
        print(f"Test R²: {test_metrics['overall_r2']:.4f}")
        print(f"Test RMSLE (Public): {test_metrics['public_rmsle']:.4f}")
        print(f"Test RMSLE (Private): {test_metrics['private_rmsle']:.4f}")
        
        # Save final results
        test_results = {
            'test_metrics': {
                **{k: float(v) for k, v in test_metrics.items()
                    if not isinstance(test_metrics[k], list)},
                **{k: [float(i) for i in test_metrics[k]]
                    for k in test_metrics if isinstance(test_metrics[k], list)}
            },
            'val_best_rmsle': float(trainer.best_val_rmsle),
            'experiment_config': experiment_config
        }
        
        test_results_path = model_dir / 'logs' / 'final_test_results.json'
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\nAll outputs saved to: {model_dir}")
    
    # Step 9: Inference with Saved Model
    if args.predict:
        """
        Load a trained model and generate predictions.
        Compares performance with Kaggle leaderboard benchmark.
        """
        print("\nLoading best model for inference...")
        
        # Determine model path
        if args.model_path:
            checkpoint_path = Path(args.model_path)
            if 'checkpoints' in str(checkpoint_path):
                model_dir = checkpoint_path.parent.parent
            else:
                model_dir = checkpoint_path.parent
        else:
            # Find most recent model
            saved_models_dir = Path(args.saved_models_dir)
            if saved_models_dir.exists():
                model_dirs = sorted(saved_models_dir.glob('*/'), key=lambda x: x.stat().st_mtime, reverse=True)
                if model_dirs:
                    model_dir = model_dirs[0]
                    checkpoint_path = model_dir / 'checkpoints' / 'best_model.pth'
                else:
                    print(f"No models found in {saved_models_dir}")
                    return
            else:
                print(f"Saved models directory not found: {saved_models_dir}")
                return
        
        if not checkpoint_path.exists():
            print(f"Model not found at {checkpoint_path}")
            return
        
        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )
        
        # Extract hyperparameters
        hyperparams = checkpoint.get('hyperparams', {})
        model_type = hyperparams.get('model_type', args.model_type)
        
        # Create model with saved hyperparameters
        model = create_model(
            model_type,
            feature_info,
            timesteps=hyperparams.get('timesteps', args.timesteps),
            latent_dim=hyperparams.get('latent_dim', args.latent_dim),
            dropout=hyperparams.get('dropout', args.dropout)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        print(f"Loaded model from epoch {checkpoint['epoch']} with val RMSLE {checkpoint.get('best_val_rmsle', 'N/A')}")
        
        # Run inference
        print("\nRunning inference on test set...")
        test_metrics, test_predictions, test_targets = evaluate_on_test_set(
            model, test_loader, device, save_dir=model_dir
        )
        
        print("\nTest Set Results:")
        print(f"Test RMSLE: {test_metrics['overall_rmsle']:.4f}")
        print(f"Test R²: {test_metrics['overall_r2']:.4f}")
        print(f"Test RMSLE (Public): {test_metrics['public_rmsle']:.4f}")
        print(f"Test RMSLE (Private): {test_metrics['private_rmsle']:.4f}")
        
        # Compare with competition benchmark
        print("\n" + "-"*60)
        print("COMPARISON WITH KAGGLE LEADERBOARD")
        print("-"*60)
        print("5th Place Solution RMSLE: ~0.514")
        print(f"Our Model RMSLE: {test_metrics['overall_rmsle']:.4f}")
        print(f"Difference: {test_metrics['overall_rmsle'] - 0.514:.4f}")
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Favorita Sales Forecasting with Deep Learning')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./', 
                       help='Directory containing train-final.csv and test-final.csv')
    parser.add_argument('--prepare_data', action='store_true', 
                       help='Prepare and pivot the feature-engineered data')
    parser.add_argument('--inspect_data', action='store_true', 
                       help='Inspect data structure and batches')
    parser.add_argument('--validate_data', action='store_true', 
                       help='Validate data pipeline and model forward pass')
    
    # Model arguments - updated choices to reflect actual behavior
    parser.add_argument('--model_type', type=str, default='cnn', 
                       choices=['cnn', 'seq2seq', 'multi-step-seq2seq'],
                       help='Model type: cnn (one-shot CNN), seq2seq (autoregressive), multi-step-seq2seq (one-shot LSTM)')
    parser.add_argument('--timesteps', type=int, default=200, 
                       help='Number of historical timesteps')
    parser.add_argument('--latent_dim', type=int, default=32, 
                       help='Latent dimension for models')
    parser.add_argument('--dropout', type=float, default=0.25, 
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--train', action='store_true', 
                       help='Train model')
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--criterion', type=str, default='rmsle', choices=['mse', 'rmsle'],
                       help='Loss criterion')
    parser.add_argument('--patience', type=int, default=5, 
                       help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of data loader workers')
    
    # Hyperparameter tuning arguments
    parser.add_argument('--tune', action='store_true',
                       help='Run hyperparameter tuning')
    parser.add_argument('--tune_epochs', type=int, default=10,
                       help='Max epochs for each tuning experiment')
    parser.add_argument('--tune_patience', type=int, default=3,
                       help='Early stopping patience for tuning')
    
    # Evaluation arguments
    parser.add_argument('--evaluate_all', action='store_true',
                       help='Evaluate all saved models')
    parser.add_argument('--evaluate_model', type=str, default=None,
                       help='Path to specific model to evaluate')
    parser.add_argument('--plot_predictions', action='store_true',
                       help='Plot predictions for evaluated model')
    parser.add_argument('--analyze_errors', action='store_true',
                       help='Perform error analysis for evaluated model')
    
    # Prediction arguments
    parser.add_argument('--predict', action='store_true', 
                       help='Generate predictions using saved model')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to specific model for prediction')
    
    # Other arguments
    parser.add_argument('--saved_models_dir', type=str, default='./saved_models',
                       help='Base directory for saved models')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--no_cuda', action='store_true', 
                       help='Disable CUDA')
    parser.add_argument('--verbose', action='store_true', 
                       help='Print detailed information')
    
    args = parser.parse_args()
    
    # Usage examples with corrected descriptions:
    # 1. Prepare data: python main.py --prepare_data --data_dir ../data/
    # 2. Train CNN (one-shot): python main.py --train --model_type cnn --lr 0.003 --latent_dim 64 --dropout 0.2 --epochs 20
    # 3. Train Autoregressive: python main.py --train --model_type seq2seq --lr 0.001 --latent_dim 100 --dropout 0.3 --epochs 20
    # 4. Train One-Shot LSTM: python main.py --train --model_type multi-step-seq2seq --lr 0.001 --latent_dim 100 --dropout 0.3 --epochs 20
    # 5. Evaluate all: python main.py --evaluate_all --data_dir ../data/
    # 6. Run inference: python main.py --predict --model_path saved_models/cnn_xxx/checkpoints/best_model.pth
    
    main(args)