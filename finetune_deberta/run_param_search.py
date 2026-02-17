import os
import json
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from scipy.stats import spearmanr
from peft import LoraConfig, get_peft_model, TaskType

from dataset import SimplePlausibilityDataset
from model import create_model

from peft import PeftModel


# ============================================================================
# PARAMETER GRID - Enhanced with new options
# ============================================================================
"""PARAM_GRID = {
    'batch_size': [4], 
    'epochs': [5],
    'learning_rate': [1e-4],
    'pooling': ['cls', 'mean', 'attention'],  
    'loss_type': ['mse', 'huber'],
    'huber_delta': [1.0],
    'ranking_weight': [0, 0.25, 0.5],
    'use_uncertainty_loss': [True, False],  
    'uncertainty_weight': [0.1, 0.3, 0.5],
    'lora_r': [4, 8, 12],
    'lora_alpha': [32],
    'model_name': ['deberta-large', 'roberta-large'],  
    'warmup_ratio': [0.1],  
    'weight_decay': [0.01],  
    'lora_dropout': [0.1],  
}"""

PARAM_GRID = {
    'batch_size': [8], 
    'epochs': [10],
    'learning_rate': [1e-4],
    'pooling': ['mean'],  
    'loss_type': ['mse'],
    'huber_delta': [1.0],
    'ranking_weight': [0.25],
    'use_uncertainty_loss': [True],  
    'uncertainty_weight': [0],
    'lora_r': [8],
    'lora_alpha': [32],
    'model_name': ['deberta-large'],  
    'warmup_ratio': [0.1],  
    'weight_decay': [0.01],  
    'lora_dropout': [0.1],  
}


# Constants
USE_RANKING_LOSS = True
USE_LORA = True
MODELS = {
    "deberta-large": "microsoft/deberta-v3-large",
    "roberta-large": "FacebookAI/roberta-large",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_param_combinations():
    """Generate all parameter combinations, handling conditional parameters"""
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    all_combinations = list(itertools.product(*values))
    
    # Filter out invalid combinations
    valid_combinations = []
    for combo in all_combinations:
        params = dict(zip(keys, combo))
        
        # Skip invalid: huber_delta only matters when loss_type='huber'
        if params['loss_type'] != 'huber' and params['huber_delta'] != 1.0:
            continue
        
        # Skip invalid: uncertainty_weight only matters when use_uncertainty_loss=True
        if not params['use_uncertainty_loss'] and params['uncertainty_weight'] != 0.3:
            continue
        
        valid_combinations.append(params)
    
    return valid_combinations


def create_experiment_name(params):
    """Create a unique experiment name from parameters"""
    name_parts = [
        f"model_{params['model_name']}",
        f"bs{params['batch_size']}",
        f"ep{params['epochs']}",
        f"lr{params['learning_rate']:.0e}",
        f"pool_{params['pooling']}",
        f"loss_{params['loss_type']}",
    ]
    
    if params['loss_type'] == 'huber':
        name_parts.append(f"hd{params['huber_delta']}")
    
    name_parts.append(f"rw{params['ranking_weight']}")
    
    # NEW: Add uncertainty loss info
    if params['use_uncertainty_loss']:
        name_parts.append(f"unc{params['uncertainty_weight']}")
    
    name_parts.extend([
        f"r{params['lora_r']}",
        f"a{params['lora_alpha']}",
    ])
    
    return "_".join(name_parts)


def compute_metrics_with_std(eval_pred, std_devs):
    """Compute evaluation metrics including Acc@Std"""
    predictions, labels = eval_pred
    predictions = predictions.squeeze() if predictions.ndim > 1 else predictions

    spearman, _ = spearmanr(predictions, labels)
    mae = np.mean(np.abs(predictions - labels))
    rmse = np.sqrt(np.mean((predictions - labels) ** 2))
    acc_05 = np.mean(np.abs(predictions - labels) <= 0.5)
    acc_10 = np.mean(np.abs(predictions - labels) <= 1.0)

    std_devs = np.maximum(std_devs, 1.0)
    within_std = np.abs(predictions - labels) <= std_devs
    acc_std = np.mean(within_std)

    return {
        "spearman": spearman,
        "mae": mae,
        "rmse": rmse,
        "acc_0.5": acc_05,
        "acc_1.0": acc_10,
        "acc_std": acc_std,
    }


def apply_lora(model, params):
    """Apply LoRA to model"""
    if "deberta" in params['model_name']:
        target_modules = ["query_proj", "key_proj", "value_proj"]
    else:
        target_modules = ["query", "key", "value"]

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=params['lora_r'],
        lora_alpha=params['lora_alpha'],
        lora_dropout=params['lora_dropout'],
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    return model


# ============================================================================
# CUSTOM TRAINER FOR PER-EPOCH LOGGING
# ============================================================================

class MetricsTracker(Trainer):
    """Custom Trainer that tracks metrics at each epoch"""
    
    def __init__(self, *args, epoch_metrics_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_metrics_file = epoch_metrics_file
        self.epoch_metrics = []
    
    def log(self, logs, start_time=None):
        """Override log to capture epoch-level metrics"""
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        
        # Check if this is an epoch end (has eval metrics)
        if 'eval_spearman' in logs:
            epoch_data = {
                'epoch': logs.get('epoch', 0),
                'train_loss': logs.get('loss', None),
                'eval_loss': logs.get('eval_loss', None),
                'eval_spearman': logs.get('eval_spearman', None),
                'eval_mae': logs.get('eval_mae', None),
                'eval_rmse': logs.get('eval_rmse', None),
                'eval_acc_0.5': logs.get('eval_acc_0.5', None),
                'eval_acc_1.0': logs.get('eval_acc_1.0', None),
                'eval_acc_std': logs.get('eval_acc_std', None),
            }
            self.epoch_metrics.append(epoch_data)
            
            # Save to file immediately
            if self.epoch_metrics_file:
                df = pd.DataFrame(self.epoch_metrics)
                df.to_csv(self.epoch_metrics_file, index=False)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_with_params(params, train_file, dev_file, base_output_dir):
    """Train model with given parameters"""
    
    experiment_name = create_experiment_name(params)
    output_dir = os.path.join(base_output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*100)
    print(f"EXPERIMENT: {experiment_name}")
    print("="*100)
    print("Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()
    
    # Save parameters
    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    
    # Load tokenizer and datasets
    model_path = MODELS[params['model_name']]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    train_dataset = SimplePlausibilityDataset(train_file, tokenizer)
    eval_dataset = SimplePlausibilityDataset(dev_file, tokenizer)
    eval_std_devs = getattr(eval_dataset, "std_devs", None)
    
    # Create model with new parameters
    model = create_model(
        model_path,
        freeze_transformer=True,
        pooling=params['pooling'],
        loss_type=params['loss_type'],
        huber_delta=params['huber_delta'],
        use_ranking_loss=USE_RANKING_LOSS,
        ranking_weight=params['ranking_weight'],
        use_uncertainty_loss=params['use_uncertainty_loss'],  # NEW
        uncertainty_weight=params['uncertainty_weight'],  # NEW
    )
    
    # Apply LoRA
    if USE_LORA:
        model = apply_lora(model, params)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)\n")
    
    # Prepare metrics function
    if eval_std_devs is not None:
        def metrics_fn(eval_pred):
            return compute_metrics_with_std(eval_pred, eval_std_devs)
    else:
        def metrics_fn(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.squeeze() if predictions.ndim > 1 else predictions
            spearman, _ = spearmanr(predictions, labels)
            mae = np.mean(np.abs(predictions - labels))
            rmse = np.sqrt(np.mean((predictions - labels) ** 2))
            return {"spearman": spearman, "mae": mae, "rmse": rmse}
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=params['epochs'],
        per_device_train_batch_size=params['batch_size'],
        per_device_eval_batch_size=params['batch_size'] * 2,
        learning_rate=params['learning_rate'],
        warmup_ratio=params['warmup_ratio'],
        weight_decay=params['weight_decay'],
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="spearman",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
        seed=42,
    )
    
    # Create trainer with custom metrics tracker
    epoch_metrics_file = os.path.join(output_dir, 'epoch_metrics.csv')
    trainer = MetricsTracker(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metrics_fn,
        epoch_metrics_file=epoch_metrics_file,
    )
    
    # Train
    try:
        trainer.train()
        
        # Final evaluation
        final_metrics = trainer.evaluate()
        
        # Save final metrics
        with open(os.path.join(output_dir, 'final_metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # Save model
        #trainer.save_model(os.path.join(output_dir, 'final_model'))
        #tokenizer.save_pretrained(os.path.join(output_dir, 'final_model'))
        
        # ------------------------------------------------------------------
        # Save MERGED model (easy to load later)
        # ------------------------------------------------------------------
        merged_dir = os.path.join(output_dir, "final_model_merged")
        os.makedirs(merged_dir, exist_ok=True)

        model_to_save = trainer.model

        # If using LoRA, merge it into the base model
        if USE_LORA:
            model_to_save = model_to_save.merge_and_unload()

        model_to_save.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

        print(f"Saved merged model to: {merged_dir}")



        print("\n" + "="*100)
        print("FINAL RESULTS:")
        print(f"  Spearman: {final_metrics.get('eval_spearman', 0):.4f}")
        print(f"  MAE: {final_metrics.get('eval_mae', 0):.4f}")
        print(f"  Acc@Std: {final_metrics.get('eval_acc_std', 0):.4f}")
        print("="*100 + "\n")
        
        return {
            'experiment_name': experiment_name,
            'success': True,
            'final_metrics': final_metrics,
            'params': params,
        }
        
    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'experiment_name': experiment_name,
            'success': False,
            'error': str(e),
            'params': params,
        }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_experiment_metrics(output_dir, experiment_name):
    """Create line plots for a single experiment"""
    
    metrics_file = os.path.join(output_dir, experiment_name, 'epoch_metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"No metrics file found for {experiment_name}")
        return
    
    df = pd.read_csv(metrics_file)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Metrics: {experiment_name}', fontsize=14, fontweight='bold')
    
    # Plot 1: Loss
    if 'train_loss' in df.columns and df['train_loss'].notna().any():
        axes[0, 0].plot(df['epoch'], df['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    if 'eval_loss' in df.columns and df['eval_loss'].notna().any():
        axes[0, 0].plot(df['epoch'], df['eval_loss'], 'r-s', label='Eval Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss over Epochs')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Spearman Correlation
    if 'eval_spearman' in df.columns:
        axes[0, 1].plot(df['epoch'], df['eval_spearman'], 'g-o', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Spearman Correlation')
        axes[0, 1].set_title('Spearman Correlation over Epochs')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: MAE
    if 'eval_mae' in df.columns:
        axes[0, 2].plot(df['epoch'], df['eval_mae'], 'purple', marker='o', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('MAE')
        axes[0, 2].set_title('MAE over Epochs')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: RMSE
    if 'eval_rmse' in df.columns:
        axes[1, 0].plot(df['epoch'], df['eval_rmse'], 'orange', marker='o', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('RMSE over Epochs')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Acc@Std
    if 'eval_acc_std' in df.columns:
        axes[1, 1].plot(df['epoch'], df['eval_acc_std'], 'brown', marker='o', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Acc@Std')
        axes[1, 1].set_title('Accuracy Within Std Dev over Epochs')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Acc@0.5 and Acc@1.0
    if 'eval_acc_0.5' in df.columns:
        axes[1, 2].plot(df['epoch'], df['eval_acc_0.5'], 'cyan', marker='o', label='Acc@0.5', linewidth=2)
    if 'eval_acc_1.0' in df.columns:
        axes[1, 2].plot(df['epoch'], df['eval_acc_1.0'], 'magenta', marker='s', label='Acc@1.0', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Accuracy Metrics over Epochs')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, experiment_name, 'training_curves.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plots to: {plot_file}")


def create_comparison_visualizations(results_df, output_dir):
    """Create comparison visualizations across all experiments"""
    
    # 1. Top 20 experiments by Spearman
    fig, ax = plt.subplots(figsize=(14, 10))
    top_20 = results_df.nlargest(20, 'spearman')
    
    y_pos = np.arange(len(top_20))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_20)))
    
    bars = ax.barh(y_pos, top_20['spearman'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_20['experiment_name'], fontsize=8)
    ax.set_xlabel('Spearman Correlation', fontsize=12)
    ax.set_title('Top 20 Experiments by Spearman Correlation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_20['spearman'])):
        ax.text(value + 0.005, i, f'{value:.4f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_20_spearman.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved top 20 comparison to: {os.path.join(output_dir, 'top_20_spearman.png')}")
    
    # 2. Pooling strategy comparison
    if 'pooling' in results_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        pooling_comparison = results_df.groupby('pooling')['spearman'].apply(list)
        ax.boxplot(pooling_comparison.values, labels=pooling_comparison.index)
        ax.set_ylabel('Spearman Correlation', fontsize=12)
        ax.set_xlabel('Pooling Strategy', fontsize=12)
        ax.set_title('Pooling Strategy Impact on Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pooling_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved pooling comparison to: {os.path.join(output_dir, 'pooling_comparison.png')}")
    
    # 3. Uncertainty loss comparison
    if 'use_uncertainty_loss' in results_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        unc_comparison = results_df.groupby('use_uncertainty_loss')['spearman'].apply(list)
        ax.boxplot(unc_comparison.values, labels=['Without Uncertainty', 'With Uncertainty'])
        ax.set_ylabel('Spearman Correlation', fontsize=12)
        ax.set_title('Uncertainty Loss Impact on Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'uncertainty_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved uncertainty comparison to: {os.path.join(output_dir, 'uncertainty_comparison.png')}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive Parameter Search with Attention Pooling & Uncertainty Loss')
    parser.add_argument('--train_file', type=str, default='data/train.json',
                       help='Training data file')
    parser.add_argument('--dev_file', type=str, default='data/dev.json',
                       help='Development data file')
    parser.add_argument('--output_dir', type=str, default='results/parameter_search',
                       help='Base output directory')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous run (skip completed experiments)')
    parser.add_argument('--compare_and_visualise', action='store_true',
                       help='Resume from previous run (skip completed experiments)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate all parameter combinations
    param_combinations = generate_param_combinations()
    
    print("="*100)
    print("COMPREHENSIVE PARAMETER SEARCH")
    print("NEW FEATURES: Attention Pooling + Uncertainty-Aware Loss")
    print("="*100)
    print(f"Total experiments: {len(param_combinations)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Training data: {args.train_file}")
    print(f"Dev data: {args.dev_file}")
    print("="*100 + "\n")
    
    # Track results
    all_results = []
    completed_count = 0
    failed_count = 0
    
    # Run experiments
    for idx, params in enumerate(param_combinations, 1):
        print(f"\n{'#'*100}")
        print(f"# EXPERIMENT {idx}/{len(param_combinations)}")
        print(f"{'#'*100}\n")
        
        # Check if already completed
        experiment_name = create_experiment_name(params)
        final_metrics_path = os.path.join(args.output_dir, experiment_name, 'final_metrics.json')
        
        if args.resume and os.path.exists(final_metrics_path):
            print(f"Skipping {experiment_name} (already completed)")
            with open(final_metrics_path, 'r') as f:
                final_metrics = json.load(f)
            result = {
                'experiment_name': experiment_name,
                'success': True,
                'final_metrics': final_metrics,
                'params': params,
            }
            all_results.append(result)
            completed_count += 1
            continue
        
        # Run experiment
        result = train_with_params(params, args.train_file, args.dev_file, args.output_dir)
        all_results.append(result)
        
        if result['success']:
            completed_count += 1
            # Create plots for this experiment
            plot_experiment_metrics(args.output_dir, result['experiment_name'])
        else:
            failed_count += 1
        
        # Save progress
        progress_file = os.path.join(args.output_dir, 'progress.json')
        with open(progress_file, 'w') as f:
            json.dump({
                'total': len(param_combinations),
                'completed': completed_count,
                'failed': failed_count,
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
    

    if args.compare_and_visualise:
        # ========================================================================
        # CREATE COMPREHENSIVE COMPARISON
        # ========================================================================
        
        print("\n" + "="*100)
        print("CREATING COMPARISON TABLES AND VISUALIZATIONS")
        print("="*100 + "\n")
        
        # Extract results into DataFrame
        results_data = []
        for result in all_results:
            if result['success']:
                row = {
                    'experiment_name': result['experiment_name'],
                    'spearman': result['final_metrics'].get('eval_spearman', 0),
                    'mae': result['final_metrics'].get('eval_mae', 0),
                    'rmse': result['final_metrics'].get('eval_rmse', 0),
                    'acc_0.5': result['final_metrics'].get('eval_acc_0.5', 0),
                    'acc_1.0': result['final_metrics'].get('eval_acc_1.0', 0),
                    'acc_std': result['final_metrics'].get('eval_acc_std', 0),
                }
                # Add parameter values
                row.update(result['params'])
                results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        # Sort by Spearman (descending)
        results_df = results_df.sort_values('spearman', ascending=False)
        
        # Save full results
        full_results_file = os.path.join(args.output_dir, 'all_results.csv')
        results_df.to_csv(full_results_file, index=False)
        print(f"Saved all results to: {full_results_file}")
        
        # Save top 20 results
        top_20_file = os.path.join(args.output_dir, 'top_20_results.csv')
        results_df.head(20).to_csv(top_20_file, index=False)
        print(f"Saved top 20 results to: {top_20_file}")
        
        # Create summary statistics
        summary_stats = {
            'total_experiments': len(param_combinations),
            'successful': completed_count,
            'failed': failed_count,
            'best_spearman': float(results_df['spearman'].max()),
            'mean_spearman': float(results_df['spearman'].mean()),
            'std_spearman': float(results_df['spearman'].std()),
            'best_experiment': results_df.iloc[0]['experiment_name'],
            'best_params': results_df.iloc[0][list(PARAM_GRID.keys())].to_dict(),
        }
        
        summary_file = os.path.join(args.output_dir, 'summary_statistics.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        print(f"Saved summary statistics to: {summary_file}")
        
        # Create visualizations
        create_comparison_visualizations(results_df, args.output_dir)
        
        # Print summary
        print("\n" + "="*100)
        print("FINAL SUMMARY")
        print("="*100)
        print(f"Total experiments: {len(param_combinations)}")
        print(f"Successful: {completed_count}")
        print(f"Failed: {failed_count}")
        print(f"\nBest Spearman: {summary_stats['best_spearman']:.4f}")
        print(f"Mean Spearman: {summary_stats['mean_spearman']:.4f} +/- {summary_stats['std_spearman']:.4f}")
        print(f"\nBEST EXPERIMENT: {summary_stats['best_experiment']}")
        print(f"   Model path: {os.path.join(args.output_dir, summary_stats['best_experiment'], 'final_model')}")
        print("\nBest parameters:")
        for key, value in summary_stats['best_params'].items():
            print(f"  {key}: {value}")
        print("="*100 + "\n")
        
        print("Parameter search complete!")
        print(f"All results saved to: {args.output_dir}")
        print(f"Check visualizations: {args.output_dir}/*.png")
        print(f"Check summaries: {args.output_dir}/*.csv")


if __name__ == "__main__":
    main()