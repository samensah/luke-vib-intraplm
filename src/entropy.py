import argparse
import os
import torch
from pathlib import Path
import pickle
from types import SimpleNamespace
from itertools import product
import warnings
import random

from transformers import AutoConfig, AutoTokenizer
from transformers.utils import logging

from model import REModelForMetrics
from data_processor import TACREDProcessor
from train import REDataset, data_collator

from utils.metrics.metric_calling import (
    EvaluationMetricSpecifications, 
    calculate_and_save_layerwise_metrics
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()


# ==========================================
# 1. Custom Specification Wrapper
# ==========================================
class REModelSpecifications:
    """Model Specs """
    def __init__(self, 
                 checkpoint_path="outputs/rb_tacred/0/checkpoint-1", 
                 ):
        path_parts = Path(checkpoint_path).parts
        self.model_family = path_parts[-3]
        self.checkpoint_path = Path(checkpoint_path)


# ==========================================
# 2. Data Loading Helper
# ==========================================
def get_dataloader(args, tokenizer, split='test', num_samples=1000, batch_size=32):
    """
    Encapsulates the TACRED dataset loading logic.
    """
    # Create namespace expected by TACREDProcessor
    processor_args = SimpleNamespace(
        data_dir=args.data_dir,
        input_format=args.input_format,
        max_seq_length=args.max_seq_length,
    )

    processor = TACREDProcessor(processor_args, tokenizer)
    
    # Load specific split
    input_file = f"{args.data_dir}/{split}.json"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Could not find dataset split at {input_file}")
        
    features = processor.read(input_file)
    
    random.seed(args.seed)
    random.shuffle(features)

    # Optional slicing for debugging/speed
    if num_samples is not None and num_samples > 0:
        features = features[:num_samples]

    dataset = REDataset(features)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
        # num_workers=4 # Optional: Enable for faster data loading
    )
    
    return dataloader

# ==========================================
# 3. Main Evaluation Logic
# ==========================================
def run_re_metrics(model, tokenizer, model_specs, args):
    
    # Define metrics to run
    metrics = ['prompt-entropy', 'dataset-entropy', 'curvature']
    splits = ['test', 'test_cf'] # Add 'text_cf', 'dev' or 'train' if needed
    
    # Metrics that require standard dataloader (no special augmentations implemented yet for RE)
    # Note: LIDAR usually requires augmentations, but for RE we verify logic on standard input first 
    # or assume the user adds augmentation logic to REDataset later.

    # Extract Dataset Name (e.g., "tacred" from "./data/tacred")
    dataset_name = Path(args.data_dir).name
    normalizations = ['maxEntropy', 'logN', 'logD', 'logNlogD', 'raw', 'length']
    
    for split, metric in product(splits, metrics):
        try:
            print(f"\n--- Running evaluation for {metric} on {split} set ---")
            
            # Define Metric Specifications
            if metric == 'prompt-entropy':
                evaluation_metric_specs = EvaluationMetricSpecifications(
                    evaluation_metric=metric,
                    num_samples=args.num_samples,
                    alpha=1.0,
                    normalizations=normalizations
                )
            elif metric == 'dataset-entropy':
                evaluation_metric_specs = EvaluationMetricSpecifications(
                    evaluation_metric=metric,
                    num_samples=args.num_samples,
                    alpha=1.0,
                    normalizations=normalizations
                )
            elif metric == 'curvature':
                evaluation_metric_specs = EvaluationMetricSpecifications(
                    evaluation_metric=metric,
                    num_samples=args.num_samples,
                    curvature_k=1
                )
            else:
                continue

            # Define Base Directory (inside the checkpoint folder)
            metrics_results_dir = Path(model_specs.checkpoint_path).parent / 'metrics'
            metrics_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Construct Detailed Filename
            # Format: {dataset}_{model}_{split}_{metric}_N{samples}_L{length}_B{batch}.pkl
            filename = (
                f"{dataset_name}_"
                f"{model_specs.model_family}_"
                f"{split}_"
                f"{metric}_"
                f"N{args.num_samples}_"  # N = Number of samples
                f"L{args.max_seq_length}_"    # L = Max Length
                f"B{args.batch_size}"     # B = Batch Size
                f".pkl"
            )

            results_path = metrics_results_dir / filename
            
            # Check existence
            if results_path.exists() and not args.overwrite:
                print(f"Skipping {metric}: Results exist at {results_path}")
                continue

            # Get Dataloader
            dataloader = get_dataloader(
                args, 
                tokenizer, 
                split=split, 
                num_samples=args.num_samples, 
                batch_size=args.batch_size
            )

            # Calculate and Save
            # Note: ensure calculate_and_save_layerwise_metrics supports REModelForMetrics
            # specifically how it extracts hidden states.
            results = calculate_and_save_layerwise_metrics(
                model=model, 
                dataloader=dataloader, 
                evaluation_metric_specs=evaluation_metric_specs, 
            )

            # Save results
            print(f"Saving results to: {results_path}")
            with open(results_path, "wb") as f:
                pickle.dump(results, f)

            # Optional: Print summary to console
            if 'maxEntropy' in results:
                print(f"Result ({metric}): {results['maxEntropy']}")
            elif 'raw' in results:
                print(f"Result ({metric}): {results['raw']}")

        except Exception as e:
            print(f"Error running evaluation for {metric} - {split}: {str(e)}")
            raise e

# ==========================================
# 4. CLI Entry Point
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run metrics on TACRED RE Model")
    # Model Args
    parser.add_argument('--checkpoint_path', type=str, default="outputs/rb_tacred/-1/checkpoint-2000", help="Path to model checkpoint")
    # Data Args
    parser.add_argument('--data_dir', type=str, default="./data/tacred", help="Directory containing tacred json files")
    parser.add_argument('--input_format', type=str, default="typed_entity_marker_punct", help="Input format for processor")
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--num_samples', type=int, default=1000, help="Number of samples to evaluate (set to 0 for all)")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=23)
    # Execution Args
    parser.add_argument('--device', type=str, default='cpu', help="cuda or cpu")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing result files")
    return parser.parse_args()

# ==========================================
# 5. Main Execution
# ==========================================
def main():
    args = parse_args()
    
    print(f"Loading configuration from {args.checkpoint_path}...")
    config = AutoConfig.from_pretrained(args.checkpoint_path)
    
    print("Loading Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    
    model = REModelForMetrics.from_pretrained(
        args.checkpoint_path,
        config=config,
        use_safetensors=True,
        attn_implementation="eager"
    )
    model.to(args.device)
    model.eval()
    
    # Setup Specs for File Naming
    model_specs = REModelSpecifications(args.checkpoint_path)
    
    # Run loop
    run_re_metrics(model, tokenizer, model_specs, args)

if __name__ == "__main__":
    main()
    # Usage: 
    # python src/entropy.py --checkpoint_path outputs/rb_tacred/-1/checkpoint-2000 --data_dir data/tacred --num_samples 1000 --batch_size 32 --device cuda --overwrite 