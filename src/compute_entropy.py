"""
Compute layer-wise dataset entropy for REModel on test sets.
Usage: python compute_entropy.py --checkpoint_path outputs/lb_retacred/-2/checkpoint-3500 --split test
"""

import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data_processor import TACREDProcessor
from model import REModel
from transformers import AutoConfig, AutoTokenizer
from sklearn.model_selection import train_test_split

# Import from main training script
import sys
sys.path.insert(0, os.path.dirname(__file__))
from train import REDataset, data_collator


def load_model_and_tokenizer(checkpoint_path):
    """Load model and tokenizer from checkpoint."""
    config = AutoConfig.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = REModel.from_pretrained(
        checkpoint_path, 
        config=config,
        use_safetensors=True,
        attn_implementation="eager"
    )
    return model, tokenizer, config


def load_dataset_split(data_dir, split_name, args, tokenizer, processor):
    """Load a specific dataset split."""
    file_map = {
        # 'dev': 'dev.json',
        'test': 'test.json',
        'cf_test': 'test_cf.json',
    }
    
    if split_name not in file_map:
        raise ValueError(f"Unknown split: {split_name}")
    
    file_path = os.path.join(data_dir, file_map[split_name])
    features = processor.read(file_path)
    return features


def stratified_sample(features, n_samples, seed=42):
    """Stratified sampling by relation type (labels)."""
    labels = np.array([feature['labels'] for feature in features])
    indices = np.arange(len(features))
    
    # Stratified split: keep n_samples, discard the rest
    sampled_indices, _ = train_test_split(
        indices,
        test_size=len(features) - n_samples,
        stratify=labels,
        random_state=seed
    )
    
    sampled_features = [features[i] for i in sorted(sampled_indices)]
    return sampled_features


def compute_entropy_on_split(model, dataset, device='cuda'):
    """
    Compute layer-wise entropy on a dataset.
    Returns dict: {layer_idx: {'maxEntropy': value}}
    """
    model.eval()
    model.to(device)
    
    layer_entropies_all = {}
    
    with torch.no_grad():
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=len(dataset),  # Load all at once (you said no batch processing needed)
            collate_fn=data_collator,
        )
        
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Remove labels so that layer_entropies are computed in model.forward()
            batch.pop('labels', None)
            
            # Forward pass with labels=None to compute entropy
            _ = model(**batch)
            
            # Extract layer entropies (computed during forward pass)
            if hasattr(model, 'layer_entropies'):
                for layer_idx, ent_dict in enumerate(model.layer_entropies):
                    if ent_dict is not None:
                        layer_entropies_all[layer_idx] = ent_dict
    
    return layer_entropies_all


def print_entropy_results(split_name, layer_entropies):
    """Print entropy results to console."""
    print(f"\n{'='*60}")
    print(f"Layer Entropy Results for {split_name.upper()}")
    print(f"{'='*60}")
    
    for layer_idx in sorted(layer_entropies.keys()):
        ent_value = layer_entropies[layer_idx]['maxEntropy']
        print(f"Layer {layer_idx:2d}: maxEntropy = {ent_value:.6f}")
    
    print(f"{'='*60}\n")


def plot_entropy_comparison(results_dict, output_path):
    """
    Plot entropy across layers for all splits.
    results_dict: {split_name: {layer_idx: {'maxEntropy': value}}}
    """
    plt.figure(figsize=(12, 6))
    
    for split_name, layer_entropies in results_dict.items():
        layer_indices = sorted(layer_entropies.keys())
        entropy_values = [layer_entropies[idx]['maxEntropy'] for idx in layer_indices]
        
        plt.plot(layer_indices, entropy_values, marker='o', label=split_name, linewidth=2)
    
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Max Entropy (normalized)', fontsize=12)
    plt.title('Dataset-Level Entity Entropy Across Layers', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compute layer-wise entropy for REModel')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data/retacred',
                        help='Path to data directory')
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'cf_test', 'all'],
                        help='Which split(s) to evaluate on')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of samples to use (stratified by relation type)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for stratified sampling')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--input_format', type=str, default='typed_entity_marker_punct',
                        help='Input format for data processor')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Max sequence length')
    
    args = parser.parse_args()

    # Use parent directory of checkpoint as output_dir if not specified
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.checkpoint_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model from: {args.checkpoint_path}")
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint_path)
    
    # Determine encoder type
    encoder_type = getattr(config, 'encoder_type', 'luke')
    
    # Create processor with dummy args for data loading
    class DummyArgs:
        def __init__(self, args, encoder_type):
            self.input_format = args.input_format
            self.max_seq_length = args.max_seq_length
            self.encoder_type = encoder_type
            self.data_dir = args.data_dir
    
    dummy_args = DummyArgs(args, encoder_type=encoder_type)
    processor = TACREDProcessor(dummy_args, tokenizer)
    
    # Determine which splits to evaluate
    splits = ['test', 'cf_test'] if args.split == 'all' else [args.split]
    
    results = {}
    
    for split_name in splits:
        print(f"\nProcessing {split_name} split...")
        
        # Load dataset
        features = load_dataset_split(args.data_dir, split_name, dummy_args, tokenizer, processor)
        print(f"  Total samples in {split_name}: {len(features)}")
        
        # Stratified sampling
        sampled_features = stratified_sample(features, args.n_samples, seed=args.seed)
        print(f"  Using {len(sampled_features)} stratified samples")
        
        # Create dataset
        dataset = REDataset(sampled_features)
        
        # Compute entropy
        print(f"  Computing entropy across layers...")
        layer_entropies = compute_entropy_on_split(model, dataset, device=args.device)
        
        results[split_name] = layer_entropies
        
        # Print results
        print_entropy_results(split_name, layer_entropies)
    
    # Save results to JSON
    results_path = os.path.join(args.output_dir, f'entropy_results-{args.n_samples}.json')
    with open(results_path, 'w') as f:
        # Convert to serializable format
        results_serializable = {
            split: {str(k): v for k, v in layer_dict.items()}
            for split, layer_dict in results.items()
        }
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Plot comparison
    if len(results) > 1:
        plot_path = os.path.join(args.output_dir, f'entropy_comparison-{args.n_samples}.png')
        plot_entropy_comparison(results, plot_path)
    else:
        plot_path = os.path.join(args.output_dir, f'entropy_{splits[0]}.png')
        plot_entropy_comparison(results, plot_path)
    
    print(f"\nDone! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

    # # Single split
    # python compute_entropy.py --checkpoint_path outputs/lb_retacred/-2/checkpoint-3500 --split test --n_samples 1000

    # # All splits
    # python src/compute_entropy.py --data_dir ./data/retacred --checkpoint_path outputs/lb_retacred/original/checkpoint-4000 --split all --n_samples 200

    # # Custom seed/output
    # python compute_entropy.py --data_dir ./data/retacred  --checkpoint_path outputs/lb_retacred/-2/checkpoint-3500 --split cf_test --n_samples 100 --seed 64 --output_dir ./my_results