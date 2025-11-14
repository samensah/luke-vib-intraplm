"""
Compute layer-wise dataset entropy for multiple REModel checkpoints on test sets.
Plots all models' entropy in a 1x2 grid (test vs cf_test).
Usage: python compute_entropy_multi_model.py --data_dir ./data/retacred --n_samples 1000
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
from pathlib import Path

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
    
    # Handle case where n_samples >= len(features)
    if n_samples >= len(features):
        return features
    
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
            batch_size=len(dataset),  # Load all at once
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


def get_model_checkpoints(base_dir='outputs/lb_retacred'):
    """
    Automatically discover model checkpoints.
    Returns a dict mapping layer_index to checkpoint path.
    """
    checkpoints = {}
    base_path = Path(base_dir)
    
    # Define expected layer indices and their descriptions
    layer_info = {
        -3: "Original (without VIB)",
        -2: "VIB on Word Embeddings",
        -1: "VIB at last Transformer layer",
        0: "VIB at first Transformer layer",
        # 1: "VIB at layer 1",
        # 2: "VIB at layer 2",
        # 3: "VIB at layer 3",
        # 4: "VIB at layer 4",
        # 5: "VIB at layer 5",
        # 6: "VIB at layer 6",
        # 7: "VIB at layer 7",
        # 8: "VIB at layer 8",
        # 9: "VIB at layer 9",
        # 10: "VIB at layer 10",
    }
    
    for layer_idx in layer_info.keys():
        layer_dir = base_path / str(layer_idx)
        if layer_dir.exists():
            # Find checkpoint directories
            checkpoint_dirs = sorted([d for d in layer_dir.glob('checkpoint-*') if d.is_dir()])
            if checkpoint_dirs:
                # Use the last checkpoint (highest number)
                checkpoints[layer_idx] = str(checkpoint_dirs[-1])
                print(f"Found checkpoint for layer {layer_idx}: {checkpoints[layer_idx]}")
    
    return checkpoints, layer_info


def plot_entropy_grid_comparison(all_results, output_path, layer_info):
    """
    Plot entropy across layers for all models in a 1x2 grid.
    all_results: {model_name: {split_name: {layer_idx: {'maxEntropy': value}}}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color palette for different models
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_results)))
    
    splits = ['test', 'cf_test']
    split_titles = {'test': 'Test Split', 'cf_test': 'Counterfactual Test Split'}
    
    for ax_idx, split_name in enumerate(splits):
        ax = axes[ax_idx]
        
        for model_idx, (model_name, model_results) in enumerate(sorted(all_results.items())):
            if split_name in model_results:
                layer_entropies = model_results[split_name]
                layer_indices = sorted(layer_entropies.keys())
                entropy_values = [layer_entropies[idx]['maxEntropy'] for idx in layer_indices]
                
                # Create label with VIB info
                vib_layer = int(model_name.split('_')[-1])
                # label = f"Layer {vib_layer}: {layer_info.get(vib_layer, 'VIB at layer ' + str(vib_layer))}"
                label = f"{layer_info.get(vib_layer, 'VIB at layer ' + str(vib_layer))}"
                
                ax.plot(layer_indices, entropy_values, 
                       marker='o', 
                       label=label, 
                       linewidth=2,
                       color=colors[model_idx],
                       markersize=4)
        
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Max Entropy (normalized)', fontsize=12)
        ax.set_title(f'Entity Entropy - {split_titles[split_name]}', fontsize=14)
        ax.legend(fontsize=9, loc='best', ncol=1)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Dataset-Level Entity Entropy Across Layers and Models', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Grid plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compute layer-wise entropy for multiple REModel checkpoints')
    parser.add_argument('--base_dir', type=str, default='outputs/lb_retacred',
                        help='Base directory containing model checkpoints')
    parser.add_argument('--checkpoint_paths', type=str, nargs='+', default=None,
                        help='Explicit list of checkpoint paths (optional, auto-discovery if not provided)')
    parser.add_argument('--data_dir', type=str, default='./data/retacred',
                        help='Path to data directory')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of samples to use (stratified by relation type)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for stratified sampling')
    parser.add_argument('--output_dir', type=str, default='./entropy_analysis',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--input_format', type=str, default='typed_entity_marker_punct',
                        help='Input format for data processor')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Max sequence length')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir = './'+args.base_dir.split('/')[1]+'_entropy_analysis'
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get model checkpoints
    if args.checkpoint_paths:
        # Use explicitly provided paths
        checkpoints = {}
        layer_info = {}
        for i, path in enumerate(args.checkpoint_paths):
            # Try to extract layer index from path
            parts = path.split('/')
            for part in parts:
                if part.lstrip('-').isdigit():
                    layer_idx = int(part)
                    checkpoints[layer_idx] = path
                    layer_info[layer_idx] = f"Model {layer_idx}"
                    break
            else:
                # If no layer index found, use sequential numbering
                checkpoints[i] = path
                layer_info[i] = f"Model {i}"
    else:
        # Auto-discover checkpoints
        checkpoints, layer_info = get_model_checkpoints(args.base_dir)
    
    if not checkpoints:
        raise ValueError("No model checkpoints found!")
    
    print(f"\nFound {len(checkpoints)} model checkpoints")
    
    # Store results for all models
    all_results = {}
    splits = ['test', 'cf_test']
    
    # checkpoints['original'] = 'outputs/lb_retacred/original/checkpoint-4000'
    # Process each model
    for layer_idx, checkpoint_path in sorted(checkpoints.items()):
        # if layer_idx not in ['original', str(-2), str(-1), str(0)]:
        #     continue
        model_name = f"vib_layer_{layer_idx}"
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*60}")
        
        # Load model and tokenizer
        model, tokenizer, config = load_model_and_tokenizer(checkpoint_path)
        
        # Determine encoder type
        encoder_type = getattr(config, 'encoder_type', 'luke')
        
        # Create processor
        class DummyArgs:
            def __init__(self, args, encoder_type):
                self.input_format = args.input_format
                self.max_seq_length = args.max_seq_length
                self.encoder_type = encoder_type
                self.data_dir = args.data_dir
        
        dummy_args = DummyArgs(args, encoder_type=encoder_type)
        processor = TACREDProcessor(dummy_args, tokenizer)
        
        model_results = {}
        
        for split_name in splits:
            print(f"\n  Processing {split_name} split...")
            
            # Load dataset
            features = load_dataset_split(args.data_dir, split_name, dummy_args, tokenizer, processor)
            print(f"    Total samples in {split_name}: {len(features)}")
            
            # Stratified sampling
            sampled_features = stratified_sample(features, args.n_samples, seed=args.seed)
            print(f"    Using {len(sampled_features)} stratified samples")
            
            # Create dataset
            dataset = REDataset(sampled_features)
            
            # Compute entropy
            print(f"    Computing entropy across layers...")
            layer_entropies = compute_entropy_on_split(model, dataset, device=args.device)
            
            model_results[split_name] = layer_entropies
            
            # Print brief summary
            if layer_entropies:
                avg_entropy = np.mean([v['maxEntropy'] for v in layer_entropies.values()])
                print(f"    Average entropy: {avg_entropy:.6f}")
        
        all_results[model_name] = model_results
        
        # Save individual model results
        model_results_path = os.path.join(args.output_dir, f'entropy_{model_name}.json')
        with open(model_results_path, 'w') as f:
            results_serializable = {
                split: {str(k): v for k, v in layer_dict.items()}
                for split, layer_dict in model_results.items()
            }
            json.dump(results_serializable, f, indent=2)
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Save combined results
    combined_results_path = os.path.join(args.output_dir, f'entropy_all_models_{args.n_samples}samples.json')
    with open(combined_results_path, 'w') as f:
        all_results_serializable = {
            model_name: {
                split: {str(k): v for k, v in layer_dict.items()}
                for split, layer_dict in model_results.items()
            }
            for model_name, model_results in all_results.items()
        }
        json.dump(all_results_serializable, f, indent=2)
    print(f"\nCombined results saved to: {combined_results_path}")
    
    # Create 1x2 grid plot
    plot_path = os.path.join(args.output_dir, f'entropy_comparison_grid_{args.n_samples}samples.png')
    plot_entropy_grid_comparison(all_results, plot_path, layer_info)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
    # python src/compute_entropy_multi_model.py --data_dir ./data/retacred --n_samples 200