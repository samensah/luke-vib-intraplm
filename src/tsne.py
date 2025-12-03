import torch
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm
import math


# Local imports
from model import REModelForMetrics
from train import REDataset, data_collator
from data_processor import TACREDProcessor

# ==========================================
# CONFIGURATION
# ==========================================
# List your checkpoints here, ordered by VIB layer application
# Example: [model_vib_0, model_vib_1, ..., model_vib_11]
CHECKPOINTS = [
    "outputs/rb_tacred/0/checkpoint-1",
    "outputs/rb_tacred/0/checkpoint-1",
    "outputs/rb_tacred/0/checkpoint-1",
    # ... add all others ...
    "outputs/rb_tacred/0/checkpoint-1"
]

DATA_FILE = "data/tacred/test.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_SAMPLES = 32 

def create_single_entity_mask(entity_mask, start_index):
    """Isolates the span of 1s starting near start_index."""
    single_mask = torch.zeros_like(entity_mask)
    seq_len = entity_mask.size(0)
    curr = start_index.item()
    
    # Skip markers (0s)
    while curr < seq_len and entity_mask[curr] == 0:
        curr += 1
        if curr - start_index > 10: break
    # Grab content (1s)
    while curr < seq_len and entity_mask[curr] == 1:
        single_mask[curr] = 1
        curr += 1
    return single_mask

def extract_final_layer(model, dataloader, device):
    """
    Extracts the FINAL layer representation using the selected aggregation method.
    Returns: List of tuples (vector_numpy, label_id)
    """
    extracted_data = [] 
    model.eval()
    
    dataset_list = list(dataloader.dataset)
    
    # We iterate batch by batch
    for i in range(0, len(dataset_list), dataloader.batch_size):
        batch_items = dataset_list[i : i + dataloader.batch_size]
        batch_tensors = data_collator(batch_items)
        inputs = {k: v.to(device) for k, v in batch_tensors.items() if k != 'feature_indices'}
        
        # Get labels for plot color coding
        batch_labels = [item['labels'] for item in batch_items]
        with torch.no_grad():
            outputs = model(**inputs)
        # last hidden state vectors
        final_layer = outputs['hidden_states'][-1]
        
        # =========================================================
        # Separates masks -> Mean Pool Subj -> Mean Pool Obj -> Concat
        # =========================================================
        vec_list = []
        curr_bs = final_layer.size(0)
        
        for b in range(curr_bs):
            entity_mask = inputs['entity_mask'][b]
            
            # Split Masks
            subj_mask = create_single_entity_mask(entity_mask, inputs['ss'][b]).unsqueeze(0)
            obj_mask = create_single_entity_mask(entity_mask, inputs['os'][b]).unsqueeze(0)
            # Prepare final layer tensor (1, Seq, Dim)
            layer_unsqueezed = final_layer[b].unsqueeze(0)
            # Pool Separately
            v_s = model._get_pooled_hidden_states(layer_unsqueezed, subj_mask)
            v_o = model._get_pooled_hidden_states(layer_unsqueezed, obj_mask)
            # Concat
            v_cat = torch.cat((v_s, v_o), dim=-1) # (1, 2*Dim)
            vec_list.append(v_cat.squeeze(0))
        # Stack into a batch tensor
        batch_vectors = torch.stack(vec_list)


        # Store Data: Move to CPU/Numpy and zip with labels
        batch_vectors_np = batch_vectors.cpu().numpy()
        for i, label in enumerate(batch_labels):
            extracted_data.append((batch_vectors_np[i], label))

    return extracted_data

def plot_grid(all_models_data, id_to_label, output_path):
    """
    Plots a grid of PCA visualizations.
    """
    n_models = len(all_models_data)
    cols = 4
    rows = math.ceil(n_models / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()
    
    # Filter for top N relations to keep plots clean
    # We calculate top N based on the first model's data
    first_data = all_models_data[0]['data']
    all_labels = [id_to_label[x[1]] for x in first_data]
    from collections import Counter
    counts = Counter(all_labels)
    if 'no_relation' in counts: del counts['no_relation']
    top_relations = [k for k, _ in counts.most_common(6)]

    for idx, model_info in enumerate(all_models_data):
        ax = axes[idx]
        data = model_info['data']
        vib_layer = model_info['vib_layer']
        
        # Filter and Prepare PCA
        vecs = []
        labels = []
        for vec, label_id in data:
            label_name = id_to_label[label_id]
            if label_name in top_relations:
                vecs.append(vec)
                labels.append(label_name)
        
        if not vecs: continue
            
        X = np.array(vecs)
        pca = PCA(n_components=2)
        X_r = pca.fit_transform(X)
        
        # Scatter
        # Map labels to colors consistently
        unique_labels = sorted(list(set(labels)))
        sns.scatterplot(
            x=X_r[:, 0], y=X_r[:, 1], hue=labels, style=labels,
            palette="tab10", ax=ax, legend=False, s=60, alpha=0.7
        )
        ax.set_title(f"VIB Applied at Layer {vib_layer}")
        ax.set_xticks([])
        ax.set_yticks([])

    # Create a single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(top_relations), bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved comparison grid to {output_path}")


# ==========================================
# MAIN EXECUTION
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze VIB Impact across models")
    # Example usage: --checkpoints outputs/vib_0/ckpt outputs/vib_1/ckpt ...
    parser.add_argument('--checkpoints', nargs='+', required=True, 
                        help="List of checkpoint paths ordered by layer index")
    parser.add_argument('--data_file', type=str, default="data/tacred/test.json")
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--output_dir', type=str, default="analysis_results")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup Output Directory

    
    # 2. Setup Data Config (Once)
    # We assume all models use the same tokenizer base (e.g., RoBERTa/LUKE)
    # If strictly necessary to reload tokenizer per model, move this inside loop.
    print(f"Loading Tokenizer from {args.checkpoints[0]}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoints[0])
    
    proc_args = SimpleNamespace(
        data_dir=str(Path(args.data_file).parent),
        input_format="typed_entity_marker_punct",
        max_seq_length=512,
        encoder_type='luke' # Default, will likely be overridden by config
    )
    
    # 3. Load Data & Deterministic Shuffle
    print("Loading and preparing dataset...")
    processor = TACREDProcessor(proc_args, tokenizer)
    features = processor.read(args.data_file)
    
    random.seed(args.seed)
    random.shuffle(features)
    
    if args.num_samples:
        features = features[:args.num_samples]
    
    # Prepare Dataloader
    dataset = REDataset(features)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_collator)
    
    # Map IDs to Labels for plotting
    id_to_label = {v: k for k, v in processor.LABEL_TO_ID.items()}

    all_results = []
    print(f"\nStarting analysis for {len(args.checkpoints)} models...")
    for i, ckpt in enumerate(tqdm(args.checkpoints)):
        try:
            # Attempt to infer VIB layer from path (e.g., "outputs/rb_tacred/0/checkpoint-1")
            path_parts = Path(ckpt).parts
            vib_layer_idx = path_parts[-2]

            # Load Model
            config = AutoConfig.from_pretrained(ckpt)
            model = REModelForMetrics.from_pretrained(ckpt, config=config)
            model.to(args.device)
            model.eval()
            
            # Extract Vectors
            data = extract_final_layer(model, dataloader, args.device)
            
            all_results.append({
                'vib_layer': vib_layer_idx,
                'data': data,
                'path': ckpt
            })
            
            # Clear GPU memory to prevent OOM loop
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing checkpoint {ckpt}: {e}")

    # 5. Generate Visualizations & Metrics
    if all_results:
        # # Sort results by layer index just in case input list wasn't sorted
        # all_results.sort(key=lambda x: x['vib_layer'])
        
        # A. Plot Grid of Manifolds
        entity_viz_dir = Path(ckpt).parent.parent / "entity_viz" 
        entity_viz_dir.mkdir(parents=True, exist_ok=True)
        plot_grid_path = entity_viz_dir / "vib_grid_comparison"
        plot_grid(all_results, id_to_label, plot_grid_path)
        
        
        print(f"\nAnalysis complete. Results saved to: {entity_viz_dir.resolve()}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()

#     python tsne_v2.py \
#   --checkpoints "outputs/rb_tacred/0/checkpoint-1" \
#   --data_file data/tacred/test.json \
#   --method concat_pool \
#   --output_dir results/vib_analysis \
#   --device cpu