import torch
from torch.utils.data import DataLoader
# Ensure this imports your modified metric_calling.py
from utils.metrics.metric_calling import EvaluationMetricSpecifications, calculate_and_save_layerwise_metrics
# Ensure this imports your model.py containing REModelForMetrics
from model import REModelForMetrics 
from types import SimpleNamespace
from torch.utils.data import DataLoader
from data_processor import TACREDProcessor
from train import REDataset, data_collator
from transformers import AutoConfig
from transformers import AutoTokenizer


config = AutoConfig.from_pretrained("outputs/rb_tacred/0/checkpoint-1")
config.encoder_type = "roberta" # or "luke" depending on your model


tokenizer = AutoTokenizer.from_pretrained("outputs/rb_tacred/0/checkpoint-1")
model = REModelForMetrics.from_pretrained(
    "outputs/rb_tacred/0/checkpoint-1",
    config=config,
    use_safetensors=True,
    attn_implementation="eager"
)


# model.to('cuda')
model.eval()

# =========================================================
# 2. Setup Configuration
# =========================================================
# Dummy object required by the metric library for logging/saving context
class ModelSpecs:
    model_name = "REModel_Entity_Analysis"
model_specs = ModelSpecs()

# Placeholder for dataloader arguments (can be empty if not saving results to disk)
loader_kwargs = {} 

# Assume 'test_dataloader' is your existing dataloader
# It MUST yield batches with keys: {'input_ids', 'attention_mask', 'entity_mask', ...}


# 1. Mock the arguments required by the Processor
#    (Replace 'config' and 'tokenizer' with your loaded objects)
args = SimpleNamespace(
    data_dir="./data/tacred",
    input_format="typed_entity_marker_punct",
    max_seq_length=512,
    encoder_type=getattr(config, 'encoder_type', 'luke') 
)

# 2. Load the features from the file (e.g., test.json)
processor = TACREDProcessor(args, tokenizer)
features = processor.read(f"{args.data_dir}/test.json")

# 3. (Optional) Slice the list if you want fewer samples
# features = features[:1000] 

# 4. Create the DataLoader
dataset = REDataset(features)
dataloader = DataLoader(
    dataset, 
    batch_size=len(dataset), # Or set a specific batch size like 32
    collate_fn=data_collator,
    shuffle=False
)

# =========================================================
# 3. Compute Metrics (Run Separately as needed)
# =========================================================

# --- Metric A: Entity Prompt Entropy ---
# What it measures: "How diverse are the entity tokens within a single sentence?"
# High score = The model sees the entity as a complex object with many distinct features.
# Low score = The model has compressed the entity into a very simple representation.
print("\n--- Computing Entity Prompt Entropy ---")
prompt_entropy_specs = EvaluationMetricSpecifications(
    evaluation_metric='prompt-entropy', 
    num_samples=1000, # Adjust based on your test set size
    alpha=1.0,        # Shannon Entropy
    normalizations=['maxEntropy'] # Normalized by log(N)
)

pe_results = calculate_and_save_layerwise_metrics(
    model=model,
    dataloader=dataloader, 
    model_specs=model_specs,
    evaluation_metric_specs=prompt_entropy_specs,
    dataloader_kwargs=loader_kwargs,
    should_save_results=False
)
print("Layer-wise Prompt Entropy:", pe_results['maxEntropy'])


# --- Metric B: Entity Dataset Entropy ---
# What it measures: "How distinguishable are different entities from each other across the dataset?"
# High score = The model can easily tell Entity A apart from Entity B.
# Low score = All entities look the same (Collapse).
print("\n--- Computing Entity Dataset Entropy ---")
dataset_entropy_specs = EvaluationMetricSpecifications(
    evaluation_metric='dataset-entropy', 
    num_samples=1000,
    alpha=1.0, 
    normalizations=['maxEntropy']
)

de_results = calculate_and_save_layerwise_metrics(
    model=model,
    dataloader=dataloader,
    model_specs=model_specs,
    evaluation_metric_specs=dataset_entropy_specs,
    dataloader_kwargs=loader_kwargs,
    should_save_results=False
)
print("Layer-wise Dataset Entropy:", de_results['maxEntropy'])


# --- Metric C: Entity Curvature ---
# What it measures: "How jagged is the path through the network?"
# This analyzes the sequence of entity tokens. 
# Lower curvature usually implies the model has linearized the representation (easier for the classifier).
print("\n--- Computing Entity Curvature ---")
curvature_specs = EvaluationMetricSpecifications(
    evaluation_metric='curvature',
    num_samples=1000,
    curvature_k=1 
)

curv_results = calculate_and_save_layerwise_metrics(
    model=model,
    dataloader=dataloader,
    model_specs=model_specs,
    evaluation_metric_specs=curvature_specs,
    dataloader_kwargs=loader_kwargs,
    should_save_results=False
)
print("Layer-wise Curvature:", curv_results['raw'])