import argparse
import os
import json
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List
from transformers import (
    AutoConfig, 
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed as hf_set_seed
)
from torch.utils.data import Dataset
from data_processor import TACREDProcessor
from evaluation import get_f1
from model import REModel
from sklearn.model_selection import train_test_split
from utils.utils import collate_fn


# ============================================================================
# Custom Dataset Class
# ============================================================================
class REDataset(Dataset):
    """Dataset class for Relation Extraction compatible with HF Trainer"""
    
    def __init__(self, features: List[Dict]):
        self.features = features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


# ============================================================================
# Data Collator
# ============================================================================
def data_collator(batch):
    """Wrapper to convert collate_fn output to dict format"""
    input_ids, attention_mask, labels, ss, os, entity_mask = collate_fn(batch)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'ss': ss,
        'os': os,
        'entity_mask': entity_mask,
    }


# ============================================================================
# Custom Trainer with VIB Loss
# ============================================================================
class RETrainer(Trainer):
    """Custom trainer that handles VIB loss scaling"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation with VIB loss scaling
        
        Args:
            model: The model
            inputs: Dict of inputs
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in batch (added in newer Transformers)
        """
        outputs = model(**inputs)
        
        # Unpack outputs
        loss = outputs[0]  # Main classification loss
        vib_loss = outputs[1]  # VIB regularization loss

        if vib_loss is not None:
            # Dynamic scaling: scale VIB loss by ratio of main loss to VIB loss
            scale_num = loss.item() / (vib_loss.item() + 1e-8)
            total_loss = loss + scale_num * vib_loss
            
            # Log individual losses for monitoring
            metrics = {
                'train/ce_loss': round(loss.item(), 3),
                'train/vib_loss': round(vib_loss.item(), 3),
                # 'train/scale_factor': scale_num,
                # 'train/entity_entropy': model.entity_entropy,
            }

            # Add entropy for each layer
            if hasattr(model, 'layer_entropies'):
                for layer_idx, ent in enumerate(model.layer_entropies):
                    metrics[f'train/maxEntropy_layer_{layer_idx}'] = ent['maxEntropy']
        else:
            # No VIB loss, just use the main loss
            total_loss = loss
            metrics = {
                'train/ce_loss': round(loss.item(), 3),
                # 'train/entity_entropy': model.entity_entropy,
            }

        self.log(metrics)
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only,
            ignore_keys = None,):
            
            # Use the parent class method to get outputs
            # This calls your model's forward() method internally.
            # Outputs format: (loss, vib_loss, logits) if labels are present.
            outputs = super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            
            if prediction_loss_only:
                # If we only need loss, the super() call handles this correctly
                return outputs 
            
            # The super() call returns a 3-tuple structure: (loss, accumulated_logits, labels)
            # In our case, the "accumulated_logits" is currently a TUPLE: (vib_loss_tensor, actual_logits_tensor)
            
            loss, collected_outputs, labels = outputs

            # We need to extract ONLY the actual logits tensor from the collected_outputs tuple
            # Based on your forward pass: outputs = (loss, vib_loss) + (logits,)
            # The 'collected_outputs' tuple holds everything *after* the first item (loss).
            # So it's (vib_loss_tensor, actual_logits_tensor)

            # Extract the final logits tensor, which is the second element here:
            actual_logits = collected_outputs[1]

            # Return the corrected structure: (loss, actual_logits_tensor, labels)
            return (loss, actual_logits, labels)




# ============================================================================
# Metrics Computation
# ============================================================================
def compute_metrics(eval_pred):
    """Compute F1 score for relation extraction"""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1 = get_f1(labels, preds)
    
    return {
        'f1': f1 * 100,
        'precision': precision * 100,
        'recall': recall * 100,
    }


# ============================================================================
# Helper: Evaluate on a dataset and return metrics
# ============================================================================
def evaluate_dataset(trainer, dataset, tag):
    """Evaluate on a single dataset and return results"""
    print(f"\nEvaluating on {tag} set...")
    pred_output = trainer.predict(dataset)
    metrics = compute_metrics(pred_output)
    
    print(f"{tag.upper()} Results:")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    
    return metrics


# ============================================================================
# Main Training Function
# ============================================================================
def main():
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument("--data_dir", default="./data/tacred", type=str)
    parser.add_argument("--input_format", default="typed_entity_marker_punct", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--data_percentage", type=float, default=100.0)
    
    # Model arguments
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--num_class", type=int, required=True)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--k_size", type=int, default=3)
    
    # VIB-specific arguments
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--vib_layer_idx", type=int, default=None)
    parser.add_argument("--disable_vib", action="store_true", default=False,
                        help="Disable VIB completely - sets vib_layer_idx to None")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=float, default=10.0)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    
    # Evaluation arguments
    parser.add_argument("--eval_strategy", type=str, default="steps", 
                       choices=['no', 'steps', 'epoch'])
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--load_best_model_at_end", action="store_true", default=True)
    parser.add_argument("--metric_for_best_model", type=str, default="f1")
    parser.add_argument("--greater_is_better", action="store_true", default=True)
    
    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    
    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    
    # Misc
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    
    args = parser.parse_args()

    # Handle VIB disabling
    if args.disable_vib:
        args.vib_layer_idx=None
    
    # Auto-detect encoder type
    if 'luke' in args.model_name_or_path.lower():
        args.encoder_type = 'luke'
    else:
        args.encoder_type = 'roberta'
    
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Encoder type: {args.encoder_type}")
    print(f"Output directory: {args.output_dir}")
    print(f"VIB layer: {args.vib_layer_idx}")
    print(f"Beta: {args.beta}")
    print(f"Seed: {args.seed}")
    print(f"Data percentage: {args.data_percentage}%")
    print(f"{'='*60}\n")
    
    # Set seed
    if args.seed > 0:
        hf_set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========================================================================
    # Load tokenizer and config
    # ========================================================================
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    
    config.gradient_checkpointing = True
    config.num_class = args.num_class
    config.dropout_prob = args.dropout_prob
    config.k_size = args.k_size

    if args.vib_layer_idx is not None:
        config.beta = args.beta
        config.vib_layer_idx = args.vib_layer_idx
    else:
        config.beta = None
        config.vib_layer_idx = None

    config.encoder_type = args.encoder_type
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    
    # ========================================================================
    # Load model
    # ========================================================================
    if os.path.exists(args.resume_from_checkpoint):
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        model = REModel.from_pretrained(
            args.resume_from_checkpoint, 
            config=config,
            use_safetensors=True,
            attn_implementation="eager"
        )
    else:
        model = REModel.from_pretrained(
            args.model_name_or_path,
            config=config,
            use_safetensors=True,
            attn_implementation="eager"
        )
    
    # ========================================================================
    # Load and process data
    # ========================================================================
    print("\nLoading data...")
    train_file = os.path.join(args.data_dir, "train.json")
    dev_file = os.path.join(args.data_dir, "dev.json")
    test_file = os.path.join(args.data_dir, "test.json")
    cf_test_file = os.path.join(args.data_dir, "test_cf.json")
    
    # Use test as dev if dev doesn't exist
    if not os.path.exists(dev_file):
        dev_file = test_file
    
    processor = TACREDProcessor(args, tokenizer)
    train_features = processor.read(train_file)
    dev_features = processor.read(dev_file)
    test_features = processor.read(test_file)
    cf_test_features = processor.read(cf_test_file)
    
    # Apply data sampling if needed
    if args.data_percentage < 100.0:
        labels = [feature['labels'] for feature in train_features]
        indices = np.arange(len(train_features))
        test_size = args.data_percentage / 100.0
        
        _, sampled_indices, _, _ = train_test_split(
            indices, 
            labels, 
            test_size=test_size,
            stratify=labels,
            random_state=args.seed
        )
        train_features = [train_features[i] for i in sampled_indices]
        print(f"Using {args.data_percentage}% of training data: {len(train_features)} samples")
    else:
        print(f"Using 100% of training data: {len(train_features)} samples")
    
    # Resize token embeddings if new tokens were added
    if len(processor.new_tokens) > 0:
        if args.encoder_type == 'luke':
            model.luke.resize_token_embeddings(len(tokenizer))
        else:
            model.roberta.resize_token_embeddings(len(tokenizer))
        print(f"Resized token embeddings to {len(tokenizer)}")
    
    # Create datasets
    train_dataset = REDataset(train_features)
    eval_dataset = REDataset(dev_features)  # Only dev for training
    test_dataset = REDataset(test_features)
    cf_test_dataset = REDataset(cf_test_features)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples: {len(eval_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"CF Test samples: {len(cf_test_dataset)}\n")
    
    # ========================================================================
    # Setup training arguments
    # ========================================================================
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        
        # Training hyperparameters
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        adam_epsilon=args.adam_epsilon,
        
        # # Precision
        # fp16=args.fp16 and not torch.cuda.is_bf16_supported(),
        # bf16=args.bf16 or (torch.cuda.is_bf16_supported() and not args.fp16),
        
        # Evaluation and saving (based on dev set)
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        
        # Logging
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        
        # Misc
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,  # Important: keep custom columns
        save_safetensors=True,
        gradient_checkpointing=True,
    )
    
    # ========================================================================
    # Setup callbacks
    # ========================================================================
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        ))
    
    # ========================================================================
    # Initialize trainer
    # ========================================================================
    trainer = RETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Only dev set
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        processing_class=tokenizer,
    )
    
    # ========================================================================
    # Train!
    # ========================================================================
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    
    # ========================================================================
    # Final evaluation on ALL test sets (after training completes)
    # ========================================================================
    print("\n" + "="*60)
    print("Final Evaluation on All Test Sets")
    print("="*60 + "\n")
    
    # Evaluate on dev
    dev_metrics = evaluate_dataset(trainer, eval_dataset, "dev")
    
    # Evaluate on test
    test_metrics = evaluate_dataset(trainer, test_dataset, "test")
    
    # Evaluate on cf_test
    cf_test_metrics = evaluate_dataset(trainer, cf_test_dataset, "cf_test")
    
    # Save all results to file
    results_path = os.path.join(
        args.output_dir,
        f"final_results-{args.seed}-{args.vib_layer_idx}.txt"
    )
    
    with open(results_path, "w", encoding='utf-8') as fp:
        fp.write(f"Final Results\n")
        fp.write(f"{'='*60}\n")
        fp.write(f"Model: {args.model_name_or_path}\n")
        fp.write(f"Encoder type: {args.encoder_type}\n")

        if args.vib_layer_idx is not None:
            fp.write(f"VIB layer: {args.vib_layer_idx}\n")
            fp.write(f"Beta: {args.beta}\n")
        else:
            fp.write(f"VIB disabled\n")
            
        fp.write(f"Seed: {args.seed}\n")
        fp.write(f"{'='*60}\n\n")
        
        fp.write(f"DEV:\n")
        fp.write(f"  F1: {dev_metrics['f1']:.4f}\n")
        fp.write(f"  Precision: {dev_metrics['precision']:.4f}\n")
        fp.write(f"  Recall: {dev_metrics['recall']:.4f}\n\n")
        
        fp.write(f"TEST:\n")
        fp.write(f"  F1: {test_metrics['f1']:.4f}\n")
        fp.write(f"  Precision: {test_metrics['precision']:.4f}\n")
        fp.write(f"  Recall: {test_metrics['recall']:.4f}\n\n")
        
        fp.write(f"CF_TEST:\n")
        fp.write(f"  F1: {cf_test_metrics['f1']:.4f}\n")
        fp.write(f"  Precision: {cf_test_metrics['precision']:.4f}\n")
        fp.write(f"  Recall: {cf_test_metrics['recall']:.4f}\n\n")
    
    print(f"\nFinal results saved to: {results_path}")
    print(f"Model checkpoints saved to: {args.output_dir}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()