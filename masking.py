import argparse
import csv
import os
import re

import torch
import tqdm
from torch.nn import functional as F
from peft import get_peft_model, PrefixTuningConfig, TaskType
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default configuration values (override via CLI)
DEFAULT_MODEL_NAME = "princeton-nlp/Sheared-LLaMA-1.3B"
DEFAULT_TRAIN_CSV = "fine_tuning_data_first_six_paragraphs.csv"
DEFAULT_EVAL_CSV = "fine_tuning_data_first_six_paragraphs.csv"
DEFAULT_OUTPUT_DIR = "sheared-llama-privacy-prefix"

# Define privacy tags that should be detected
TAG_RE = re.compile(r"^(CODE|PERSON|DATETIME|LOC|ORG|DEM|QUANTITY|MISE):\s*(.*)$", re.MULTILINE)


def build_models(model_name: str):
    """Load tokenizer/base model and attach prefix tuning adapter."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Using {tokenizer.pad_token} as padding token")

    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        prefix_projection=True,
        inference_mode=False,
        token_dim=base_model.config.hidden_size,
        num_attention_heads=base_model.config.num_attention_heads,
        num_layers=base_model.config.num_hidden_layers,
    )

    model = get_peft_model(base_model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable}")

    return tokenizer, base_model, model



class WeightedPrivacyDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"Loading data from {file_path}")
        try:
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    input_text = row.get('input', '').strip()
                    output_text = row.get('output', '').strip()
                    
                    if input_text and output_text:
                        self.examples.append({
                            "input": input_text,
                            "output": output_text
                        })
        except Exception as e:
            print(f"Error loading CSV: {e}")
            # Create a minimal example to avoid errors
            self.examples = [{
                "input": "Sample input",
                "output": "Sample output"
            }]
            
        print(f"Loaded {len(self.examples)} examples")
    
    def find_private_spans(self, input_text):
        """Find private information in input text."""
        private_values = []
        for match in TAG_RE.finditer(input_text):
            tag_type = match.group(1)
            value = match.group(2).strip()
            private_values.append({
                "tag_type": tag_type,
                "value": value
            })
        return private_values
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        try:
            example = self.examples[idx]
            
            # Process input and output
            input_text = example["input"]
            output_text = example["output"]
            
            # Find private values in the input
            private_values = self.find_private_spans(input_text)
            
            # Create combined text with separator
            combined_text = input_text + "\n" + output_text
            
            # Tokenize with padding and truncation
            encoding = self.tokenizer(
                combined_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Get input length to identify output section
            input_encoding = self.tokenizer(
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            )
            input_length = len(input_encoding.input_ids[0])
            
            # Extract tensors
            input_ids = encoding.input_ids.squeeze()
            attention_mask = encoding.attention_mask.squeeze()
            
            # Create labels: -100 for input section
            labels = input_ids.clone()
            labels[:input_length] = -100
            
            # Create weights tensor (1 = public, 0 = private)
            weights = torch.zeros_like(input_ids, dtype=torch.float)
            weights[input_length:] = 1.0  # All output tokens start as public
            
            # For each private value, process both the full value and its individual terms
            if private_values:
                # Tokenize output section for faster comparison
                output_ids = input_ids[input_length:].tolist()
                
                for private_item in private_values:
                    value = private_item["value"].strip()
                    if not value:
                        continue
                    
                    # Process the full value
                    self._mark_matching_tokens(value, output_text, output_ids, weights, input_length)
                    
                    # Also process individual terms within the value
                    terms = value.split()
                    for term in terms:
                            self._mark_matching_tokens(term, output_text, output_ids, weights, input_length)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "weights": weights,
                "input_length": input_length
            }
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            # Return a minimal valid example as in the original code
            # [error handling unchanged]
        
    def _mark_matching_tokens(self, search_term, output_text, output_ids, weights, input_length):
        """Helper method to find and mark tokens matching a search term."""

            
        # Convert to lowercase for case-insensitive matching
        search_term_lower = search_term.lower()
        output_text_lower = output_text.lower()
        
        # Get the token representation of the search term
        term_tokens = self.tokenizer.encode(search_term, add_special_tokens=False)
        
        # If the term doesn't tokenize to anything meaningful, skip it
        if not term_tokens:
            return
        
        # First check if term exists in output text (fast string check)
        if search_term_lower in output_text_lower:
            # Slide through output looking for ALL matches (not just the first one)
            i = 0
            while i <= len(output_ids) - len(term_tokens):
                match = True
                for j in range(len(term_tokens)):
                    if i+j >= len(output_ids) or output_ids[i+j] != term_tokens[j]:
                        match = False
                        break
                
                if match:
                    # Found a match, mark these tokens as private (weight=0)
                    for j in range(len(term_tokens)):
                        position = input_length + i + j
                        if position < len(weights):
                            weights[position] = 0.0  # Mark as private
                    # Skip past this match before continuing search
                    i += len(term_tokens)
                else:
                    i += 1
        
        # Handle subword tokenization issues
        term_pieces = self.tokenizer.tokenize(search_term)
        if len(term_pieces) > 1:
            # Try to match the first few subword pieces
            first_pieces = term_pieces[:min(len(term_pieces), 2)]  # Use first 2 pieces as marker
            output_tokens = self.tokenizer.convert_ids_to_tokens(output_ids)
            
            i = 0
            while i <= len(output_tokens) - len(first_pieces):
                match = True
                for j in range(len(first_pieces)):
                    if i+j >= len(output_tokens) or output_tokens[i+j] != first_pieces[j]:
                        match = False
                        break
                
                if match:
                    # Found a potential match of subword pieces
                    # Mark these tokens and a few following ones as private
                    span_length = min(len(term_pieces), len(output_tokens) - i)
                    for j in range(span_length):
                        position = input_length + i + j
                        if position < len(weights):
                            weights[position] = 0.0  # Mark as private
                    # Skip past this match
                    i += span_length
                else:
                    i += 1
# Custom trainer with privacy-aware loss
class PrivacyPrefixTrainer:
    def __init__(self, model, base_model, train_dataset, eval_dataset, tokenizer, output_dir,
                 batch_size=2, gradient_accumulation_steps=4, learning_rate=5e-5,
                 num_epochs=3, lm_loss_ratio=1.0, contrastive_loss_ratio=4.0, kl_loss_ratio=1.6):
        self.model = model
        self.base_model = base_model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.lm_loss_ratio = lm_loss_ratio
        self.contrastive_loss_ratio = contrastive_loss_ratio
        self.kl_loss_ratio = kl_loss_ratio
        
        os.makedirs(output_dir, exist_ok=True)
    
    def token_weighted_loss(self, loss_type, inputs, targets, weights):
        """Apply loss only to tokens with non-zero weights."""
        if loss_type == 'cross_entropy':
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        elif loss_type == 'kl':
            loss_fct = torch.nn.KLDivLoss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        loss = loss_fct(inputs, targets)
        if loss_type == 'kl':
            loss = loss.sum(dim=-1)
        
        # Apply weights
        weighted_loss = loss * weights
        # Normalize by the sum of weights
        if weights.sum() > 0:
            weighted_loss = weighted_loss.sum() / weights.sum()
        else:
            weighted_loss = torch.tensor(0.0, device=loss.device)
        
        return weighted_loss
    
    def compute_loss(self, batch):
        """Compute custom loss for privacy-aware training."""
        # Unpack batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        weights = batch["weights"]
        input_length = batch["input_length"]
        
        device = input_ids.device
        
        # Standard forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        lm_loss = outputs.loss
        
        # Create loss dictionary for reporting
        losses = {"lm_loss": lm_loss.item()}
        total_loss = lm_loss * self.lm_loss_ratio
        
        # Only compute additional losses if we have the base model
        if self.base_model is not None:
            # Get logits from current model
            shift_logits = outputs.logits[..., :-1, :]
            shift_labels = input_ids[..., 1:].unsqueeze(-1)
            shift_weights = weights[..., 1:]
            
            # For contrastive loss
            if self.contrastive_loss_ratio > 0:
                # Get probabilities of correct labels
                shift_probs = F.softmax(shift_logits, dim=-1)
                correct_label_probs = torch.gather(
                    shift_probs, -1, shift_labels
                ).squeeze(-1)
                
                # Run base model to get reference probabilities
                with torch.no_grad():
                    base_outputs = self.base_model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask
                    )
                    base_shift_logits = base_outputs.logits[..., :-1, :]
                    base_shift_probs = F.softmax(base_shift_logits, dim=-1)
                    base_label_probs = torch.gather(
                        base_shift_probs, -1, shift_labels
                    ).squeeze(-1)
                
                # Stack probabilities for contrastive loss
                contrastive_probs = torch.stack((correct_label_probs, base_label_probs), dim=-1)
                contrastive_probs = F.normalize(contrastive_probs, p=1, dim=-1)
                contrastive_log_probs = torch.log(contrastive_probs + 1e-10)
                
                # For private tokens, we want to prefer our model's predictions
                private_weights = shift_weights == 0
                
                # Only apply contrastive loss to tokens in output section with private info
                private_contrastive_weights = torch.zeros_like(shift_weights)
                for i in range(len(input_length)):
                    if i < len(private_weights) and input_length[i] < private_weights[i].size(0):
                        # Only consider output section
                        private_contrastive_weights[i, input_length[i]:] = private_weights[i, input_length[i]:]
                
                # Target is always 0 (prefer first option - our model)
                contrastive_targets = torch.zeros_like(shift_labels.squeeze(-1)).to(device)
                
                # Calculate contrastive loss only for private tokens
                if private_contrastive_weights.sum() > 0:
                    contrastive_loss = F.nll_loss(
                        contrastive_log_probs.view(-1, 2),
                        contrastive_targets.view(-1),
                        reduction='none'
                    ).view_as(private_contrastive_weights)
                    
                    # Apply weights
                    contrastive_loss = (contrastive_loss * private_contrastive_weights).sum() / private_contrastive_weights.sum().clamp(min=1)
                    contrastive_loss = contrastive_loss * self.contrastive_loss_ratio
                    
                    # Add to total loss
                    total_loss = total_loss + contrastive_loss
                    losses["contrastive_loss"] = contrastive_loss.item()
            
          
                if self.kl_loss_ratio > 0:
                    # Apply KL divergence to public tokens
                    public_weights = shift_weights > 0
                    
                    # Only apply KL to tokens in output section with public info
                    public_kl_weights = torch.zeros_like(shift_weights)
                    for i in range(len(input_length)):
                        if i < len(public_weights) and input_length[i] < public_weights[i].size(0):
                            # Only consider output section
                            public_kl_weights[i, input_length[i]:] = public_weights[i, input_length[i]:]
                    
                    if public_kl_weights.sum() > 0:
                        # Apply temperature scaling to logits for numerical stability
                        temperature = 1.0
                        scaled_shift_logits = shift_logits / temperature
                        scaled_base_shift_logits = base_shift_logits / temperature
                        
                        # Get log probabilities from fine-tuned model
                        shift_log_probs = F.log_softmax(scaled_shift_logits, dim=-1)
                        
                        # Get REGULAR probabilities from base model (not log probabilities)
                        with torch.no_grad():
                            base_shift_probs = F.softmax(scaled_base_shift_logits, dim=-1)
                        
                        # Ensure numerical stability
                        base_shift_probs = base_shift_probs.clamp(min=1e-10)
                        
                        # KL divergence loss with correct inputs
                        kl_loss = F.kl_div(
                            shift_log_probs.view(-1, shift_log_probs.size(-1)),
                            base_shift_probs.view(-1, base_shift_probs.size(-1)),  # REGULAR probs here
                            reduction='batchmean'  # More stable reduction
                        )
                        
                        # Scale by ratio
                        kl_loss = kl_loss * self.kl_loss_ratio
                        
                        # Add to total loss
                        total_loss = total_loss + kl_loss
                        losses["kl_loss"] = kl_loss.item()
        
        # Record total loss
        losses["total_loss"] = total_loss.item()
        
        return total_loss, losses
    
    def train(self):
        """Train the model with privacy-aware loss."""
        print("Starting training...")
        
        # Set up data loaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Set up optimizer
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate
        )
        
        # Number of training steps
        num_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        
        # Set up learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1.0,
            end_factor=0.1,
            total_iters=num_steps
        )
        
        # Put model in training mode
        self.model.train()
        self.base_model.eval()
        
        # Track best eval loss
        best_eval_loss = float('inf')
        
        # Training loop
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Training
            self.model.train()
            train_losses = {}
            
            progress_bar = tqdm.tqdm(train_loader, desc=f"Training epoch {epoch+1}")
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(next(self.model.parameters()).device) for k, v in batch.items()}
                
                # Compute loss
                loss, losses = self.compute_loss(batch)
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update running losses
                for k, v in losses.items():
                    if k not in train_losses:
                        train_losses[k] = 0.0
                    train_losses[k] += v / self.gradient_accumulation_steps
                
                # Update parameters every gradient_accumulation_steps
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    # Update progress bar
                    progress_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_losses.items()])
                    progress_bar.set_postfix_str(progress_str)
                    
                    # Reset running losses
                    train_losses = {}
            
            # Evaluation
            self.model.eval()
            eval_losses = {}
            num_eval_steps = 0
            
            with torch.no_grad():
                for batch in tqdm.tqdm(eval_loader, desc="Evaluation"):
                    # Move batch to device
                    batch = {k: v.to(next(self.model.parameters()).device) for k, v in batch.items()}
                    
                    # Compute loss
                    _, losses = self.compute_loss(batch)
                    
                    # Update running losses
                    for k, v in losses.items():
                        if k not in eval_losses:
                            eval_losses[k] = 0.0
                        eval_losses[k] += v
                    
                    num_eval_steps += 1
            
            # Calculate average eval losses
            for k in eval_losses:
                eval_losses[k] /= num_eval_steps
            
            # Print eval results
            eval_str = ", ".join([f"{k}: {v:.4f}" for k, v in eval_losses.items()])
            print(f"Eval results: {eval_str}")
            
            # Save model if it's the best so far
            if eval_losses.get("total_loss", float('inf')) < best_eval_loss:
                best_eval_loss = eval_losses.get("total_loss", float('inf'))
                print(f"New best eval loss: {best_eval_loss:.4f}")
                
                # Save model
                checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch+1}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                self.model.save_pretrained(checkpoint_dir)
                self.tokenizer.save_pretrained(checkpoint_dir)
                
                # Also save as best model
                best_dir = os.path.join(self.output_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                self.model.save_pretrained(best_dir)
                self.tokenizer.save_pretrained(best_dir)
        
        # Save final model
        final_dir = os.path.join(self.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        
        print("Training complete!")
        
        return self.model

def run_masking(
    model_name: str = DEFAULT_MODEL_NAME,
    train_csv_file: str = DEFAULT_TRAIN_CSV,
    eval_csv_file: str = DEFAULT_EVAL_CSV,
    output_dir: str = DEFAULT_OUTPUT_DIR,
):
    """Train the masking model with the provided configuration."""
    tokenizer, base_model, model = build_models(model_name)

    train_dataset = WeightedPrivacyDataset(train_csv_file, tokenizer)
    eval_dataset = WeightedPrivacyDataset(eval_csv_file, tokenizer)

    trainer = PrivacyPrefixTrainer(
        model=model,
        base_model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        output_dir=output_dir,
        batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_epochs=3,
        lm_loss_ratio=1.0,
        contrastive_loss_ratio=4.0,
        kl_loss_ratio=1.6,
    )

    trainer.train()
    return trainer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the masking-based privacy model.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base model to load from Hugging Face")
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV, help="Training CSV file path")
    parser.add_argument("--eval-csv", default=DEFAULT_EVAL_CSV, help="Evaluation CSV file path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to write checkpoints")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_masking(
        model_name=args.model_name,
        train_csv_file=args.train_csv,
        eval_csv_file=args.eval_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
