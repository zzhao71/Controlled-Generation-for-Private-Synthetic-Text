from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import csv, torch
from peft import get_peft_model, PrefixTuningConfig, TaskType
from torch.utils.data import Dataset
import os

# Specify the model name
model_name = "princeton-nlp/Sheared-LLaMA-1.3B"
train_csv_file = "fine_tuning_data_first_six_paragraphs.csv"
eval_csv_file = "fine_tuning_data_first_six_paragraphs_test.csv"  
output_model_dir = "sheared-llama-prefix-tuned"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Configure prefix tuning
peft_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,  # Number of virtual tokens in the prefix
    prefix_projection=True,  # Use projection layer for prefix
    inference_mode=False,    # Training mode
)

# Apply prefix tuning configuration to the model
model = get_peft_model(base_model, peft_config)


# Create dataset class
class PrivacyDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Read and prepare the data
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                control_code = row['input']
                text_output = row['output']
                
                # Combine input and output as a single training example
                input_text = f"{control_code}"
                output_text = f"{text_output}"
                
                self.examples.append({
                    "input_text": input_text,
                    "output_text": output_text
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize input and output separately
        input_encoding = self.tokenizer(example["input_text"], 
                                        max_length=self.max_length//2,
                                        padding="max_length",
                                        truncation=True,
                                        return_tensors="pt")
        
        output_encoding = self.tokenizer(example["output_text"], 
                                         max_length=self.max_length//2, 
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
        
        # Combine input and output for training
        input_ids = torch.cat([input_encoding.input_ids.squeeze(), 
                               output_encoding.input_ids.squeeze()])[:self.max_length]
        attention_mask = torch.cat([input_encoding.attention_mask.squeeze(), 
                                    output_encoding.attention_mask.squeeze()])[:self.max_length]
        
        # Create labels: -100 for input (not computed in loss), actual ids for output
        labels = torch.cat([torch.full_like(input_encoding.input_ids.squeeze(), -100), 
                            output_encoding.input_ids.squeeze()])[:self.max_length]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Data collator
def data_collator(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Create datasets for training and evaluation
train_dataset = PrivacyDataset(train_csv_file, tokenizer)
eval_dataset = PrivacyDataset(eval_csv_file, tokenizer)


# Define training arguments
training_args = TrainingArguments(
    output_dir=output_model_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    learning_rate=5e-5,
    fp16=True,  # Use mixed precision training if GPU supports it
    # Disable load_best_model_at_end to avoid the error
    load_best_model_at_end=False,
    logging_dir='./logs',
    logging_steps=10,
    report_to="none",  # Disable reporting to wandb, etc.
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save a checkpoint after training
model_save_path = os.path.join(output_model_dir, "final_checkpoint")
os.makedirs(model_save_path, exist_ok=True)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")

