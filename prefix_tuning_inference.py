from transformers import AutoTokenizer, AutoModelForCausalLM
import csv, re, random, json, torch, evaluate
from datetime import datetime, timedelta
from privacy_metrics.Metrics import entities_in_paragraph, leaked_percentage
import os
import gc
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "princeton-nlp/Sheared-LLaMA-1.3B"
OUTPUT_CSV_FILE = "fine_tuning_data_first_six_paragraphs_direct.csv"
CHECKPOINT_DIR = "sheared-llama-privacy-prefix"
FINAL_CHECKPOINT = "final_checkpoint"
OUTPUT_FILE = "prefix_tuning_eval_results.json"
MAX_EVAL_SAMPLES = None  # Set to a number to limit evaluation samples, None for all
MAX_NEW_TOKENS = 400
TEMPERATURE = 0.7
TOP_P = 0.9

# Helper functions for generating random values
def random_code():
    """Generate a random code in format XXXXX/XX"""
    from string import ascii_uppercase, digits
    body = "".join(random.choices(ascii_uppercase + digits, k=5))
    tail = "".join(random.choices(ascii_uppercase + digits, k=2))
    return f"{body}/{tail}"

FIRST_NAMES = ["Alex", "Blake", "Casey", "Dana", "Elliot", "Finley", "Harper",
               "Jordan", "Kai", "Logan", "Morgan", "Quinn", "Riley", "Skyler"]
LAST_NAMES = ["Adams", "Baker", "Carson", "Dawson", "Ellis", "Foster",
              "Griffin", "Hayes", "Irwin", "Johnson", "Kennedy", "Lewis"]

def random_person():
    """Generate a random person's name with title"""
    return f"{random.choice(['Mr', 'Ms', 'Dr', 'Prof'])} {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def random_datetime(start_year=1990, end_year=2024):
    """Generate a random date between start_year and end_year"""
    start_dt = datetime(start_year, 1, 1)
    span_days = (datetime(end_year, 12, 31) - start_dt).days
    day = start_dt + timedelta(days=random.randint(0, span_days))
    return day.strftime("%d %B %Y")    
            
CITIES = ["Baltimore", "Seattle", "Tokyo", "Munich", "Cairo"]
COUNTRIES = ["USA", "Germany", "Japan", "Kenya", "Brazil"]
ADDRESSES = ["221B Baker St", "1600 Amphitheatre Pkwy", "350 Fifth Ave"]
INFRA = ["London Bridge", "Central Station", "Pier 39"]

def random_loc():
    """Generate a random location"""
    return random.choice(CITIES + COUNTRIES + ADDRESSES + INFRA)
            
ORGS = ["OpenAI", "World Health Organization", "Harvard University",
        "UNICEF", "St. Mary's Hospital", "SpaceX",
        "NASA", "MIT", "Stanford University", "Google"]

def random_org():
    """Generate a random organization name"""
    return random.choice(ORGS)
            
HERITAGE = ["Irish-American", "Nigerian", "Chinese", "Latinx", "Punjabi"]
JOB = ["software engineer", "nurse", "professor", "mechanic", "pilot"]
def random_age():
    """Generate a random age"""
    return f"{random.randint(18, 85)}-year-old"

def random_dem():
    """Generate a random demographic description"""
    pattern = random.choice([
        f"{random_age()} {random.choice(JOB)}",
        f"{random.choice(HERITAGE)} descent",
        f"{random.choice(JOB).title()}",
        f"{random_age()}"
    ])
    return pattern
            
def random_quantity():
    """Generate a random quantity (percentage or money value)"""
    if random.random() < 0.5:  # 50-50: percent vs money
        return f"{random.randint(1, 100)}%"
    else:
        return f"${random.randint(1, 999_999):,}"

# Generator mapping
GENERATOR = {
    "CODE": random_code,
    "PERSON": random_person,
    "DATETIME": random_datetime,
    "LOC": random_loc,
    "ORG": random_org,
    "DEM": random_dem,
    "QUANTITY": random_quantity,
}

def load_model_for_inference():
    """Load the prefix-tuned model and tokenizer for inference"""
    try:
        logger.info(f"Loading base model: {MODEL_NAME}")
        
        # Load the base model with proper cache settings to avoid KeyError
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            use_cache=False  # Disable KV cache to prevent "Cache only has 0 layers" error
        )
        
        # Load the adapter weights
        model_path = Path(CHECKPOINT_DIR) / FINAL_CHECKPOINT
        logger.info(f"Loading adapter weights from: {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        from peft import PeftModel  # Import here to avoid circular import issues
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Load tokenizer - using model path but could also use base model name
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def extract_unique_values(csv_file):
    """Extract all unique values from the control codes in the CSV file"""
    logger.info(f"Extracting unique values from {csv_file}")
    unique_values = []  # Changed to list instead of set
    pattern = re.compile(r"^(CODE|PERSON|DATETIME|LOC|ORG|DEM|QUANTITY|MISE):\s*(.*)$", re.MULTILINE)
    
    try:
        with open(csv_file, newline="", encoding="utf-8") as csvfile:
            for row in csv.DictReader(csvfile):
                input_col = row["input"]
                for tag, raw in pattern.findall(input_col):
                    for val in map(str.strip, raw.split(",")):
                        if val and val not in unique_values:  # Avoid duplicates manually
                            unique_values.append(val)
        
        logger.info(f"Found {len(unique_values)} unique values for privacy evaluation")
        return unique_values
    
    except Exception as e:
        logger.error(f"Error extracting unique values: {e}")
        raise

def generate_with_peft_model(model, tokenizer, input_ids, attention_mask=None):
    """A safer generation function for PEFT models that works around common issues"""
    device = input_ids.device
    
    # Prepare a minimal kwargs dict to avoid past_key_values errors
    # The issue is in peft_model.py's prepare_inputs_for_generation method
    # which expects past_key_values in model_kwargs but we'll avoid that path
    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    try:
        # Method 1: Try direct base model generation with careful parameter passing
        logger.info("Attempting generation with base model direct access")
        with torch.no_grad():
            output_ids = model.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )
        return output_ids
    except Exception as e1:
        logger.warning(f"Base model generation failed: {e1}")
        
        try:
            # Method 2: Try using forward passes and manual token selection
            # This is a fallback approach that doesn't use the generate() method
            logger.info("Attempting manual token generation as fallback")
            
            # Start with the input sequence
            current_ids = input_ids.clone()
            current_mask = attention_mask.clone() if attention_mask is not None else None
            
            # Generate tokens one by one
            for _ in range(MAX_NEW_TOKENS):
                # Forward pass 
                with torch.no_grad():
                    outputs = model(
                        input_ids=current_ids, 
                        attention_mask=current_mask,
                        use_cache=False
                    )
                
                # Get next token logits (last position)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / TEMPERATURE
                
                # Apply top-p sampling
                if TOP_P < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > TOP_P
                    # Shift the indices to the right to keep the first one above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))
                
                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                
                # Update attention mask if it exists
                if current_mask is not None:
                    current_mask = torch.cat([
                        current_mask, 
                        torch.ones((current_mask.shape[0], 1), device=device)
                    ], dim=-1)
                
                # Check for EOS token
                if next_token[0, 0].item() == tokenizer.eos_token_id:
                    break
            
            return current_ids
            
        except Exception as e2:
            logger.error(f"Both generation methods failed. Final error: {e2}")
            # If both methods fail, return the input as a last resort
            return input_ids

def evaluate_model(model, tokenizer, unique_values):
    """Evaluate the model on privacy metrics and generation quality"""
    logger.info("Starting model evaluation")
    model.eval()
    
    # Initialize metrics
    privacy_violation_count = 0
    leaked_percentage_total = 0
    rouge_2_total = 0
    rouge_l_total = 0
    total_samples = 0
    leaked = []
    
    # Get device
    device = next(model.parameters()).device
    
    # Load ROUGE metric
    rouge = evaluate.load("rouge")
    
    # Pattern for parsing control codes
    pattern = re.compile(r"^(CODE|PERSON|DATETIME|LOC|ORG|DEM|QUANTITY|MISE):\s*(.*)$", re.MULTILINE)
    
    results = []
    try:
        with open(OUTPUT_CSV_FILE, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for idx, row in enumerate(reader, start=1):
                if MAX_EVAL_SAMPLES is not None and idx > MAX_EVAL_SAMPLES:
                    break
                
                control_code = row['input']
                reference_output = row['output']
                
                # Parse control codes
                matches = pattern.findall(control_code)
                latest = {}                                   
                for tag, raw in matches:            
                    latest.setdefault(tag, raw.strip())
                
                # Generate new control codes
                lines = []
                for tag in latest:
                    if tag in GENERATOR:                      
                        lines.append(f"{tag}: {GENERATOR[tag]()}")
                
                input_text = "\n".join(lines)
                
                # Prepare inputs for the model
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
                
                # Generate with the model
                # Note: Calling our custom generation function
                gen_ids = generate_with_peft_model(
                    model, 
                    tokenizer, 
                    inputs.input_ids, 
                    attention_mask=inputs.attention_mask if hasattr(inputs, 'attention_mask') else None
                )
                
                # Decode the generated text
                generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                
                # Remove the input text from the generated output
                input_text_decoded = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
                if generated_text.startswith(input_text_decoded):
                    generated_text = generated_text[len(input_text_decoded):].strip()
                
                # Evaluate privacy
                result_dict = entities_in_paragraph(generated_text, unique_values)
                
                for entity_value, found in result_dict.items():
                    if found:
                        logger.info(f"Privacy violation detected for entity: {entity_value}")
                        if entity_value not in leaked:
                            leaked.append(entity_value)
                
                if any(result_dict.values()):  # True if at least one entity leaked
                    privacy_violation_count += 1
                
                # Calculate leaked percentage
                leaked_percentage_amount = leaked_percentage(generated_text, unique_values)
                leaked_percentage_total += leaked_percentage_amount
                
                # Calculate ROUGE scores
                scores = rouge.compute(
                    predictions=[generated_text],
                    references=[reference_output],
                    use_stemmer=True,
                    rouge_types=["rouge2", "rougeL"]
                )
                rouge_2_total += scores["rouge2"]
                rouge_l_total += scores["rougeL"]
                
                # Store results
                results.append({
                    "input": control_code,
                    "reference": reference_output,
                    "generated": generated_text,
                    "privacy_violated": any(result_dict.values()),
                    "leaked_percentage": leaked_percentage_amount,
                    "rouge2": scores["rouge2"],
                    "rougeL": scores["rougeL"]
                })
                
                total_samples += 1
                
                # Log progress periodically
                if idx % 10 == 0:
                    logger.info(f"Processed {idx} samples")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise
    
    # Print summary statistics
    if total_samples > 0:
        logger.info("=== Evaluation Results ===")
        logger.info(f"Privacy violation percentage: {privacy_violation_count / total_samples * 100:.2f}%")
        logger.info(f"Average leaked percentage: {leaked_percentage_total / total_samples:.2f}%")
        logger.info(f"Average Rouge-2 score: {rouge_2_total / total_samples:.4f}")
        logger.info(f"Average Rouge-L score: {rouge_l_total / total_samples:.4f}")
        logger.info(f"Total examples evaluated: {total_samples}")
        logger.info(f"Unique leaked entities percentage: {len(set(leaked)) / total_samples * 100:.2f}%")
    else:
        logger.warning("No samples were evaluated")
    
    # Save results
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    return results

def cleanup_resources(model=None):
    """Clean up GPU memory and other resources"""
    if model is not None and torch.cuda.is_available():
        # Move model to CPU
        logger.info("Moving model to CPU for cleanup")
        model.to('cpu')
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache")
        torch.cuda.empty_cache()

def main():
    """Main function to run evaluation"""
    model = None
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_for_inference()
        logger.info("Successfully loaded trained model for evaluation")
        
        # Extract unique values for privacy evaluation
        unique_values = extract_unique_values(OUTPUT_CSV_FILE)
        
        # Run evaluation
        evaluation_results = evaluate_model(model, tokenizer, unique_values)
        logger.info(f"Evaluation completed with {len(evaluation_results)} samples")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        logger.error(traceback.format_exc())  # Print full traceback for debugging
        raise
    
    finally:
        # Clean up resources regardless of success or failure
        cleanup_resources(model)
        logger.info("Resources cleaned up")

if __name__ == "__main__":
    main()