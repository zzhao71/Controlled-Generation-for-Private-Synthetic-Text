from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import csv, re, string, random, json, torch, math, evaluate
from datetime import datetime, timedelta
from privacy_metrics.Metrics import entities_in_paragraph, leaked_percentage
from tqdm.auto import tqdm
# Specify the model name
model_name = "princeton-nlp/Sheared-LLaMA-1.3B"
output_csv_file = "fine_tuning_data_first_six_paragraphs_direct.csv"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt               = ""      
privacy_percentage   = 0
leaked_percentage_total = 0
total                = 0

generate_records = []
preds, refs = [], []
rouge_2 = 0
rouge_l = 0         



class StopOnCode(StoppingCriteria):
    def __init__(self, tokenizer, stop_string="CODE:"):
        super().__init__()
        self.stop_ids = tokenizer.encode(stop_string, add_special_tokens=False)

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        # abort if we don’t yet have enough tokens to compare
        if input_ids.shape[1] < len(self.stop_ids):
            return False

        # compare the last |stop_ids| tokens to the pattern
        tail = input_ids[0, -len(self.stop_ids):].tolist()
        return tail == self.stop_ids      # True ⇒ stop generation
    
    

with open(output_csv_file, newline='', encoding='utf-8') as f:
    n_records = sum(1 for _ in csv.DictReader(f))
batches = math.ceil(n_records / 3)

with open(output_csv_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
  
    pbar = tqdm(total = batches)
    for idx, row in enumerate(reader, start = 1):
        control_code = row['input']
        text_output = row['output']
        prompt += f"{control_code}\n{text_output}\n\n"

        #if idx == 7:
            #if total:
            
                #print(f"Privacy percentage so far: {privacy_percentage / total * 100:.2f}%")
                #print(f"Leaked percentage so far: {leaked_percentage_total / total:.2f}%")
            #break
            
        if (idx) % 3 == 0:
            pattern = re.compile(r"^(CODE|PERSON|DATETIME|LOC|ORG|DEM|QUANTITY|MISE):\s*(.*)$", re.MULTILINE)
            matches = pattern.findall(prompt)
            

            extracted_values = []
            for label, raw_value in matches:
                parts = [item.strip() for item in raw_value.split(',')]
                extracted_values.extend(parts)
                
            extracted_values = [v.strip() for v in extracted_values if v.strip()]
            refs.append(text_output)

            
            
            def random_code():
                body  = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
                tail  = "".join(random.choices(string.ascii_uppercase + string.digits, k=2))
                return f"{body}/{tail}"

            FIRST_NAMES = ["Alex", "Blake", "Casey", "Dana", "Elliot", "Finley", "Harper",
                            "Jordan", "Kai", "Logan", "Morgan", "Quinn", "Riley", "Skyler",]
            LAST_NAMES = ["Adams", "Baker", "Carson", "Dawson", "Ellis", "Foster",
                            "Griffin", "Hayes", "Irwin", "Johnson", "Kennedy", "Lewis",]

            def random_person():
                return f"{random.choice(['Mr', 'Ms', 'Dr', 'Prof'])} {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

            def random_datetime(start_year=1990, end_year=2024):
                start_dt = datetime(start_year, 1, 1)
                span_days = (datetime(end_year, 12, 31) - start_dt).days
                day = start_dt + timedelta(days=random.randint(0, span_days))
                return day.strftime("%d %B %Y")    
            
            CITIES = ["Baltimore", "Seattle", "Tokyo", "Munich", "Cairo"]
            COUNTRIES = ["USA", "Germany", "Japan", "Kenya", "Brazil"]
            ADDRESSES = ["221B Baker St", "1600 Amphitheatre Pkwy", "350 Fifth Ave"]
            INFRA = ["London Bridge", "Central Station", "Pier 39"]

            def random_loc():
                choice = random.choice([random.choice(CITIES), random.choice(COUNTRIES), random.choice(ADDRESSES), random.choice(INFRA)])
                return choice 
            
            ORGS = ["OpenAI", "World Health Organization", "Harvard University",
                    "UNICEF", "St. Mary’s Hospital", "SpaceX"
                    "NASA", "MIT", "Stanford University", "Google"]
            def random_org():
                return random.choice(ORGS)
            
            HERITAGE = ["Irish-American", "Nigerian", "Chinese", "Latinx", "Punjabi"]
            JOB      = ["software engineer", "nurse", "professor", "mechanic", "pilot"]
            AGE      = lambda: f"{random.randint(18, 85)}-year-old"

            def random_dem():
                pattern = random.choice([
                f"{AGE()} {random.choice(JOB)}",
                f"{random.choice(HERITAGE)} descent",
                f"{random.choice(JOB).title()}",
                f"{AGE()}"])
                return pattern
            
            def random_quantity():
                if random.random() < 0.5:                          # 50-50: percent vs money
                    return f"{random.randint(1, 100)}%"
                else:
                    return f"${random.randint(1, 999_999):,}"

            GENERATOR = {
                "CODE":     random_code,
                "PERSON":   random_person,
                "DATETIME": random_datetime,
                "LOC": random_loc,
                "ORG": random_org,
                "DEM": random_dem,
                "QUANTITY": random_quantity,
            }
            latest = {}                                   
            for tag, raw in (matches):            
                latest.setdefault(tag, raw.strip())

            
            lines = []
            for tag in latest:
                if tag in GENERATOR:                      
                    lines.append(f"{tag}: {GENERATOR[tag]()}")

            if lines:
                prompt += "\n" + "\n".join(lines)
                


            stopper = StoppingCriteriaList([StopOnCode(tokenizer)])

                             
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=400,
                stopping_criteria = stopper,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
            generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)[len(prompt):]
            generated_text = generated_text.split("CODE:")[0].rstrip()
            preds.append(generated_text)
            
            generate_records.append({
                "input": prompt,
                "generated_text": generated_text
            })
            
            
            result_dict = entities_in_paragraph(generated_text, extracted_values) 
            
            # Update privacy percentage
            #for entity_value, found in result_dict.items():
                #print(f"Entity '{entity_value}' found? {found}") #know which is found and which is not
                #if found:
                    #privacy_percentage += 1
            #total += 1
            
            if any(result_dict.values()):     # True if at least one entity leaked
                privacy_percentage += 1
            total += 1

            
            # Calculate leaked percentage for this batch
            leaked_percentage_amount = leaked_percentage(generated_text, extracted_values)
            leaked_percentage_total += leaked_percentage_amount
            pbar.update(1)
            pbar.set_postfix({
                "batch":   total,                       
                "privacy": f"{privacy_percentage/total*100:5.2f}%",
                "leaked":  f"{leaked_percentage_total/total:5.2f}%"
            })
            if preds:                           
                rouge = evaluate.load("rouge")

                scores = rouge.compute(
                    predictions = preds,
                    references  = refs,
                    use_stemmer = True,        
                    rouge_types = ["rouge2", "rougeL"]
                )
                rouge_2 += scores["rouge2"]
                rouge_l += scores["rougeL"]
            prompt = ""
            preds = []
            refs = []
            
            
            
pbar.close()


if total > 0:
    print(f"Final Privacy percentage: {privacy_percentage / total * 100:.2f}%")
    print(f"Final Leaked percentage: {leaked_percentage_total / total:.2f}%")   
    print(f"Final Rouge-2 score: {rouge_2 / total:.4f}")
    print(f"Final Rouge-L score: {rouge_l / total:.4f}")
    print(f"Total records processed: {total}")

out_path = "synthetic_ft_data_icl.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for rec in generate_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"Generated records have been saved to {out_path}.")
