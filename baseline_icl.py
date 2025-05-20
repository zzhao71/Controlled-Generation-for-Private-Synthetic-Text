from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import csv, re, json, torch, math, evaluate
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
extract = ""


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
        text_output = row['output']
        control_code = row['input']
        prompt += f"{text_output}\n\n"
        extract += f"{control_code}\n{text_output}\n\n"

        if idx == 7:
            if total:
            
                print(f"Privacy percentage so far: {privacy_percentage / total * 100:.2f}%")
                print(f"Leaked percentage so far: {leaked_percentage_total / total:.2f}%")
            break
            
        if (idx) % 3 == 0:
            pattern = re.compile(r"^(CODE|PERSON|DATETIME|LOC|ORG|DEM|QUANTITY|MISE):\s*(.*)$", re.MULTILINE)
            matches = pattern.findall(extract)
            

            extracted_values = []
            for label, raw_value in matches:
                parts = [item.strip() for item in raw_value.split(',')]
                extracted_values.extend(parts)
                
            extracted_values = [v.strip() for v in extracted_values if v.strip()]
            refs.append(text_output)
            print(extracted_values)

            
            
                


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
            extract = ""
            preds = []
            refs = []
            
            
            
pbar.close()


if total > 0:
    print(f"Final Privacy percentage: {privacy_percentage / total * 100:.2f}%")
    print(f"Final Leaked percentage: {leaked_percentage_total / total:.2f}%")   
    print(f"Final Rouge-2 score: {rouge_2 / total:.4f}")
    print(f"Final Rouge-L score: {rouge_l / total:.4f}")
    print(f"Total records processed: {total}")

out_path = "synthetic_ft_data_icl_baseline.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for rec in generate_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"Generated records have been saved to {out_path}.")
