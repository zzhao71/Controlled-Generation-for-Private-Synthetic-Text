"""
import json
import pandas as pd
from collections import defaultdict
import csv

with open('extracted_data.json', 'r') as file:
    data = json.load(file)

dataset = []

for annotation in data:
    control_code_dict = defaultdict(list)
    entity_mentions = annotation.get("entity_mentions", [])
    
    for entity_mention in entity_mentions:
        control_code = entity_mention.get("control_code")
        span_text = entity_mention.get("span_text")
        identifier = entity_mention.get("identifier_type")
        
        if control_code and span_text and (identifier in ["DIRECT", "QUASI"]):
            control_code_dict[control_code].append(span_text)
    

    control_code_list = [f"{control_code}: {', '.join(span_texts)}" for control_code, span_texts in control_code_dict.items()]
    text = annotation.get("text", "")
    control_code_list.append(f"text: {text}")
    

    dataset.append({"input": control_code_list, "output": text})


df = pd.DataFrame(dataset)
df.to_csv('fine_tuning_data.csv', index=False, quotechar='"', quoting=csv.QUOTE_ALL)

print("Dataset has been created and saved to 'fine_tuning_data_validation.csv'")
print(df.head())

"""

import json
import pandas as pd
from collections import defaultdict
import csv

# Load JSON data
with open('extracted_data.json', 'r') as file:
    data = json.load(file)

# Initialize dataset list
dataset = []

# Process each annotation
for annotation in data:
    control_code_dict = defaultdict(list)
    entity_mentions = annotation.get("entity_mentions", [])
    
    # Get the first six paragraphs of the text
    text = annotation.get("text", "")
    paragraphs = text.split('\n')
    first_paragraph = paragraphs[0] if len(paragraphs) > 0 else ""
    second_paragraph = paragraphs[1] if len(paragraphs) > 1 else ""
    third_paragraph = paragraphs[2] if len(paragraphs) > 2 else ""
    fourth_paragraph = paragraphs[3] if len(paragraphs) > 3 else ""
    fifth_paragraph = paragraphs[4] if len(paragraphs) > 4 else ""
    sixth_paragraph = paragraphs[5] if len(paragraphs) > 5 else ""
    
    # Combine first, second, third, fourth, fifth, and sixth paragraphs
    combined_paragraphs = first_paragraph
    if second_paragraph:
        combined_paragraphs += '\n' + second_paragraph
    if third_paragraph:
        combined_paragraphs += '\n' + third_paragraph
    if fourth_paragraph:
        combined_paragraphs += '\n' + fourth_paragraph
    if fifth_paragraph:
        combined_paragraphs += '\n' + fifth_paragraph
    if sixth_paragraph:
        combined_paragraphs += '\n' + sixth_paragraph
    
    # Collect control codes and span texts within the first six paragraphs
    for entity_mention in entity_mentions:
        control_code = entity_mention.get("control_code")
        span_text = entity_mention.get("span_text")
        identifier = entity_mention.get("identifier_type")
        
        #if control_code and span_text and (identifier in ["DIRECT", "QUASI"]) and (span_text in combined_paragraphs):
        if control_code and span_text and (identifier in ["DIRECT"]) and (span_text in combined_paragraphs):
            if span_text not in control_code_dict[control_code]:
                control_code_dict[control_code].append(span_text)
    
    # Create list of strings for input without including the text
    control_code_list = [f"{control_code}: {', '.join(span_texts)}" for control_code, span_texts in control_code_dict.items()]
    
    # Append to dataset
    #dataset.append({"input": control_code_list, "output": combined_paragraphs})#
    
   
    control_code_string = "\n".join(control_code_list)
    dataset.append({
        "input": control_code_string,
        "output": combined_paragraphs
    })


# Create DataFrame and save to CSV
df = pd.DataFrame(dataset)
df.to_csv('fine_tuning_data_first_six_paragraphs_direct.csv', index=False, quotechar='"', quoting=csv.QUOTE_ALL)

print("Dataset has been created and saved to 'fine_tuning_data_first_six_paragraphs_direct.csv'")
print(df.head())


