

# %%
import openai

# Function to evaluate the fluency of a text using GPT-4
def rate_fluency(text_list):
    
    fluency_ratings = []
    
    for text in text_list:
        prompt = f"Please rate the fluency of the following text on a scale of 1 to 5, where 1 is least fluent and 5 is most fluent: \"{text}\". Provide only the number."
        
        messages = [
            {"role": "system", "content": "You are a text evaluation assistant."},
            {"role": "user", "content": prompt}
        ]
        rating = outputModel_Managed_LLM(GPT_4_O_API_ENPOINT, GPT_4_O_MODEL_NAME, messages)
        
        # Append the rating to the list
        fluency_ratings.append({text: rating})
    
    return fluency_ratings

# Example list of strings
texts = [
    "This is a simple sentence.",
    "I are not good at speaking English.",
    "The quick brown fox jumps over the lazy dog.",
    "Grammar mistake sentence writing bad."
]

# Call the function to rate fluency
# fluency_results = rate_fluency(texts)

# # Print the fluency ratings
# for result in fluency_results:
#     print(result)
import json
# %%
modelName = 'meta-llama/Meta-Llama-3-8B-Instruct'
task = 'toxicity'
scale = False
saveFilePath = f'{modelName.split("/")[1]}_{task}_scaling_{scale}_fluency_ratings.json'

# %%
from Utils.utils import *
layers, fileName = getInfo(modelName, task, scale)
import os
if(os.path.exists(f'{saveFilePath}') == False):
    import json
    with open(f'{saveFilePath}', 'w') as f:
        json.dump([], f)
    print(f'File created: {saveFilePath}')
with open(f'{saveFilePath}', 'r') as f:
    data = json.load(f)
print(f'File loaded: {saveFilePath}')
print(f'Previous data length: {len(data)}')
import pickle
with open(f'{fileName}', 'rb') as f:
    texts = pickle.load(f)
for i in range(len(data), len(texts)):
    print(f'{i+1}/{len(texts)}')
    text = texts[i]
    fluency_ratings = rate_fluency([text])
    # breakpoint()
    import time
    time.sleep(1)
    data.append({text: fluency_ratings[0]})
    with open(f'{saveFilePath}', 'w') as f:
        json.dump(data, f, indent=4)
    print(f'Fluency rating for text {i+1}: {fluency_ratings[0]}')
with open(f'{saveFilePath}', 'r') as f:
    data = json.load(f)
    
    
        


