import torch
from Utils.templates import *
def createDataset(datasetName, model, tokenizer, testingDataSize=200, trainingDataSize=20, modelName=''):
    if(datasetName == 'modified_arithmetic'):
        import json
        with open(f'bigbench/{datasetName}/combined.jsonl', 'r') as f:
            train = [json.loads(line) for line in f]
            
        message = getMessageTemplate(datasetName)
        
        prompts = []
        labels = []
        for i in range(len(train)):
            if(len(tokenizer.encode(train[i]['targets'][0], return_tensors='pt', add_special_tokens=False)[0]) != 1):
                # print(train[i]['targets'])
                continue
            message[-1]['content'] = train[i]['inputs']
            prompts.append({
                'message': message,
                'combined': tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True),
                'TEMPLATE': "{}",
                'prompt': train[i]['inputs']
            })
            labels.append({'complete': train[i]['targets'][0], 'short': train[i]['targets'][0]})
    
    elif(datasetName == 'arithmetic' or datasetName == 'elementary_math_qa'):
        import json
        with open(f'bigbench/{datasetName}/combined.jsonl', 'r') as f:
            train = [json.loads(line) for line in f]
            
        message = getMessageTemplate(datasetName)
        prompts = []
        labels = []
        for i in range(len(train)):
            if(len(tokenizer.encode(train[i]['targets'][0], return_tensors='pt', add_special_tokens=False)[0]) != 1):
                # print(train[i]['targets'])
                continue
            message[-1]['content'] = train[i]['inputs'][:-3]
            prompts.append({
                'message': message,
                'combined': tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True),
                'TEMPLATE': "{}",
                'prompt': train[i]['inputs'][:-3]
            })
            labels.append({'complete': train[i]['targets'][0], 'short': train[i]['targets'][0]})
           
    elif(datasetName == 'causal_judgment' or datasetName == 'entailed_polarity'):  
        import json
        with open(f'bigbench/{datasetName}/combined.jsonl', 'r') as f:
            train = [json.loads(line) for line in f]
        
        message = getMessageTemplate(datasetName)
        prompts = []
        labels = []
        for i in range(len(train)):
            
            message[-1]['content'] = train[i]['inputs'][:-3]
            prompts.append({
                'message': message,
                'combined': tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True),
                'TEMPLATE': "{}",
                'prompt': train[i]['inputs'][:-3]
            })
            if(train[i]['targets'][0].lower() == 'yes'):
                labels.append({'complete': 'Yes', 'short': 'Yes'})
            elif(train[i]['targets'][0].lower() == 'no'):
                labels.append({'complete': 'No', 'short': 'No'})
            else:
                raise ValueError("Invalid label")
            # labels.append({'complete': train[i]['targets'][0], 'short': train[i]['targets'][0]})
        
    elif(datasetName == 'object_counting'):
        import json
        with open(f'bigbench/{datasetName}/combined.jsonl', 'r') as f:
            train = [json.loads(line) for line in f]

        message = [
            {
                "role": "system",
                "content": "Count the number of objects in text."
            },
            {
                "role": "user",
                "content": "Q: I have a rabbit, a dog, a snake, and a pig. How many animals do I have?"
            },
            {
                "role": "assistant",
                "content": "four"
            },
            {
                "role": "user",
                "content": ""
            }
]
        prompts = []
        labels = []
        for i in range(len(train)):
            if(len(tokenizer.encode(train[i]['targets'][0], return_tensors='pt', add_special_tokens=False)[0]) != 1):
                # print(train[i]['targets'])
                continue
            message[-1]['content'] = train[i]['inputs'][:-3]
            prompts.append({
                'message': message,
                'combined': tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True),
                'TEMPLATE': "{}",
                'prompt': train[i]['inputs'][:-3]
            })
            labels.append({'complete': train[i]['targets'][0], 'short': train[i]['targets'][0]})
    elif(datasetName == 'color'):
        import json
        with open(f'bigbench/{datasetName}/combined.jsonl', 'r') as f:
            train = [json.loads(line) for line in f]
        message = getMessageTemplate(datasetName)
        prompts = []
        labels = []
        for i in range(len(train)):
            if(len(tokenizer.encode(train[i]['targets'][0], return_tensors='pt', add_special_tokens=False)[0]) != 1):
                # print(train[i]['targets'])
                continue
            message[-1]['content'] = train[i]['inputs'][:-3]
            prompts.append({
                'message': message,
                'combined': tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True),
                'TEMPLATE': "{}",
                'prompt': train[i]['inputs'][:-3]
            })
            labels.append({'complete': train[i]['targets'][0], 'short': train[i]['targets'][0]})
    elif(datasetName == 'navigate'):
        import json
        with open(f'bigbench/{datasetName}/combined.jsonl', 'r') as f:
            train = [json.loads(line) for line in f]
        
        message = getMessageTemplate(datasetName)
        prompts = []
        labels = []
        for i in range(len(train)):
            if(len(tokenizer.encode(train[i]['targets'][0], return_tensors='pt', add_special_tokens=False)[0]) != 1):
                # print(train[i]['targets'])
                continue
            message[-1]['content'] = train[i]['inputs'][:-3]
            prompts.append({
                'message': message,
                'combined': tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True),
                'TEMPLATE': "{}",
                'prompt': train[i]['inputs'][:-3]
            })
            labels.append({'complete': train[i]['targets'][0], 'short': train[i]['targets'][0]})
    elif(datasetName == 'winowhy'):
        import json
        with open(f'bigbench/{datasetName}/combined.jsonl', 'r') as f:
            train = [json.loads(line) for line in f]
            
        message = getMessageTemplate(datasetName)
        prompts = []
        labels = []
        for i in range(len(train)):
            if(len(tokenizer.encode(train[i]['targets'][0], return_tensors='pt', add_special_tokens=False)[0]) != 1):
                continue
            message[-1]['content'] = train[i]['inputs']
            prompts.append({
                'message': message,
                'combined': tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True),
                'TEMPLATE': "{}",
                'prompt': train[i]['inputs']
            })
            if(train[i]['targets'][0].lower() == 'correct'):
                if(modelName == 'mixtral'):
                    labels.append({'complete': 'Correct', 'short': 'Correct'})
                else:
                    labels.append({'complete': 'Correct', 'short': 'Correct'})
            elif(train[i]['targets'][0].lower() == 'incorrect'):
                if(modelName == 'mixtral'):
                    labels.append({'complete': 'Incorrect', 'short': 'Incorrect'})
                elif(modelName == 'phi-3'):
                    labels.append({'complete': 'Incorrect', 'short': 'In'})
                else:
                    labels.append({'complete': 'Incorrect', 'short': 'Incorrect'})
                
    elif(datasetName == 'operators'):
        import json
        with open(f'bigbench/{datasetName}/combined.jsonl', 'r') as f:
            train = [json.loads(line) for line in f]
            
        message = getMessageTemplate(datasetName)
        prompts = []
        labels = []
        for i in range(len(train)):
            if(len(tokenizer.encode(train[i]['targets'][0], return_tensors='pt', add_special_tokens=False)[0]) != 1):
                # print(train[i]['targets'])
                continue
            message[-1]['content'] = train[i]['inputs']
            prompts.append({
                'message': message,
                'combined': tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True),
                'TEMPLATE': "{}",
                'prompt': train[i]['inputs']
            })
            labels.append({'complete': train[i]['targets'][0], 'short': train[i]['targets'][0]})

    elif(datasetName == 'SNLI'):
        from datasets import load_dataset

        ds = load_dataset("stanfordnlp/snli")
        message = getMessageTemplate(datasetName)
        ds = ds['test']
        prompt_Template = "Premise: {premise}\nHypothesis:{hypothesis}\nQuestion: Does the premise entail, contradict, or is it neutral to the hypothesis?" 
        
        prompts = []
        labels = []
        for i in range(len(ds)):
            message[-1]['content'] = prompt_Template.format(premise=ds[i]['premise'], hypothesis=ds[i]['hypothesis'])
            prompts.append({
                'message': message,
                'combined': tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True),
                'TEMPLATE': prompt_Template,
                'prompt': prompt_Template.format(premise=ds[i]['premise'], hypothesis=ds[i]['hypothesis'])
            })
            # prompts.append(tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True))
            if(ds[i]['label'] == 0):
                # labels.append('entailment')
                # labels.append({'complete': 'Entailment', 'short': 'Ent'})
                labels.append({'complete': 'ent', 'short': 'ent'})
                # labels.append({'ent'})
            if(ds[i]['label'] == 1):
                # labels.append('neutral')
                # labels.append({'complete': 'Neutral', 'short': 'Ne'})
                labels.append({'complete': 'ne', 'short': 'ne'})
                # labels.append('ne')
            if(ds[i]['label'] == 2):
                # labels.append('contradiction')
                labels.append({'complete': 'con', 'short': 'con'})
                # labels.append({'complete': 'Contradiction', 'short': 'Con'})
                # labels.append('con')
                
            
    elif(datasetName == 'ag_news'):
        import json
        with open('BaysianOptimization/dataset/ag_news/test.json', 'r') as f:
            data = json.load(f)
        message = getMessageTemplate(datasetName)
        prompt_template = "News Article: {review}\nQuestion: Question: What category does this news article belong to?"
        
        prompts = []
        labels = []
        for i in range(len(data)):
            message[-1]['content'] = prompt_template.format(review=data[i]['text'])
            prompts.append({
                'message': message,
                'combined': tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True),
                'TEMPLATE': prompt_template,
                'prompt': prompt_template.format(review=data[i]['text'])
            })
            labels.append({'complete': data[i]['label'], 'short': data[i]['label']})
            # labels.append(data[i]['label'])
    
                
    fewShotPrompts, fewShotLabels, answer_tokens = getTrainingFewShotExamples(model, tokenizer, prompts, labels, number_of_examples=trainingDataSize)
    otherFewShotPrompts, otherFewShotLabels = [], []
    
    for i in range(len(prompts)):
        if(prompts[i] not in fewShotPrompts):
            otherFewShotPrompts.append(prompts[i])
            otherFewShotLabels.append(labels[i]['complete'])
            if(len(otherFewShotPrompts) == 20):
                break
    
    testPrompts, testLabels = [], []
    for i in range(len(prompts)):
        if(prompts[i] not in fewShotPrompts and prompts[i] not in otherFewShotPrompts):
            testPrompts.append(prompts[i])
            testLabels.append(labels[i])
            if(len(testPrompts) == testingDataSize):
                break
    
    return {'train_prompts': fewShotPrompts, 'train_labels': fewShotLabels,'train_answer_tokens':answer_tokens, 'otherFewShotPrompts': otherFewShotPrompts, 'otherFewShotLabels': otherFewShotLabels, 'test_prompts': testPrompts, 'test_labels': testLabels}
    
    
        
def getZeroShotAndFewShotPrompts(prompts, labels, fewShotExamples, fewShotLabels, datasetName, tokenizer):
    zeroShotPrompts = []
    fewShotPrompts = []
    
    message_template = getMessageTemplate(datasetName)
    count = 6
    for i in range(len(prompts)):
        message_template[-1]['content'] = prompts[i]['prompt']
        zeroShotPrompts.append(tokenizer.apply_chat_template(message_template, tokenize=False, add_generation_prompt=True))
    new_message = message_template.copy()[:-1]
    for i in range(count):
        prompt = fewShotExamples[i]
        new_message.append({'role': 'user', 'content': prompt['prompt']})
        new_message.append({'role': 'assistant', 'content': fewShotLabels[i]})
    new_message.append({'role': 'user', 'content': ''})
    for i in range(len(prompts)):
        new_message[-1]['content'] = prompts[i]['prompt']
        fewShotPrompts.append(tokenizer.apply_chat_template(new_message, tokenize=False, add_generation_prompt=True))
    # breakpoint()
    return zeroShotPrompts, fewShotPrompts, labels
    # breakpoint()
    # if(datasetName == 'causal_judgment'):
    #     count = 2
    # else:
    #     count = 4
    # print("length of few shot examples", count)
    # Template = """"""
    # for i in range(count):
    #     # breakpoint()
    #     if(type(fewShotExamples[i]) == dict):
    #         Template += f"{fewShotExamples[i]['prompt']}{fewShotLabels[i]}\n\n"
    #     else:
    #         Template += f"{fewShotExamples[i]}{fewShotLabels[i]}\n\n"
    # for i in range(len(prompts)):
    #     if(type(prompts[i]) == dict):
    #         zeroShotPrompts.append(prompts[i]['combined'])
    #     else:
    #         zeroShotPrompts.append(prompts[i])
        
    #     if(type(prompts[i]) == dict):
    #         fewShotPrompts.append(Template + prompts[i]['combined'])
    #     else:
    #         fewShotPrompts.append(Template + prompts[i])
    # breakpoint()
    return zeroShotPrompts, fewShotPrompts, labels
        

def runDefaultModel(model, tokenizer, prompt, device='cuda', store_cache=False):
    model.reset_hooks()
    if(type(prompt) == dict):
        prompt = prompt['combined']
    input_id = tokenizer.encode(prompt, return_tensors='pt').to(device)
    model.reset_hooks()
    output = model(input_id)
    predicted_token = torch.argmax(output[:, -1, :], dim=1)[0]
    # breakpoint()
    del input_id, output
    return str(tokenizer.decode(predicted_token, skip_special_tokens=True)), predicted_token.item()

def getOriginalAccuracy(model, tokenizer, prompts, labels):
    correct = 0
    gold_answers = []
    predictions = []
    from tqdm import tqdm
    pbar = tqdm(total=len(prompts))
    answer_tokens = []
    prob = 0
    count = 0
    for prompt, label in tqdm(zip(prompts, labels), desc="Original Accuracy", total=len(prompts)):
        output, answer_token = runDefaultModel(model, tokenizer, prompt)
            
        if output.lower().strip() == label['short'].lower().strip() or output.lower().strip() == label['complete'].lower().strip() or label['complete'].lower().strip().startswith(output.lower().strip()):
            correct += 1
            answer_tokens.append(answer_token)
            gold_answers.append(label['complete'].lower().strip())
            predictions.append(label['complete'].lower().strip())
            
        else:
            gold_answers.append(label['complete'].lower().strip())
            predictions.append(output.lower().strip())
            answer_tokens.append(tokenizer.encode(label['short'], return_tensors='pt', add_special_tokens=False)[0][0].item())
            
        count += 1
        # breakpoint()
        # gold_answers.append(label['short'].lower().strip())
        # predictions.append(output.lower().strip())
        pbar.set_description(f"Original Accuracy: {correct / count}")
        pbar.update(1)
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    print(classification_report(gold_answers, predictions))
    print(f"Accuracy: {correct / len(prompts)}")
    print(f"F1 Score: {f1_score(gold_answers, predictions, average='weighted')}")
    # breakpoint()
    return correct / len(prompts), answer_tokens, f1_score(gold_answers, predictions, average='weighted')



def getTrainingFewShotExamples(model, tokenizer, prompts, labels, number_of_examples=20):
    correct = 0
    correct_prompts = []
    correct_labels = []
    incorrect_prompts = []
    incorrect_labels = []
    correct_answer_tokens = []
    incorrect_answer_tokens = []
    from tqdm import tqdm
    pbar = tqdm(total=len(prompts))
    answer_tokens = []
    prob = 0
    count = 0
    for prompt, label in tqdm(zip(prompts, labels), desc="Original Accuracy", total=len(prompts)):
        # label = label['short']
        # if(type(prompt) == dict):
            # prompt = prompt['combined']
        output, answer_token = runDefaultModel(model, tokenizer, prompt)
        # breakpoint()
        output_tokens = tokenizer.encode(label['complete'], return_tensors='pt', add_special_tokens=False)
        try:
            if(type(output_tokens[0].item()) == int):
                output_tokens = output_tokens[0].item()
        except:
            output_tokens = output_tokens[0][0].item()
        # breakpoint()
        # if(type(output_tokens[0].item()) == int):
        #     output_tokens = output_tokens[0].item()
        # else:
        #     output_tokens = output_tokens[0][0].item()
        if output.lower().strip() == label['complete'].lower().strip() or output.lower().strip() == label['short'].lower().strip() or output_tokens == answer_token:
            correct += 1
            correct_prompts.append(prompt)
            correct_labels.append(label)
            correct_answer_tokens.append(answer_token)
        else:
            # breakpoint()
            incorrect_prompts.append(prompt)
            incorrect_labels.append(label)
            # breakpoint()
            # print(tokenizer.encode(label['short'], return_tensors='pt', add_special_tokens=False)[0].item())
            incorrect_answer_token = tokenizer.encode(label['short'], return_tensors='pt', add_special_tokens=False)
            try:
                if(type(incorrect_answer_token[0].item()) == int):
                    incorrect_answer_token = incorrect_answer_token[0].item()
            except:
                incorrect_answer_token = incorrect_answer_token[0][0].item()
            incorrect_answer_tokens.append(incorrect_answer_token)
            # if(type(tokenizer.encode(label['short'], return_tensors='pt', add_special_tokens=False)[0].item()) == int):
            #     incorrect_answer_tokens.append(tokenizer.encode(label['short'], return_tensors='pt', add_special_tokens=False)[0].item())
            # else:
            #     incorrect_answer_tokens.append(tokenizer.encode(label['short'], return_tensors='pt', add_special_tokens=False)[0][0].item())
            # incorrect_answer_tokens.append(tokenizer.encode(label['short'], return_tensors='pt', add_special_tokens=False)[0][0].item())
        count += 1
        pbar.set_description(f"Original Accuracy: {correct / count}")
        pbar.update(1)
        print(output, label, tokenizer.decode(answer_token, skip_special_tokens=True))
        if(len(correct_prompts) >= number_of_examples // 2 and len(incorrect_prompts) >= number_of_examples // 2):
            break
    prompts = correct_prompts[:number_of_examples // 2] + incorrect_prompts[:number_of_examples // 2]
    labels = correct_labels[:number_of_examples // 2] + incorrect_labels[:number_of_examples // 2]
    answer_tokens = correct_answer_tokens[:number_of_examples // 2] + incorrect_answer_tokens[:number_of_examples // 2]
    return prompts, labels, answer_tokens


def convert_to_float(frac_str):
    if('neg' in frac_str):
        frac_str = frac_str.replace('neg ', '-')
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        # breakpoint()
        return whole - frac if whole < 0 else whole + frac