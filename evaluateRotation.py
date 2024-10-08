import numpy as np
from skopt import gp_minimize
from skopt.space import Real
import torch
import transformer_lens.utils as utils
from functools import partial
import torch
from botorch.models import SingleTaskGP

from botorch.optim import optimize_acqf
from botorch.acquisition import qLogExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
import pickle
from Utils.datasetUtil import *
from Utils.checkpointUtil import *
from Utils.intervention import *
from tqdm import tqdm
from Utils.utils import *
import argparse

parser = argparse.ArgumentParser(description='Train a model with RoPE')
parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3-8B', help='The model to use')
parser.add_argument('--layer', type=int, nargs=2, default=[0, 32], help='The layer range to rotate')
parser.add_argument('--dataset', type=str, default='modified_arithmetic', help='The dataset to use')
parser.add_argument('--moduleType', type=str, default='module', help='The type of module to rotate')
parser.add_argument('--metric', type=str, default='accuracy', help='The metric to optimize')
parser.add_argument('--fewShotCategory', type=str, default='mix', help='The few shot category', required=False, choices=['zero', 'mix', 'sixShot'])
parser.add_argument('--rotationMethod', type=str, default='rotary', help='The rotation method to use', choices=['rotary', 'givens'])
parser.add_argument('--testingDataSize', type=int, default=200, help='The size of the testing data')
parser.add_argument('--angle_0', type=str)
parser.add_argument('--angle_1', type=str)
arg = parser.parse_args()
# python3 evaluateRotation.py --model llama-3-8b --layer 0 32 --dataset modified_arithmetic
modelPath = arg.model.strip()
metric = arg.metric
testingDataSize = arg.testingDataSize
moduleType = arg.moduleType
fewShotCategory = arg.fewShotCategory
rotationMethod = arg.rotationMethod
print(f"Model: {modelPath}")
# print(f"Layer: {arg.layer}")
# print(f"Dataset: {arg.dataset}")
print(f"Module Type: {moduleType}")
print(f"Metric: {metric}")
MODEL_PATH = getModel(modelPath)
model, tokenizer = loadTransformerLensModel(MODEL_PATH)

test_prompt = "The quick brown fox jumped over the lazy dog"
print("Num tokens:", len(model.to_tokens(test_prompt)[0]))

hook_z_weight = 0
mlp_out_weight = 0
H = 0
def print_name_shape_hook_function(activation, hook):
    if("attn.hook_z" in hook.name):
        global hook_z_weight, H
        hook_z_weight = activation.shape[-1]
        H = activation.shape[2]
    if("hook_mlp_out" in hook.name):
        global mlp_out_weight
        mlp_out_weight = activation.shape[-1]
    print(hook.name, activation.shape)

not_in_late_block_filter = lambda name: name.startswith("blocks.0.") or not name.startswith("blocks")

model.run_with_hooks(
    test_prompt,
    return_type=None,
    fwd_hooks=[(not_in_late_block_filter, print_name_shape_hook_function)],
)

# breakpoint()
if(moduleType == 'reasoning'):
    N = getRotationMatrixDimension(rotationMethod, hook_z_weight)
if(moduleType == 'mlp'):
    N = getRotationMatrixDimension(rotationMethod, mlp_out_weight)
L = arg.layer
datasetName = arg.dataset
# H = 32
# N = 64
num_L = L[-1] - L[0]
angle_0 = convert_to_float(arg.angle_0)
angle_1 = convert_to_float(arg.angle_1) 
angles = [(angle_0) * torch.pi, (angle_1) * torch.pi]


dtype = torch.float64

# %%
device = 'cuda' #cuda
model = model.to(device)


fileName = f'{modelPath}/{datasetName}/NN_kernel_layer_{L[0]}_{L[1]}_angle_{angles[0]}_{angles[1]}_{moduleType}_{metric}_{fewShotCategory}_{rotationMethod}_v2'

few_shot_examples = f'fewShotPrompts/{fileName}.pkl'
few_shot_labels = f'fewShotLabels/{fileName}.pkl'
checkpoint_path = f'checkpoint/{fileName}.pkl'

test_prompts = f'TestPrompts/{fileName}.pkl'
test_labels = f'TestLabels/{fileName}.pkl'

with open(few_shot_examples, 'rb') as f:
    few_shot_examples = pickle.load(f)
with open(few_shot_labels, 'rb') as f:
    few_shot_labels = pickle.load(f)
with open(test_prompts, 'rb') as f:
    test_prompts = pickle.load(f)
with open(test_labels, 'rb') as f:
    test_labels = pickle.load(f)
# few_shot_examples = few_shot_examples[10:]
# few_shot_labels = few_shot_labels[10:]
test_prompts = test_prompts[:testingDataSize]
test_labels = test_labels[:testingDataSize]
zeroShotPrompts, fewShotsPrompts, labels = getZeroShotAndFewShotPrompts(test_prompts, test_labels, few_shot_examples, few_shot_labels, datasetName, tokenizer)
breakpoint()
zeroShotNormalAccuracy = 0
zeroShotNormalAccuracy, answer_tokens, zeroShotNormalF1Score = getOriginalAccuracy(model, tokenizer, zeroShotPrompts, labels)
print(f"Normal Accuracy (zeroShot): {zeroShotNormalAccuracy}")
print(f"Normal F1 Score (zeroShot): {zeroShotNormalF1Score}")

# if(datasetName == 'causal_judgment'):
# else:   
fewShotNormalAccuracy = 0
fewShotNormalAccuracy, answer_tokens, fewShotNormalF1Score = getOriginalAccuracy(model, tokenizer, fewShotsPrompts, labels)
print(f"Normal Accuracy (fewShot): {fewShotNormalAccuracy}")
print(f"Normal F1 Score (fewShot): {fewShotNormalF1Score}")

def objective(params, prompt_list):
    if(moduleType == 'reasoning'):
        D = params.view(num_L, H, N)
    if(moduleType == 'mlp'):
        D = params.view(num_L, N)
    accuracy = 0
    prediction = []
    gold_answer = []
    count = 0
    prob = 0
    logit_diff = 0
    matrix_rank = 0
    from tqdm import tqdm
    pbar = tqdm(total=len(prompt_list))
    for prompt, label in tqdm(zip(prompt_list, labels), desc="Rotated Accuracy", total=len(prompt_list)):
        predicted_output, answer_token_prob, diff, rank  = runRotatedModel(model, tokenizer, prompt, D, 12, L,H=H, moduleType=moduleType, rotationType=rotationMethod)
        # print(predicted_output, label)
        if predicted_output.lower().strip() == label['complete'].lower().strip() or predicted_output.lower().strip() == label['short'].lower().strip():
            accuracy += 1 
        count += 1
        prob += answer_token_prob
        # prob += answer_token_prob
        logit_diff += diff
        matrix_rank += rank
        prediction.append(predicted_output.lower().strip())
        gold_answer.append(label['short'].lower().strip())
        pbar.set_description(f"Rotated Accuracy: {accuracy / count}")
        pbar.update(1)

    avg_prob = prob / count
    logit_diff = logit_diff / count
    matrix_rank = matrix_rank / count
    from sklearn.metrics import f1_score
    print(f"Accuracy: {accuracy / count}")
    print(f"Answer token prob: {avg_prob}")
    print(f"Logit diff: {logit_diff}")
    print(f"Matrix rank: {matrix_rank}")
    
    # if(metric == 'accuracy'):
    return accuracy / count, f1_score(gold_answer, prediction, average='weighted')
    # print(f"Answer token prob: {avg_prob}")
train_X, train_Y, _, iteration = load_checkpoint(checkpoint_path)
print(iteration)

combined_array = [(x, y) for x, y in zip(train_X, train_Y)]
sorted_array = sorted(combined_array, key=lambda x: x[1], reverse=False)
rotated_zero_shots_accuracy = []
rotated_few_shots_accuracy = []
rotated_zero_shots_f1Score = []
rotated_few_shots_f1Score = []
for i in range(5):
    D = sorted_array[i][0]
    accuracy, f1Score = objective(D, fewShotsPrompts)
    print(f"Few Shot Rotated: Accuracy: {accuracy}, F1 Score: {f1Score}")
    rotated_few_shots_accuracy.append(accuracy)
    rotated_few_shots_f1Score.append(f1Score)
    accuracy, f1Score = objective(D, zeroShotPrompts)
    print(f"Zero Shot Rotated: Accuracy: {accuracy}, F1 Score: {f1Score}")
    rotated_zero_shots_accuracy.append(accuracy)
    rotated_zero_shots_f1Score.append(f1Score)
# %%
import os
import json
if(os.path.exists('result.json')):
    with open('result.json', 'r') as f:
        data = json.load(f)
else:
    data = []

data.append({
    'model': modelPath,
    'dataset': datasetName,
    'layer': L,
    'angles': angles,
    'moduleType': moduleType,
    'metric': metric,
    'fewShotCategory': fewShotCategory,
    'rotationMethod': rotationMethod,
    'testingDataSize': testingDataSize,
    'zero_shot_normal_accuracy': zeroShotNormalAccuracy,
    'few_shot_normal_accuracy': fewShotNormalAccuracy,
    'few_shot_rotated_accuracy': rotated_few_shots_accuracy,
    'zero_shot_rotated_accuracy': rotated_zero_shots_accuracy,
    'few_shot_normal_f1_score': fewShotNormalF1Score,
    'zero_shot_normal_f1_score': zeroShotNormalF1Score,
    'zero_shot_rotated_f1_score': rotated_zero_shots_f1Score,
    'few_shot_rotated_f1_score': rotated_few_shots_f1Score
})

with open('result.json', 'w') as f:
    json.dump(data, f, indent=4)