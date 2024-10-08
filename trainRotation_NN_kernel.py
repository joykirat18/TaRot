# %%
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
import torch
import transformer_lens.utils as utils
from functools import partial
import torch
from botorch.models import SingleTaskGP

from botorch.optim import optimize_acqf
from botorch.acquisition import qLogExpectedImprovement, LogExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.kernels import InfiniteWidthBNNKernel
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel


import pickle
from Utils.datasetUtil import *
from Utils.checkpointUtil import *
from Utils.intervention import *
from tqdm import tqdm
from Utils.utils import *
# %%

import argparse

parser = argparse.ArgumentParser(description='Train a model with RoPE')
parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3-8B', help='The model to use')
parser.add_argument('--layer', type=int, nargs=2, default=[0, 32], help='The layer range to rotate')
parser.add_argument('--dataset', type=str, default='modified_arithmetic', help='The dataset to use')
parser.add_argument('--moduleType', type=str, default='module', help='The type of module to rotate')
parser.add_argument('--metric', type=str, default='accuracy', help='The metric to optimize')
parser.add_argument('--testingDataSize', type=int, default=200, help='The size of the testing dataset', required=False)
parser.add_argument('--trainingDataSize', type=int, default=20, help='The size of the training dataset', required=False)
parser.add_argument('--fewShotCategory', type=str, default='mix', help='The few shot category', required=False, choices=['zero', 'mix', 'sixShot'])
parser.add_argument('--rotationMethod', type=str, default='rotary', help='The rotation method to use', choices=['rotary', 'givens'])
parser.add_argument('--angle_0', type=str)
parser.add_argument('--angle_1', type=str)
arg = parser.parse_args()



# python3 trainRotation.py --model llama-3-8b --layer 0 32 --dataset modified_arithmetic
modelPath = arg.model.strip()
moduleType = arg.moduleType
metric = arg.metric
fewShotCategory = arg.fewShotCategory
testingDataSize = arg.testingDataSize
trainingDataSize = arg.trainingDataSize
rotationMethod = arg.rotationMethod
MODEL_PATH = getModel(modelPath)

model, tokenizer = loadTransformerLensModel(MODEL_PATH)

test_prompt = "The quick brown fox jumped over the lazy dog"
print("Num tokens:", len(model.to_tokens(test_prompt)[0]))

hook_z_weight = 0
mlp_out_weight = 0
H = 0
def print_name_shape_hook_function(activation, hook):
    if("attn.hook_z" in hook.name):
        global hook_z_weight
        hook_z_weight = activation.shape[-1]
        global H
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
# breakpoint()
angles = [(angle_0) * torch.pi, (angle_1) * torch.pi]
# breakpoint()
# L = [0,32]
# H = 32
# N = 64
# num_L = L[-1] - L[0]
n_initial_points = 4  # Number of initial points
dtype = torch.float64
device = 'cuda' #cuda

# %%
# meta-llama/Llama-2-7b-hf

# %%
model = model.to(device)

# %%
import torch
import numpy as np

import torch

# breakpoint()
# %%


# %%
if(moduleType == 'reasoning'):
    bounds = torch.tensor([[angles[0]] * (num_L * H * N), [angles[1]] * (num_L * H * N)], dtype=dtype).to(device)
if(moduleType == 'mlp'):
    bounds = torch.tensor([[angles[0]] * (num_L * N), [angles[1]] * (num_L * N)], dtype=dtype).to(device)

initial_points = (torch.rand((n_initial_points, bounds.size(1)), device=device) * (bounds[1] - bounds[0]) + bounds[0]).to(bounds.device)

dataset = createDataset(datasetName, model, tokenizer, testingDataSize=testingDataSize, trainingDataSize=trainingDataSize, modelName=modelPath)

train_prompts = dataset['train_prompts']
train_labels = dataset['train_labels']
train_answer_tokens = dataset['train_answer_tokens']
test_prompts = dataset['test_prompts']
test_labels = dataset['test_labels']
fewShotPrompts = dataset['otherFewShotPrompts']
fewShotLabels = dataset['otherFewShotLabels']
if(len(fewShotPrompts) < 6):
    raise ValueError("Few shot examples are less than 6")
# breakpoint()
import os
os.makedirs(f'TrainPrompts/{modelPath}/{datasetName}', exist_ok=True)
os.makedirs(f'TrainLabels/{modelPath}/{datasetName}', exist_ok=True)
os.makedirs(f'TestPrompts/{modelPath}/{datasetName}', exist_ok=True)
os.makedirs(f'TestLabels/{modelPath}/{datasetName}', exist_ok=True)
os.makedirs(f'fewShotPrompts/{modelPath}/{datasetName}', exist_ok=True)
os.makedirs(f'fewShotLabels/{modelPath}/{datasetName}', exist_ok=True)
os.makedirs(f'answer_tokens/{modelPath}/{datasetName}', exist_ok=True)
os.makedirs(f'checkpoint/{modelPath}/{datasetName}', exist_ok=True)

fileName = f'{modelPath}/{datasetName}/NN_kernel_layer_{L[0]}_{L[1]}_angle_{angles[0]}_{angles[1]}_{moduleType}_{metric}_{fewShotCategory}_{rotationMethod}_v2'
    # break

# %%
# prompt = prompts[0]

# %%


# %%
from jaxtyping import Float, Int
# %%
with open(f'TrainPrompts/{fileName}.pkl', 'wb') as f:    
    pickle.dump(train_prompts, f)
with open(f'TrainLabels/{fileName}.pkl', 'wb') as f:
    pickle.dump(train_labels, f)
with open(f'answer_tokens/{fileName}.pkl', 'wb') as f:
    pickle.dump(train_answer_tokens, f)
with open(f'TestPrompts/{fileName}.pkl', 'wb') as f:
    pickle.dump(test_prompts, f)
with open(f'TestLabels/{fileName}.pkl', 'wb') as f:
    pickle.dump(test_labels, f)
with open(f'fewShotPrompts/{fileName}.pkl', 'wb') as f:
    pickle.dump(fewShotPrompts, f)
with open(f'fewShotLabels/{fileName}.pkl', 'wb') as f:
    pickle.dump(fewShotLabels, f)
# breakpoint()
# %%
import torch
import pickle


checkpoint_path = f'checkpoint/{fileName}.pkl'

def objective(params):
    print(params.shape)
    # breakpoint()
    if(moduleType == 'reasoning'):
        D = params.view(num_L, H, N)
    if(moduleType == 'mlp'):
        D = params.view(num_L, N)
    # D = params.view(num_L, H, N)  # Ensure the tensor is reshaped correctly
    accuracy = 0
    count = 0
    prob = 0
    logit_diff = 0
    matrix_rank = 0
    from tqdm import tqdm
    # breakpoint()
    
    fewShotExamples = [{'prompt': prompt, 'label': label} for prompt, label in zip(fewShotPrompts, fewShotLabels)]
    # random.shuffle(fewShotExamples)

    for prompt, label, answer_token in tqdm(zip(train_prompts, train_labels, train_answer_tokens), desc="Rotated Accuracy", total=len(train_prompts)):
        if(fewShotCategory == 'sixShot'):
            nShot = fewShotExamples[:6]
            # print("6 shot")
            message = prompt['message']
            message = message[:-1]
            for shot in nShot:
                message.append({'role': 'user', 'content': shot['prompt']['prompt']})
                message.append({'role': 'assistant', 'content': shot['label']})
            message.append({'role': 'user', 'content': prompt['prompt']})
            prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        if(fewShotCategory == 'mix'):
            import random
            if(datasetName == 'causal_judgment'):
                random_index = random.choice([0, 1,2])
            else:
                random_index = random.choice([0, 2, 4, 6])
            nShot = fewShotExamples[:random_index]
            message = prompt['message']
            prompt_template = prompt['TEMPLATE']
            message = message[:-1]
            # breakpoint()
            for shot in nShot:
                message.append({'role': 'user', 'content': shot['prompt']['prompt']})
                message.append({'role': 'assistant', 'content': shot['label']})
            message.append({'role': 'user', 'content': prompt['prompt']})
            prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        
        predicted_output, answer_token_prob, diff, rank = runRotatedModel(model, tokenizer, prompt, D, answer_token, L, H=H,moduleType=moduleType, rotationType=rotationMethod)
        # label = label['short']
        if predicted_output.strip().lower() == label['complete'].strip().lower() or predicted_output.strip().lower() == label['short'].strip().lower():
            accuracy += 1 
        count += 1
        prob += answer_token_prob
        logit_diff += diff
        matrix_rank += rank
        # breakpoint()

    avg_prob = prob / count
    logit_diff = logit_diff / count
    matrix_rank = matrix_rank / count
    print(f"Accuracy: {accuracy / count}")
    print(f"Answer token prob: {avg_prob}")
    print(f"Logit diff: {logit_diff}")
    print(f"Matrix rank: {matrix_rank}")
    
    if(metric == 'accuracy'):
        return -torch.tensor(accuracy / count, dtype=dtype)
    if(metric == 'prob'):
        return -torch.tensor(avg_prob, dtype=dtype)
    if(metric == 'logit_diff'):
        return -torch.tensor(logit_diff, dtype=dtype)
    if(metric == 'rank'):
        return torch.tensor(matrix_rank, dtype=dtype)

ibnn_kernel = InfiniteWidthBNNKernel(12, device=device)
ibnn_kernel.weight_var = 10.0
ibnn_kernel.bias_var = 5.0
ibnn_kernel = ScaleKernel(ibnn_kernel, device=device)
# ibnn = ScaleKernel(MaternKernel(), device=device)
# breakpoint()
# %%
try:
    train_X, train_Y, gp_state_dict, start_iteration = load_checkpoint(checkpoint_path)
    train_X = train_X.to('cpu')
    train_Y = train_Y.to('cpu')
    bounds = bounds.to('cpu')

    print(f"Resuming from iteration {start_iteration}")
    
    # Reconstruct the GP model
    gp = SingleTaskGP(train_X, train_Y, input_transform=Normalize(d=train_X.shape[-1]), outcome_transform=Standardize(m=1), covar_module=ibnn_kernel)
    gp.load_state_dict(gp_state_dict)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
except FileNotFoundError:
    # No checkpoint exists, start from scratch
    train_X = initial_points
    train_Y = torch.tensor([objective(x) for x in train_X], dtype=dtype, device=device).unsqueeze(-1)
    
    gp = SingleTaskGP(train_X, train_Y, input_transform=Normalize(d=train_X.shape[-1]), outcome_transform=Standardize(m=1),covar_module=ibnn_kernel)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    start_iteration = 0

# Define the number of iterations
n_iterations = 120
optimize_hypers = True

# Optimization loop
from tqdm import tqdm
for iteration in tqdm(range(start_iteration, n_iterations)):
    
    # if optimize_hypers:
    #     mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    #     fit_gpytorch_mll(mll)
    # gp.eval()

    EI = LogExpectedImprovement(model=gp, best_f=train_Y.min())
    
    new_x, _ = optimize_acqf(
        acq_function=EI,
        bounds=bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
    )
    new_y = objective(new_x)
    
    train_X = torch.cat([train_X, new_x])
    train_Y = torch.cat([
        train_Y.clone().detach().to(device).squeeze(-1), 
        new_y.clone().detach().to(device).unsqueeze(-1)
    ]).unsqueeze(-1)

    gp = SingleTaskGP(train_X, train_Y, input_transform=Normalize(d=train_X.shape[-1]), outcome_transform=Standardize(m=1),covar_module=ibnn_kernel)
    if(optimize_hypers):
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
    gp.eval()
    # breakpoint()
    
    save_checkpoint(checkpoint_path, train_X, train_Y, gp.state_dict(), iteration + 1)
    del new_x, new_y
    print(f"Iteration {iteration + 1}/{n_iterations}, best observed value: {train_Y.min().item()}")
    # optimized_params = train_X[train_Y.argmin()].view(num_L, H, N)
    # torch.save(optimized_params, f'final_parameter/optimized_params{fileName}.pt')


