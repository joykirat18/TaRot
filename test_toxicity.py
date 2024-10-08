import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()
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
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
# from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

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
parser.add_argument('--rotationMethod', type=str, default='rotary', help='The rotation method to use', choices=['rotary', 'givens', 'scale'])
parser.add_argument('--angle_0', type=str)
parser.add_argument('--angle_1', type=str)

arg = parser.parse_args()

from Utils.datasetUtil import *
from Utils.checkpointUtil import *
from Utils.intervention import *
from tqdm import tqdm
from Utils.utils import *

modelPath = arg.model.strip()
moduleType = arg.moduleType
metric = arg.metric
fewShotCategory = arg.fewShotCategory
testingDataSize = arg.testingDataSize
trainingDataSize = arg.trainingDataSize
rotationMethod = arg.rotationMethod
MODEL_PATH = getModel(modelPath)

model, tokenizer = loadTransformerLensModel(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token


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
# %%
angles = [(angle_0) * torch.pi, (angle_1) * torch.pi]
n_initial_points = 4  # Number of initial points
dtype = torch.float64
device = 'cuda' #cuda
model = model.to(device)

import torch
import numpy as np

import torch

if(moduleType == 'reasoning'):
    bounds = torch.tensor([[angles[0]] * (num_L * H * N), [angles[1]] * (num_L * H * N)], dtype=dtype).to(device)
if(moduleType == 'mlp'):
    bounds = torch.tensor([[angles[0]] * (num_L * N), [angles[1]] * (num_L * N)], dtype=dtype).to(device)
if(rotationMethod == 'scale'):
    bounds = torch.tensor([[0] * (num_L * H), [1] * (num_L * H)], dtype=dtype).to(device)
initial_points = (torch.rand((n_initial_points, bounds.size(1)), device=device) * (bounds[1] - bounds[0]) + bounds[0]).to(bounds.device)


# %%
sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}
# %%
import wandb

wandb.init()
# %%

def build_dataset(
    tokenizer, dataset_name="allenai/real-toxicity-prompts", input_min_text_length=5, input_max_text_length=10
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    # tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name, split="train")

    def filter_fn(sample):
        toxicity = sample["prompt"]["toxicity"]
        return toxicity is not None and toxicity > 0.3

    ds = ds.filter(filter_fn, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        prompt = sample["prompt"]["text"]
        continuation = sample["continuation"]["text"]

        sample["input_ids"] = tokenizer.encode(prompt + continuation)[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]

    return ds
min_input_length = 30
max_input_length = 40
dataset = build_dataset(tokenizer, input_min_text_length=min_input_length, input_max_text_length=max_input_length)
# train_dataset = dataset[:500]
# test_dataset = dataset[100:]

train_dataset = dataset[:500]
test_dataset = dataset[500:500 + testingDataSize]
# %%
fileName = f'{modelPath}/toxicity/Generation_{L[0]}_{L[1]}_angle_{angles[0]}_{angles[1]}_{moduleType}_{metric}_{fewShotCategory}_{rotationMethod}'
checkpoint_path = f'checkpoint/{fileName}.pkl'

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
# We load the toxicity model in fp16 to save memory.
toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id, torch_dtype=torch.float16).to(
    device
)
gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id}

output_min_length = 10
output_max_length = 25
output_length_sampler = LengthSampler(output_min_length, output_max_length)
# %%
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
# %%
# test_dataset = test_dataset[:testingDataSize]
def objective(params, rotate=True):
# run batch for dataset
    # test_dataset = dataset[:testingDataSize]
    if(moduleType == 'reasoning'):
        if(rotationMethod == 'scale'):
            D = params.view(num_L, H)
        else:
            D = params.view(num_L, H, N)
    if(moduleType == 'mlp'):
        D = params.view(num_L, N)  # Ensure the tensor is reshaped correctly
    
    query = []
    response = []
    for i in tqdm(range(len(test_dataset['query']))):
        prompt = test_dataset['query'][i]
        query.append(prompt)
        gen_len = output_length_sampler()
        # breakpoint()
        message = [{'role': 'user', 'content': prompt}]
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        if(rotate):
            generated_text = generateRotatedModel(model, tokenizer, prompt, D, L, H, device, gen_len=gen_len, rotationType=rotationMethod)
        else:
            generated_text = genererateNormalModel(model, tokenizer, prompt, D, L, H, device, gen_len=gen_len)
        # generated_text = generateRotatedModel(model, tokenizer, prompt, D, L, H, device, gen_len=gen_len)
        response.append(generated_text)
        if(i == testingDataSize):
            break

    texts = [r for q, r in zip(query, response)]
    toxicity_inputs = toxicity_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
        device
    )
    logits = toxicity_model(**toxicity_inputs).logits.float()
    toxicity_labels = (logits[:, 0]).tolist()

    rewards = [torch.tensor(output) for output in toxicity_labels]
    print(f"Average reward: {sum(rewards) / len(rewards)}")
    
    # breakpoint()
    return sum(rewards) / len(rewards), response
# %%

Normal_reward, responses = objective(initial_points[0], rotate=False)
import pickle
import os
os.makedirs(f'responses/{fileName}', exist_ok=True)
with open(f'responses/{fileName}/base.pkl', 'wb') as f:
    pickle.dump(responses, f)
rotated_reward = []

train_X, train_Y, _, iteration = load_checkpoint(checkpoint_path)
print(iteration)

combined_array = [(x, y) for x, y in zip(train_X, train_Y)]
sorted_array = sorted(combined_array, key=lambda x: x[1], reverse=False)
# breakpoint()
for i in range(5):
    D = sorted_array[i][0]
    reward, responses = objective(D, rotate=True)
    import pickle
    import os
    os.makedirs(f'responses/{fileName}', exist_ok=True)
    with open(f'responses/{fileName}/response_{i}.pkl', 'wb') as f:
        pickle.dump(responses, f)
    print(f"Rotated Reward: {reward}")
    rotated_reward.append(reward)
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
    'rotated_reward': rotated_reward,
    'normal_reward': Normal_reward
})

with open('result.json', 'w') as f:
    json.dump(data, f, indent=4)