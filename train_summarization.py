# %%
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
parser.add_argument('--rotationMethod', type=str, default='rotary', help='The rotation method to use', choices=['rotary', 'givens'])
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

initial_points = (torch.rand((n_initial_points, bounds.size(1)), device=device) * (bounds[1] - bounds[0]) + bounds[0]).to(bounds.device)


# %%
sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}
# %%
import wandb

wandb.init()
# %%

def build_dataset(tokenizer, dataset_name="stanfordnlp/imdb", input_min_text_length=2, input_max_text_length=8):
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

dataset = build_dataset(tokenizer)
train_dataset = dataset[:500]
test_dataset = dataset[100:]
# %%
import os
os.makedirs(f'TrainData/{modelPath}/imdb', exist_ok=True)
os.makedirs(f'TestData/{modelPath}/imdb', exist_ok=True)
os.makedirs(f'checkpoint/{modelPath}/imdb', exist_ok=True)
# os.make
# %%
fileName = f'{modelPath}/imdb/Generation_{L[0]}_{L[1]}_angle_{angles[0]}_{angles[1]}_{moduleType}_{metric}_{fewShotCategory}_{rotationMethod}'
checkpoint_path = f'checkpoint/{fileName}.pkl'

with open(f'TrainData/{fileName}', 'wb') as f:
    pickle.dump(train_dataset, f)
with open(f'TestData/{fileName}.pkl', 'wb') as f:
    pickle.dump(test_dataset, f)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
# %%

sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)
# %%
text = "this movie was really bad!!"
sentiment_pipe(text, **sent_kwargs)
# %%
gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id}

# %%
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
def objective(params):
# run batch for dataset
    train_dataset = dataset[:trainingDataSize]
    D = params.view(num_L, H, N)  # Ensure the tensor is reshaped correctly
    
    query = []
    response = []
    for i in tqdm(range(len(train_dataset['review']))):
        prompt = train_dataset['query'][i]
        query.append(prompt)
        gen_len = output_length_sampler()
        # breakpoint()
        message = [{'role': 'user', 'content': prompt}]
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        generated_text = generateRotatedModel(model, tokenizer, prompt, D, L, H, device, gen_len=gen_len)
        pipe_outputs = sentiment_pipe(generated_text, **sent_kwargs)
        response.append(generated_text)
        if(i == trainingDataSize):
            break

    texts = [q + r for q, r in zip(query, response)]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    positive_scores = [item["score"] for output in pipe_outputs for item in output if item["label"] == "POSITIVE"]
    rewards = [torch.tensor(score) for score in positive_scores]
    print(f"Average reward: {sum(rewards) / len(rewards)}")
    # breakpoint()
    return sum(rewards) / len(rewards)
        
        # for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
        # query_tensors = batch["input_ids"]

#         #### Get response from the model (token by token generation)
#         response_tensors = []
#         for query in query_tensors:
#             gen_len = output_length_sampler()
#             input_ids = query.unsqueeze(0)  # Ensure batch dimension is present

#             # Token-by-token generation
#             for _ in range(gen_len):
#                 outputs = model(input_ids)
#                 logits = outputs[0]
#                 next_token_logits = logits[:, -1, :]  # Get the logits of the last generated token
#                 next_token = torch.argmax(next_token_logits, dim=-1)  # Get the token with the highest probability

#                 # Append the new token to the sequence
#                 input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

#                 # Stop if the EOS token is generated
#                 if next_token.item() == tokenizer.eos_token_id:
#                     break

#             response_tensors.append(input_ids.squeeze())
#         batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

#         #### Compute sentiment score
#         texts = [q + r for q, r in zip(batch["query"], batch["response"])]
#         pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
#         positive_scores = [item["score"] for output in pipe_outputs for item in output if item["label"] == "POSITIVE"]
#         rewards = [torch.tensor(score) for score in positive_scores]

#     #### Run PPO step
# stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
# ppo_trainer.log_stats(stats, batch, rewards)

# %%
ibnn_kernel = InfiniteWidthBNNKernel(12, device=device)
ibnn_kernel.weight_var = 10.0
ibnn_kernel.bias_var = 5.0
ibnn_kernel = ScaleKernel(ibnn_kernel, device=device)
# %%
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
n_iterations = 150
optimize_hypers = True

# Optimization loop
from tqdm import tqdm
for iteration in tqdm(range(start_iteration, n_iterations)):
    
    # if optimize_hypers:
    #     mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    #     fit_gpytorch_mll(mll)
    # gp.eval()

    EI = LogExpectedImprovement(model=gp, best_f=train_Y.max())
    
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
    print(f"Iteration {iteration + 1}/{n_iterations}, best observed value: {train_Y.max().item()}")
    
    
# %%
# bs = 16
# game_data = dict()
# dataset.set_format("pandas")
# df_batch = dataset[:].sample(bs)
# game_data["query"] = df_batch["query"].tolist()
# query_tensors = df_batch["input_ids"].tolist()

# response_tensors_ref, response_tensors = [], []

# #### get response from gpt2 and gpt2_ref
# for i in range(bs):
#     query = torch.tensor(query_tensors[i]).to(device)

#     gen_len = output_length_sampler()
#     query_response = ref_model.generate(query.unsqueeze(0), max_new_tokens=gen_len, **gen_kwargs).squeeze()
#     response_len = len(query_response) - len(query)
#     response_tensors_ref.append(query_response[-response_len:])

#     query_response = model.generate(query.unsqueeze(0), max_new_tokens=gen_len, **gen_kwargs).squeeze()
#     response_len = len(query_response) - len(query)
#     response_tensors.append(query_response[-response_len:])

# #### decode responses
# game_data["response (before)"] = [tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
# game_data["response (after)"] = [tokenizer.decode(response_tensors[i]) for i in range(bs)]

# #### sentiment analysis of query/response pairs before/after
# texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
# pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
# positive_scores = [item["score"] for output in pipe_outputs for item in output if item["label"] == "POSITIVE"]
# game_data["rewards (before)"] = positive_scores

# texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
# pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
# positive_scores = [item["score"] for output in pipe_outputs for item in output if item["label"] == "POSITIVE"]
# game_data["rewards (after)"] = positive_scores

# # store results in a dataframe
# df_results = pd.DataFrame(game_data)
# df_results