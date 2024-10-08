# %%
# model_name = 'microsoft/Phi-3-mini-4k-instruct'
# task = 'color'
# ag_news
# navigate
# color
def getInfo(model_name, task):
    if(model_name == 'microsoft/Phi-3-mini-4k-instruct'):
        layers = 12
        if(task == 'ag_news'):
            fileName = 'BaysianOptimization/error/phi-3/ag_news/NN_kernel_layer_0_6_angle_-0.7853981633974483_0.7853981633974483_reasoning_prob_mix_v2_0.pkl'
        if(task == 'navigate'):
            fileName = 'BaysianOptimization/error/phi-3/navigate/NN_kernel_layer_0_6_angle_-0.7853981633974483_0.7853981633974483_reasoning_prob_mix_v2_3.pkl'
        if(task == 'color'):
            fileName = 'BaysianOptimization/error/phi-3/color/NN_kernel_layer_0_6_angle_-0.7853981633974483_0.7853981633974483_reasoning_prob_mix_v2_0.pkl'
        if(task == 'entailed_polarity'):
            fileName = 'BaysianOptimization/error/phi-3/entailed_polarity/NN_kernel_layer_0_6_angle_-0.7853981633974483_0.7853981633974483_reasoning_prob_mix_v2_4.pkl'
        if(task == 'winowhy'):
            fileName = 'BaysianOptimization/error/phi-3/winowhy/NN_kernel_layer_0_6_angle_-0.7853981633974483_0.7853981633974483_reasoning_prob_mix_rotary_v2_0.pkl'
    if(model_name == 'meta-llama/Meta-Llama-3-8B-Instruct'):
        layers = 32
        if(task == 'ag_news'):
            fileName = 'BaysianOptimization/error/llama-3-8b/ag_news/NN_kernel_layer_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_rotary_v2_0.pkl'
        if(task == 'navigate'):
            fileName = 'BaysianOptimization/error/llama-3-8b/navigate/NN_kernel_layer_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_rotary_v2_4.pkl'
        if(task == 'color'):
            fileName = 'BaysianOptimization/error/llama-3-8b/color/NN_kernel_layer_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_rotary_v2_3.pkl'
        if(task == 'entailed_polarity'):
            fileName = 'BaysianOptimization/error/llama-3-8b/entailed_polarity/NN_kernel_layer_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_rotary_v2_1.pkl'
        if(task == 'winowhy'):
            fileName = 'BaysianOptimization/error/llama-3-8b/winowhy/NN_kernel_layer_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_rotary_v2_1.pkl'
    if(model_name == 'Qwen/Qwen2-1.5B-Instruct'):
        layers = 10
        if(task == 'ag_news'):
            fileName = 'BaysianOptimization/error/qwen_2/ag_news/NN_kernel_layer_0_5_angle_-0.7853981633974483_0.7853981633974483_reasoning_prob_mix_rotary_v2_0.pkl'
        if(task == 'navigate'):
            fileName = 'BaysianOptimization/error/qwen_2/navigate/NN_kernel_layer_0_5_angle_-0.7853981633974483_0.7853981633974483_reasoning_prob_mix_rotary_v2_2.pkl'
        if(task == 'color'):
            fileName = 'BaysianOptimization/error/qwen_2/color/NN_kernel_layer_0_5_angle_-0.7853981633974483_0.7853981633974483_reasoning_prob_mix_rotary_v2_2.pkl'
        if(task == 'entailed_polarity'):
            fileName = 'BaysianOptimization/error/qwen_2/entailed_polarity/NN_kernel_layer_0_5_angle_-0.7853981633974483_0.7853981633974483_reasoning_prob_mix_rotary_v2_2.pkl'
        if(task == 'winowhy'):
            fileName = 'BaysianOptimization/error/qwen_2/winowhy/NN_kernel_layer_0_5_angle_-0.7853981633974483_0.7853981633974483_reasoning_prob_mix_rotary_v2_2.pkl'
    if(model_name == 'mistralai/Mistral-7B-Instruct-v0.1'):
        layers = 32
        if(task == 'ag_news'):
            fileName = 'BaysianOptimization/error/mixtral/ag_news/NN_kernel_layer_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_rotary_v2_3.pkl'
        if(task == 'navigate'):
            fileName = 'BaysianOptimization/error/mixtral/navigate/NN_kernel_layer_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_rotary_v2_4.pkl'
        if(task == 'color'):
            fileName = 'BaysianOptimization/error/mixtral/color/NN_kernel_layer_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_rotary_v2_3.pkl'
        if(task == 'entailed_polarity'):
            fileName = 'BaysianOptimization/error/mixtral/entailed_polarity/NN_kernel_layer_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_rotary_v2_1.pkl'
        if(task == 'winowhy'):
            fileName = 'BaysianOptimization/error/mixtral/winowhy/NN_kernel_layer_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_rotary_v2_3.pkl'
    return layers, fileName
# %%
import pickle
from transformers import AutoTokenizer
modelNames = ['microsoft/Phi-3-mini-4k-instruct', 'meta-llama/Meta-Llama-3-8B-Instruct', 'Qwen/Qwen2-1.5B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.1']
tasks = ['ag_news','navigate', 'color', 'entailed_polarity', 'winowhy']
for model_name in modelNames:
    minlogit = []
    
    for task in tasks:
        layers, fileName = getInfo(model_name, task)
        with open(fileName, 'rb') as f:
            data = pickle.load(f)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f'Processing {model_name} for task {task}', len(data))
        if(len(data) == 50):
            import torch
            layer_wise = [0] * layers
            layer_wise_error = [0] * layers

            count = 0
            from tqdm import tqdm
            rotated_accuracy = 0
            normal_accuracy = 0
            for i in tqdm(range(len(data))):
                prompt = data[i]['prompt']
                label = data[i]['label']
                normal_token = data[i]['normal']
                rotated_token = data[i]['rotated']
                rotated_cache = data[i]['rotated_cache']
                normal_cache = data[i]['normal_cache']
                umembed = data[i]['umembed']
                answer_token = tokenizer.encode(label['complete'], add_special_tokens=False)[0]
                # print(answer_token, rotated_token, normal_token)
                if(label['complete'].startswith(rotated_token)):
                    
                    count += 1
                    for j in range(layers):
                        # print(j)
                        if(f'blocks.{j}.hook_resid_post' in rotated_cache):
                            key = f'blocks.{j}.hook_resid_post'
                        else:
                            key = f'blocks.{j}.hook_resid_mid'
                        normal_logits_layer = umembed(normal_cache[key].to('cuda')).detach().cpu()
                        rotated_logits_layer = umembed(rotated_cache[key].to('cuda')).detach().cpu()
                        
                        rotated_answer_token_prob = torch.nn.functional.softmax(rotated_logits_layer[:, -1, :], dim=-1)[0, answer_token].item()
                        normal_answer_token_prob = torch.nn.functional.softmax(normal_logits_layer[:, -1, :], dim=-1)[0, answer_token].item()
                        
                        _, sorted_indices = torch.sort(rotated_logits_layer[:, -1, :][0], descending=True)
                        rotated_rank = (sorted_indices == answer_token).nonzero(as_tuple=True)[0].item()
                        
                        _, sorted_indices = torch.sort(normal_logits_layer[:, -1, :][0], descending=True)
                        normal_rank = (sorted_indices == answer_token).nonzero(as_tuple=True)[0].item()
                        # print(rotated_rank, normal_rank)
                        # if(j == '31?otated_rank, normal_rank)
                        rotated_logit_normalised = (rotated_logits_layer[:, -1] - rotated_logits_layer[:, -1].min()) / (rotated_logits_layer[:, -1].max() - rotated_logits_layer[:, -1].min())
                        normal_logit_normalised = (normal_logits_layer[:, -1] - normal_logits_layer[:, -1].min()) / (normal_logits_layer[:, -1].max() - normal_logits_layer[:, -1].min())
                        
                        prob_diff = normal_answer_token_prob - rotated_answer_token_prob
                        layer_wise[j] += prob_diff
                        layer_wise_error[j] += prob_diff ** 2
        elif(len(data) == 51):
            import torch
            layer_wise = [0] * layers
            layer_wise_error = [0] * layers
            umembed = data[-1]['unembed']
            count = 0
            from tqdm import tqdm
            rotated_accuracy = 0
            normal_accuracy = 0
            for i in tqdm(range(len(data) - 1)):
                prompt = data[i]['prompt']
                label = data[i]['label']
                normal_token = data[i]['normal']
                rotated_token = data[i]['rotated']
                rotated_cache = data[i]['rotated_cache']
                normal_cache = data[i]['normal_cache']
                # umembed = data[i]['umembed']
                answer_token = tokenizer.encode(label['complete'], add_special_tokens=False)[0]
                # print(answer_token, rotated_token, normal_token)
                if(label['complete'].startswith(rotated_token)):
                    
                    count += 1
                    for j in range(layers):
                        # print(j)
                        if(f'blocks.{j}.hook_resid_post' in rotated_cache):
                            key = f'blocks.{j}.hook_resid_post'
                        else:
                            key = f'blocks.{j}.hook_resid_mid'
                        normal_logits_layer = umembed(normal_cache[key].to('cuda')).detach().cpu()
                        rotated_logits_layer = umembed(rotated_cache[key].to('cuda')).detach().cpu()
                        
                        rotated_answer_token_prob = torch.nn.functional.softmax(rotated_logits_layer[:, -1, :], dim=-1)[0, answer_token].item()
                        normal_answer_token_prob = torch.nn.functional.softmax(normal_logits_layer[:, -1, :], dim=-1)[0, answer_token].item()
                        
                        _, sorted_indices = torch.sort(rotated_logits_layer[:, -1, :][0], descending=True)
                        rotated_rank = (sorted_indices == answer_token).nonzero(as_tuple=True)[0].item()
                        
                        _, sorted_indices = torch.sort(normal_logits_layer[:, -1, :][0], descending=True)
                        normal_rank = (sorted_indices == answer_token).nonzero(as_tuple=True)[0].item()
                        # print(rotated_rank, normal_rank)
                        # if(j == '31?otated_rank, normal_rank)
                        rotated_logit_normalised = (rotated_logits_layer[:, -1] - rotated_logits_layer[:, -1].min()) / (rotated_logits_layer[:, -1].max() - rotated_logits_layer[:, -1].min())
                        normal_logit_normalised = (normal_logits_layer[:, -1] - normal_logits_layer[:, -1].min()) / (normal_logits_layer[:, -1].max() - normal_logits_layer[:, -1].min())
                        
                        prob_diff = normal_answer_token_prob - rotated_answer_token_prob
                        layer_wise[j] += prob_diff
                        layer_wise_error[j] += prob_diff ** 2
        
        for j in range(layers):
            # Mean difference for each layer
            layer_wise[j] /= count
            
            # Calculate standard deviation or standard error
            variance = (layer_wise_error[j] / count) - (layer_wise[j] ** 2)
            layer_wise_error[j] = torch.sqrt(torch.tensor(variance / count))
        import matplotlib.pyplot as plt

        layers_range = range(layers)
        plt.errorbar(layers_range, layer_wise, yerr=layer_wise_error, fmt='-o', capsize=5, label='Layer-wise Difference')
        plt.xlabel('Layers')
        plt.ylabel('Difference in Answer Token Probability')
        plt.title(f'Layer wise Prob Difference for Model {model_name}, Task {fileName.split("/")[-2]}')
        # plt.legend()
        # plt.show()
        # save pdf
        model_save = model_name.replace('/', '_')
        # plt.savefig(f'Model_{model_save}_Task_{fileName.split("/")[-2]}.pdf')
        # save svg
        plt.savefig(f'svg/Model_{model_save}_Task_{fileName.split("/")[-2]}.svg')
        plt.close()
        plt.clf()
        
# model_name = 'microsoft/Phi-3-mini-4k-instruct'

# layers = 12
# fileName = 'BaysianOptimization/error/phi-3/winowhy/NN_kernel_layer_0_6_angle_-0.7853981633974483_0.7853981633974483_reasoning_prob_mix_rotary_v2_0.pkl'

# %%
# BaysianOptimization/error/llama-3-8b/entailed_polarity/NN_kernel_layer_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_rotary_v2_2.pkl

# %%
# from transformers import AutoTokenizer

# # %%
# # tokenizer = AutoTokenizer.from_pretrained(model_name)

# # %%
# with open(fileName, 'rb') as f:
#     data = pickle.load(f)

# # %%
# import torch.nn.functional as F

# def kl_divergence(input_logit, target_logit):
#     input_next_token_logit = input_logit[:, -1, :]
#     target_next_token_logit = target_logit[:, -1, :]

#     input_logProb = F.softmax(input_next_token_logit, dim=-1)
#     target_logProb = F.softmax(target_next_token_logit, dim=-1)

#     kl_div = F.kl_div(input_logProb, target_logProb,log_target=True, reduction="none").sum(dim=-1)
#     return kl_div.mean().detach().cpu()

# # %%
# len(data)

# # %%
# umembed = data[-1]['unembed']

# # %%
# import torch
# layer_wise = [0] * layers
# layer_wise_error = [0] * layers

# count = 0
# from tqdm import tqdm
# rotated_accuracy = 0
# normal_accuracy = 0
# for i in tqdm(range(len(data) - 1)):
#     prompt = data[i]['prompt']
#     label = data[i]['label']
#     normal_token = data[i]['normal']
#     rotated_token = data[i]['rotated']
#     rotated_cache = data[i]['rotated_cache']
#     normal_cache = data[i]['normal_cache']
#     # umembed = data[i]['umembed']
#     answer_token = tokenizer.encode(label['complete'], add_special_tokens=False)[0]
#     # print(answer_token, rotated_token, normal_token)
#     if(label['complete'].startswith(rotated_token)):
        
#         count += 1
#         for j in range(layers):
#             # print(j)
#             normal_logits_layer = umembed(normal_cache[f'blocks.{j}.hook_resid_post'].to('cuda')).detach().cpu()
#             rotated_logits_layer = umembed(rotated_cache[f'blocks.{j}.hook_resid_post'].to('cuda')).detach().cpu()
            
#             rotated_answer_token_prob = torch.nn.functional.softmax(rotated_logits_layer[:, -1, :], dim=-1)[0, answer_token].item()
#             normal_answer_token_prob = torch.nn.functional.softmax(normal_logits_layer[:, -1, :], dim=-1)[0, answer_token].item()
            
#             _, sorted_indices = torch.sort(rotated_logits_layer[:, -1, :][0], descending=True)
#             rotated_rank = (sorted_indices == answer_token).nonzero(as_tuple=True)[0].item()
            
#             _, sorted_indices = torch.sort(normal_logits_layer[:, -1, :][0], descending=True)
#             normal_rank = (sorted_indices == answer_token).nonzero(as_tuple=True)[0].item()
#             # print(rotated_rank, normal_rank)
#             # if(j == '31?otated_rank, normal_rank)
#             rotated_logit_normalised = (rotated_logits_layer[:, -1] - rotated_logits_layer[:, -1].min()) / (rotated_logits_layer[:, -1].max() - rotated_logits_layer[:, -1].min())
#             normal_logit_normalised = (normal_logits_layer[:, -1] - normal_logits_layer[:, -1].min()) / (normal_logits_layer[:, -1].max() - normal_logits_layer[:, -1].min())
            
#             prob_diff = normal_answer_token_prob - rotated_answer_token_prob
#             layer_wise[j] += prob_diff
#             layer_wise_error[j] += prob_diff ** 2

#         # layer_wise[j].append(rotated_answer_token_prob - normal_answer_token_prob)
    
        
        
        
        
        
        
    
    
    

# # %%
# count

# # %%
# for j in range(layers):
#     # Mean difference for each layer
#     layer_wise[j] /= count
    
#     # Calculate standard deviation or standard error
#     variance = (layer_wise_error[j] / count) - (layer_wise[j] ** 2)
#     layer_wise_error[j] = torch.sqrt(torch.tensor(variance / count))

# # %%
# import matplotlib.pyplot as plt

# layers_range = range(layers)
# plt.errorbar(layers_range, layer_wise, yerr=layer_wise_error, fmt='-o', capsize=5, label='Layer-wise Difference')
# plt.xlabel('Layers')
# plt.ylabel('Difference in Answer Token Probability')
# plt.title(f'Layer wise Prob Difference for Model {model_name}, Task {fileName.split("/")[-2]}')
# # plt.legend()
# # plt.show()
# # save pdf
# model_save = model_name.replace('/', '_')
# # plt.savefig(f'Model_{model_save}_Task_{fileName.split("/")[-2]}.pdf')
# # save svg
# plt.savefig(f'svg/Model_{model_save}_Task_{fileName.split("/")[-2]}.svg')

# # %%
# import numpy as np
# layer_wise_mean = [np.mean(layer) for layer in layer_wise]
# layer_wise_std = [np.std(layer) for layer in layer_wise]  # This will be used as the error margin


# # %%
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# layers_range = range(layers)
# plt.errorbar(layers_range, layer_wise_mean, yerr=layer_wise_std, fmt='-o', capsize=5, label='Difference in Answer Token Probabilities')

# plt.xlabel('Layer')
# plt.ylabel('Difference in Answer Token Probability (Rotated - Normal)')
# plt.legend()
# plt.title(f'Layer wise Prob Difference for Model {model_name}, Task {fileName.split("/")[-2]}')


# # %%
# rotated_logits_layer[0, -1] - normal_logits_layer[0, -1]

# # %%
# print(rotated_accuracy, normal_accuracy, count)

# # %%
# for i in range(len(layer_wise)):
#     layer_wise[i] = layer_wise[i] / count

# # %%
# count

# # %%
# layer_wise

# # %%
# # plot the layer wise difference
# import matplotlib.pyplot as plt
# plt.plot(layer_wise)
# # xaxis
# plt.xlabel('Layer')
# # yaxis
# # plt.ylabel('KL Divergence')
# # plt.ylabel('logit difference')
# plt.ylabel('Prob Difference')
# # title
# plt.title(f'Layer wise Prob Difference for Model {model_name}, Task {fileName.split("/")[-2]}')
# # plt.title(f'Layer wise logit difference for Model {model_name}, Task {fileName.split("/")[-2]}')
# # plt.title(f'Layer wise KL Divergence for Model {model_name}, Task {fileName.split("/")[-2]}')
# # plt.title(f'Layer wise Rank Difference for Model {model_name}, Task {fileName.split("/")[-2]}')

# # %%
# torch.nn.functional.softmax(rotated_logits_layer[:, -1, :], dim=1).shape

# # %%
# import torch
# torch.nn.functional.softmax(normal_logits_layer, dim=-1).shape

# # %%



