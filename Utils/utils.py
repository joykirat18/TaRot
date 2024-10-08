def getRotationMatrixDimension(rotationMethod, N):
    if(rotationMethod == 'rotary'):
        return N // 2
    elif(rotationMethod == 'givens'):
        return N - 1
    elif(rotationMethod == 'scale'):
        return N
    else:
        raise ValueError(f"Unknown rotation method: {rotationMethod}")
    
def getModel(modelName):
    if(modelName == 'llama-3-8b'):
        MODEL_PATH = 'meta-llama/Meta-Llama-3-8B-Instruct'
    elif(modelName == 'phi-3'):
        MODEL_PATH = 'microsoft/Phi-3-mini-4k-instruct'
    elif(modelName == 'qwen_2'):
        MODEL_PATH = 'Qwen/Qwen2-1.5B-Instruct'
    elif(modelName == 'mixtral'):
        MODEL_PATH = 'mistralai/Mistral-7B-Instruct-v0.1'
    else:
        raise ValueError(f"Unknown model: {modelName}")
    return MODEL_PATH

def getInfo(model_name, task, scale=False):
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
        if(task == 'toxicity'):
            if(scale == True):
                fileName = ''
            else:
                fileName = ''
        if(task == 'imdb'):
            if(scale == True):
                fileName = ''
            else:
                fileName = ''
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
        if(task == 'toxicity'):
            if(scale == True):
                fileName = 'BaysianOptimization/responses/llama-3-8b/toxicity/Generation_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_scale/response_2.pkl'
            else:
                fileName = 'BaysianOptimization/responses/llama-3-8b/toxicity/Generation_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_rotary/response_1.pkl'
        if(task == 'imdb'):
            if(scale == True):
                fileName = 'BaysianOptimization/response/mixtral/imdb/Generation_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_rotary/response_2.pkl'
            else:
                fileName = 'BaysianOptimization/response/llama-3-8b/imdb/Generation_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_rotary/response_0.pkl'
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
        if(task == 'toxicity'):
            if(scale == True):
                fileName = 'BaysianOptimization/responses/qwen_2/toxicity/Generation_0_5_angle_-0.7853981633974483_0.7853981633974483_reasoning_prob_mix_scale/response_0.pkl'
            else:
                fileName = 'BaysianOptimization/responses/qwen_2/toxicity/Generation_0_5_angle_-0.7853981633974483_0.7853981633974483_reasoning_prob_mix_rotary/response_0.pkl'
        if(task == 'imdb'):
            if(scale == True):
                fileName = 'BaysianOptimization/response/qwen_2/imdb/Generation_0_5_angle_-0.7853981633974483_0.7853981633974483_reasoning_prob_mix_scale/response_0.pkl'
            else:
                fileName = 'BaysianOptimization/response/qwen_2/imdb/Generation_0_5_angle_-0.7853981633974483_0.7853981633974483_reasoning_prob_mix_rotary/response_2.pkl'
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
        if(task == 'toxicity'):
            if(scale == True):
                fileName = 'BaysianOptimization/responses/mixtral/toxicity/Generation_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_scale/response_4.pkl'
            else:
                fileName = 'BaysianOptimization/responses/mixtral/toxicity/Generation_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_rotary/response_1.pkl'
        if(task == 'imdb'):
            if(scale == True):
                fileName = 'BaysianOptimization/response/mixtral/imdb/Generation_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_scale/response_0.pkl'
            else:
                fileName = 'BaysianOptimization/response/mixtral/imdb/Generation_0_16_angle_-0.5235987755982988_0.5235987755982988_reasoning_prob_mix_rotary/response_2.pkl'
    return layers, fileName
