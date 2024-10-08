from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch
import pickle

def loadTransformerLensModel(modelPath):
    tokenizer = AutoTokenizer.from_pretrained(modelPath)
    hf_model = AutoModelForCausalLM.from_pretrained(modelPath, trust_remote_code=True)
    model = HookedTransformer.from_pretrained(modelPath, hf_model=hf_model, device='cpu', fold_ln=False, center_writing_weights=False, center_unembed=False, tokenizer=tokenizer)

    return model, tokenizer

def save_checkpoint(filepath, train_X, train_Y, gp_state_dict, iteration):
    # Save the checkpoint dictionary to a file
    checkpoint = {
        'train_X': train_X.cpu(),  # Save on CPU to avoid GPU-specific issues
        'train_Y': train_Y.cpu(),
        'gp_state_dict': gp_state_dict,
        'iteration': iteration
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)

def load_checkpoint(filepath):
    # Load the checkpoint dictionary
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    train_X = checkpoint['train_X']
    train_Y = checkpoint['train_Y']
    gp_state_dict = checkpoint['gp_state_dict']
    iteration = checkpoint['iteration']
    
    return train_X, train_Y, gp_state_dict, iteration