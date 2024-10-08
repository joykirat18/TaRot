import torch
from jaxtyping import Float, Int
import transformer_lens.utils as utils
from functools import partial
def create_rope_rotation_matrix(angles):
    """
    Create a RoPE (Rotary Position Embedding) rotation matrix as a tensor for a given vector of angles.

    Parameters:
    angles (torch.Tensor): A 1D PyTorch tensor of rotation angles for each dimension pair.

    Returns:
    torch.Tensor: A 2D PyTorch tensor representing the RoPE rotation matrix.
    """
    n = angles.shape[0]  # Number of angles
    d = 2 * n  # Dimension of the square matrix (embedding dimension must be even)
    rope_matrix = torch.zeros((d, d))

    # Fill the rotation matrix with 2x2 rotation matrices for each dimension pair
    for i, angle in enumerate(angles):
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        # Set the 2x2 rotation submatrix
        rope_matrix[2 * i, 2 * i] = cos_angle
        rope_matrix[2 * i, 2 * i + 1] = -sin_angle
        rope_matrix[2 * i + 1, 2 * i] = sin_angle
        rope_matrix[2 * i + 1, 2 * i + 1] = cos_angle

    return rope_matrix

def givens_rotation(i, j, theta, size):
    """
    Create a Givens rotation matrix of size 'size' rotating between indices i and j by angle theta.
    """
    G = torch.eye(size)
    c = torch.cos(theta)
    s = torch.sin(theta)
    
    G[i, i] = c
    G[i, j] = -s
    G[j, i] = s
    G[j, j] = c
    
    return G
def create_givens_rotation_matrix(angles):
    """
    Compute the product of Givens matrices as described in the equation.
    
    Parameters:
    d (int): The dimension of the matrix.
    thetas (list of torch.Tensor): List of angles for each Givens rotation.
    
    Returns:
    torch.Tensor: The resulting matrix after applying the product of Givens rotations.
    """
    d = len(angles) + 1  # Dimension of the matrix
    P = torch.eye(d)  # Start with identity matrix of size d
    num_levels = int(torch.log2(torch.tensor(d)).item())  # Log base 2 of the dimension
    
    flag = 0  # To track theta indices
    
    # Loop over levels r
    for r in range(num_levels):
        step_size = 2 ** r
        num_rotations = d // (2 * step_size)
        
        # Loop over each k within the current level
        for k in range(num_rotations):
            i = 2 * step_size * k  # i corresponds to 2^r * k
            j = i + step_size  # j corresponds to 2^(r-1)(2k+1), which simplifies to i + step_size
            
            # Get the corresponding theta for this rotation
            theta = angles[flag]
            flag += 1  # Move to the next theta
            
            # Create the Givens rotation matrix for this pair (i, j) with angle theta
            G = givens_rotation(i, j, theta, d)
            
            # Multiply the current product matrix P by the new Givens matrix
            P = torch.mm(G, P)
    # breakpoint()
    return P

def rotateMatrixReasoning(
        clean_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
        hook,
        head_index,
        rotaryMatrix):
    # assert(clean_head_vector[:, :, head_index, :] == clean_head_vector[:, :, head_index, :] @ rotaryMatrix).all()
    # breakpoint()
    if(len(rotaryMatrix.shape) == 0):
        clean_head_vector[:, :, head_index, :] = clean_head_vector[:, :, head_index, :] * rotaryMatrix
    else:
        clean_head_vector[:, :, head_index, :] = clean_head_vector[:, :, head_index, :] @ rotaryMatrix
    return clean_head_vector

def rotateMatrixMLP(
   clean_mlp_vector: Float[torch.Tensor, "batch pos d_model"],
    hook,
    rotaryMatrix):
    clean_mlp_vector = clean_mlp_vector @ rotaryMatrix
    return clean_mlp_vector 
def runRotatedModel(model, tokenizer, prompt, D, answer_token, L, H=32,device='cuda', moduleType='reasoning', rotationType='', store_cache=False):
    if(type(prompt) == dict):
        prompt = prompt['combined']
    model.reset_hooks()
    # breakpoint()
    cache = {}
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
    encoded_prompt = encoded_prompt.to(device)
    cache = {}
    def store_hook_cache(activation, hook):
        cache[hook.name] = torch.from_numpy(activation.detach().cpu().numpy())
    
    list_fwd_hooks = []
    if(moduleType == 'reasoning'):
        for layer in range(L[0], L[-1]):
            for head in range(H):
                # breakpoint()
                if(rotationType == 'rotary'):
                    rotaryMatrix = create_rope_rotation_matrix(D[layer - L[0], head]).to(device)
                elif(rotationType == 'givens'):
                    rotaryMatrix = create_givens_rotation_matrix(D[layer - L[0], head]).to(device)
                elif(rotationType == 'scale'):
                    rotaryMatrix = D[layer - L[0], head].to(device)
                else:
                    raise ValueError("Invalid rotation type")
                # rotaryMatrix = create_rope_rotation_matrix(D[layer - L[0], head]).to(device)
                list_fwd_hooks.append((utils.get_act_name("z", layer, "attn"), partial(rotateMatrixReasoning, head_index=head, rotaryMatrix=rotaryMatrix)))
        if(store_cache):
            for layer in range(L[0], L[-1] * 2):
                list_fwd_hooks.append((utils.get_act_name("resid_post", layer), store_hook_cache))
                # list_fwd_hooks.append((utils.get_act_name("attn_out", layer), store_hook_cache))  
                # list_fwd_hooks.append((utils.get_act_name("mlp_out", layer), store_hook_cache))  
            # list_fwd_hooks.append((utils.get_act_name("z", layer, "attn"), store_hook_cache))
    elif(moduleType == 'mlp'):
        # breakpoint()
        for layer in range(L[0], L[-1]):
            rotaryMatrix = create_rope_rotation_matrix(D[layer - L[0]]).to(device)
            list_fwd_hooks.append((utils.get_act_name("mlp_out", layer), partial(rotateMatrixMLP, rotaryMatrix=rotaryMatrix)))
    rotated_logits = model.run_with_hooks(encoded_prompt, return_type="logits", fwd_hooks=list_fwd_hooks)
    # breakpoint()
    import copy
    rotated_cache = copy.deepcopy(cache)
    cache = {}
    list_fwd_hooks = []
    if(store_cache):
        for layer in range(L[0], L[-1] * 2):
            list_fwd_hooks.append((utils.get_act_name("resid_post", layer), store_hook_cache))
            # list_fwd_hooks.append((utils.get_act_name("attn_out", layer), store_hook_cache))
            # list_fwd_hooks.append((utils.get_act_name("mlp_out", layer), store_hook_cache))
        normal_logits = model.run_with_hooks(encoded_prompt, return_type="logits", fwd_hooks=list_fwd_hooks)
    else:
        normal_logits = model(encoded_prompt)
    predicted_token = torch.argmax(rotated_logits[:, -1, :], dim=-1)[0]
    answer_token_prob = torch.nn.functional.softmax(rotated_logits[:, -1, :], dim=1)[0, answer_token].item()
    answer_token_logit_rotated = rotated_logits[:, -1, answer_token].item()
    answer_token_logit_normal = normal_logits[:, -1, answer_token].item()
    token = tokenizer.decode(predicted_token, skip_special_tokens=True)
    sorted_logits, sorted_indices = torch.sort(rotated_logits[:, -1, :][0], descending=True)
    rank = (sorted_indices == answer_token).nonzero(as_tuple=True)[0].item()
    # breakpoint()
    rotated_token = tokenizer.decode(torch.argmax(rotated_logits[:, -1, :], dim=1), skip_special_tokens=True)
    normal_token = tokenizer.decode(torch.argmax(normal_logits[:, -1, :], dim=1), skip_special_tokens=True)
    normal_cache = copy.deepcopy(cache)
    del encoded_prompt, rotated_logits, list_fwd_hooks, cache
    return token, answer_token_prob, answer_token_logit_rotated - answer_token_logit_normal, rank, rotated_token, normal_token, rotated_cache, normal_cache


def generateRotatedModel(model, tokenizer, prompt, D, L, H=32, device='cuda', gen_len=10, rotationType=''):
    
    model.reset_hooks()
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt')
    original_size = input_ids.size(1)
    input_ids = input_ids.to(device)
    list_fwd_hooks = []
    for layer in range(L[0], L[-1]):
        for head in range(H):
            # breakpoint()
            if(rotationType == 'rotary'):
                rotaryMatrix = create_rope_rotation_matrix(D[layer - L[0], head]).to(device)
            elif(rotationType == 'givens'):
                rotaryMatrix = create_givens_rotation_matrix(D[layer - L[0], head]).to(device)
            elif(rotationType == 'scale'):
                rotaryMatrix = D[layer - L[0], head].to(device)
            else:
                raise ValueError("Invalid rotation type")
            
            # rotaryMatrix = create_rope_rotation_matrix(D[layer - L[0], head]).to(device)
            list_fwd_hooks.append((utils.get_act_name("z", layer, "attn"), partial(rotateMatrixReasoning, head_index=head, rotaryMatrix=rotaryMatrix)))
    for _ in range(gen_len):
        output_logits = model.run_with_hooks(input_ids, return_type="logits", fwd_hooks=list_fwd_hooks)
        next_token_logits = output_logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        if(next_token.item() == tokenizer.eos_token_id):
            break
    generated_text = tokenizer.decode(input_ids[0][original_size:], skip_special_tokens=False)
    # breakpoint()
    return generated_text

def genererateNormalModel(model, tokenizer, prompt, D, L, H=32, device='cuda', gen_len=10):
    
    model.reset_hooks()
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt')
    original_size = input_ids.size(1)
    input_ids = input_ids.to(device)
    
    for _ in range(gen_len):
        outputs = model(input_ids)
        # breakpoint()
        # logits = outputs[0]
        next_token_logits = outputs[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        if(next_token.item() == tokenizer.eos_token_id):
            break
    generated_text = tokenizer.decode(input_ids[0][original_size:], skip_special_tokens=False)
    # breakpoint()
    return generated_text