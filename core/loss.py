# Loss functions to compare the latents
import torch
from torch import nn

def get_positive_sum(interest_token_embeds, temperature):
    interest_token_embeds = interest_token_embeds.view(-1, 4 * 64 * 64)
    num_samples = interest_token_embeds.shape[0]
    # Normalize token predictions
    interest_token_embeds = interest_token_embeds / (interest_token_embeds.norm(dim=1)[:, None] + 1e-20)
    
    # Calculate similarity values
    token_sim = torch.mm(interest_token_embeds, interest_token_embeds.T)
    token_sim = torch.exp(token_sim / temperature)

    # Mask for positive samples
    if num_samples == 0:
        print("ERROR")

    # Calculate for different latent with same directions - upper triangle to not get repeating similarities
    mask = (torch.ones((num_samples, num_samples)).triu() * (1 - torch.eye(num_samples))).to(interest_token_embeds.device).bool()
    pos_sum = token_sim.masked_select(mask).view(mask.sum(), -1).sum()
    pos_count = mask.sum()

    return pos_sum, pos_count

def get_negative_sum(interest_token_embeds, negative_prompt_embeds, temperature):
    interest_token_embeds = interest_token_embeds.view(-1, 4 * 64 * 64)
    negative_token_embeds = negative_prompt_embeds.reshape(-1, 4 * 64 * 64)
    # Normalize the token_predictions
    interest_token_embeds = interest_token_embeds / (interest_token_embeds.norm(dim=1)[:, None] + 1e-20)
    negative_token_embeds = negative_token_embeds / (negative_token_embeds.norm(dim=1)[:, None] + 1e-20)
                
    # Calculate the similarity values
    token_sim = torch.mm(interest_token_embeds, negative_token_embeds.T)
    token_sim = torch.exp(token_sim / temperature)
                
    # Mask for negative samples 
    # Calculate for same latent with different directions - with diagonal
    mask = torch.eye(interest_token_embeds.shape[0]).to(interest_token_embeds.device).bool()
    neg_sum = token_sim.masked_select(mask).view(mask.sum(), -1).sum()
    neg_count = mask.sum()

    return neg_sum, neg_count