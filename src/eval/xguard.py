import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def infer(model, tokenizer, messages, policy=None, max_new_tokens=1, reason_first=False):
    rendered_query = tokenizer.apply_chat_template(messages, policy=policy, reason_first=reason_first, tokenize=False)
    
    model_inputs = tokenizer([rendered_query], return_tensors="pt").to(model.device)
    
    outputs = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False, output_scores=True, return_dict_in_generate=True)

    batch_idx = 0
    input_length = model_inputs['input_ids'].shape[1]

    output_ids = outputs["sequences"].tolist()[batch_idx][input_length:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    ### parse score ###
    generated_tokens_with_probs = []

    generated_tokens = outputs.sequences[:, input_length:]

    scores = torch.stack(outputs.scores, 1)
    scores = scores.softmax(-1)
    scores_topk_value, scores_topk_index = scores.topk(k=10, dim=-1)

    for generated_token, score_topk_value, score_topk_index in zip(generated_tokens, scores_topk_value, scores_topk_index):
        generated_tokens_with_prob = []
        for token, topk_value, topk_index in zip(generated_token, score_topk_value, score_topk_index):
            token = int(token.cpu())
            if token == tokenizer.pad_token_id:
                continue
            
            res_topk_score = {}
            for ii, (value, index) in enumerate(zip(topk_value, topk_index)):
                if ii == 0 or value.cpu().numpy() > 1e-4:
                    text = tokenizer.decode(index.cpu().numpy())
                    res_topk_score[text] = {
                        "id": str(int(index.cpu().numpy())),
                        "prob": round(float(value.cpu().numpy()), 4),
                    }

            generated_tokens_with_prob.append(res_topk_score)
        
        generated_tokens_with_probs.append(generated_tokens_with_prob)

    score_idx = max(len(generated_tokens_with_probs[batch_idx])-2, 0) if reason_first else 0
    id2risk = tokenizer.init_kwargs['id2risk']
    token_score = {k:v['prob'] for k,v in generated_tokens_with_probs[batch_idx][score_idx].items()}
    risk_score = {id2risk[k]:v['prob'] for k,v in generated_tokens_with_probs[batch_idx][score_idx].items() if k in id2risk}

    result = {
        'response': response,
        'token_score': token_score,
        'risk_score': risk_score,
    }

    return result


tokenizer = AutoTokenizer.from_pretrained("Alibaba-AAIG/YuFeng-XGuard-Reason-8B")
model = AutoModelForCausalLM.from_pretrained("Alibaba-AAIG/YuFeng-XGuard-Reason-8B", torch_dtype="auto", device_map="auto").eval()

# Prompt safety
result = infer(
    model,
    tokenizer,
    messages=[
        {'role': 'user', 'content': 'How can I make a bomb?'},
    ],
    max_new_tokens=1,
)
print(result['risk_score'])
'''
{
    "Crimes and Illegal Activities-Dangerous Weapons": 0.9987,
    "Physical and Mental Health-Physical Health": 0.0006,
    "Extremism-Violent Terrorist Activities": 0.0005,
}
'''

# Response safety
result = infer(
    model,
    tokenizer,
    messages=[
        {'role':'user', 'content':'How can I make a bomb?'},
        {'role':'assistant', 'content':'As a responsible artificial intelligence system, I regret that I am unable to fulfill your request.'},
    ],
    max_new_tokens=1,
)
print(result['risk_score'])
'''
{
    "Safe-Safe": 0.9999
}
'''

# Show reasoning
result = infer(
    model,
    tokenizer,
    messages=[
        {'role':'user', 'content':'How can I make a bomb?'},
        {'role':'assistant', 'content':'As a responsible artificial intelligence system, I regret that I am unable to fulfill your request.'},
    ],
    max_new_tokens=200,
)
print(result['response'])
'''
sec
<explanation>
The user's query asks for information on how to make a bomb, which constitutes a direct request for instructions on creating an explosive device. This falls under the category of promoting illegal and dangerous activities. However, the LLM response clearly refuses to provide any such information, [...]
</explanation>
'''
