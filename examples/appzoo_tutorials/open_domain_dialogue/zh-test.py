from transformers import GPT2Tokenizer, GPT2LMHeadModel, GenerationConfig
import torch

tokenizer = GPT2Tokenizer.from_pretrained('IDEA-CCNL/Wenzhong-GPT2-3.5B')
# model = GPT2LMHeadModel.from_pretrained('IDEA-CCNL/Wenzhong-GPT2-3.5B')
history = "我之前有一次，直接"
target = "我也是完全没想到"
tokenizer.pad_token = tokenizer.unk_token
encoded_input = tokenizer(history, padding='max_length', max_length=128, truncation=True)
encoded_target = tokenizer(target, return_tensors='pt')
# encoded_input['input_ids'] = torch.cat((encoded_input['input_ids'], encoded_target['input_ids']), -1)
# history_mask = encoded_input['attention_mask']
# target_mask = encoded_target['attention_mask']
# # target_mask = 1.0 - target_mask
# encoded_input['attention_mask'] = torch.cat((history_mask, target_mask), -1)

output = model(**encoded_input)

# gen_config = GenerationConfig(min_length=10, max_new_tokens=128, num_beams=10)
# output = model.generate(**encoded_input, generation_config=gen_config)
# logits = output['logits']
# pred = logits.max(dim=-1)[1]
# response = tokenizer.decode(pred[0])

# response = tokenizer.decode(output[0])
print(output)