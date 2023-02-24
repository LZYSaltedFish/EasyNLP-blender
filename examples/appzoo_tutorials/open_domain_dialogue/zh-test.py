from transformers import GPT2Tokenizer, GPT2LMHeadModel, GenerationConfig
import torch

tokenizer = GPT2Tokenizer.from_pretrained('IDEA-CCNL/Wenzhong-GPT2-110M')
# model = GPT2LMHeadModel.from_pretrained('IDEA-CCNL/Wenzhong-GPT2-110M')
history = "北京是中国的"
target = "我也是完全没想到"
pad_token = '<pad>'
bos_token = '<bos>'
tokenizer.add_tokens(pad_token)
tokenizer.add_tokens(bos_token)
tokenizer.pad_token = pad_token
tokenizer.bos_token = bos_token
# encoded_input = tokenizer(history, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
encoded_input = tokenizer(history, return_tensors='pt')
encoded_target = tokenizer(target, return_tensors='pt')
# encoded_input['input_ids'] = torch.cat((encoded_input['input_ids'], encoded_target['input_ids']), -1)
# history_mask = encoded_input['attention_mask']
# target_mask = encoded_target['attention_mask']
# # target_mask = 1.0 - target_mask
# encoded_input['attention_mask'] = torch.cat((history_mask, target_mask), -1)

# output = model(**encoded_input)

gen_config = GenerationConfig(min_new_tokens=10, max_new_tokens=80, num_beams=10)
output = model.generate(**encoded_input,
                        generation_config=gen_config,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                        num_return_sequences=5)
# logits = output['logits']
# pred = logits.max(dim=-1)[1]
# response = tokenizer.decode(pred[0])

# input_len = encoded_input['input_ids'].shape[-1]
# new_tokens = output[0, input_len:]
# response = history + tokenizer.decode(new_tokens)
# print(output)

for idx, sent in enumerate(output.sequences):
    print('next sent %d:\n'%idx, tokenizer.decode(sent).split('<|endoftext|>')[0])
    print('*'*40)