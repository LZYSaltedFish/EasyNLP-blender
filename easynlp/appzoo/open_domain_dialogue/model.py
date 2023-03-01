# coding=utf-8
# Copyright (c) 2020 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..application import Application
from easynlp.modelzoo import TransformerConfig, TransformerModel
from ...modelzoo import AutoConfig, AutoModel
from transformers import GPT2LMHeadModel, GenerationConfig
import torch

class OpenDomainDialogue(Application):

    def __init__(self, pretrained_model_name_or_path=None, **kwargs):
        super().__init__()

        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        self.backbone = GPT2LMHeadModel.from_pretrained('IDEA-CCNL/Wenzhong-GPT2-110M')
        self.PAD_IDX = kwargs.get('pad_token_id')
        self.EOS_IDX = kwargs.get('eos_token_id')

        # if kwargs.get('from_config'):
        #     self.config = kwargs.get('from_config')
        #     self.backbone = AutoModel.from_config(self.config)
        # else:
        #     self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        #     self.backbone = AutoModel.from_pretrained(pretrained_model_name_or_path)
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.PAD_IDX, reduction='none'
        )
    
    def forward(self, inputs):
        # logits: bsz * output_len * vocab_size
        # preds: bsz * output_len

        # xs = inputs.get('input_ids', None)
        # ys = inputs.get('label', None)
        # outputs = self.backbone(xs, ys=ys)
        # logits, preds, hidden_states = outputs
        # return {
        #     "hidden": hidden_states,
        #     "logits": logits,
        #     "predictions": preds,
        #     "probabilities": torch.softmax(logits, dim=-1)
        # }

        outputs = self.backbone(**inputs)
        logits = outputs['logits']
        return {
            "logits": logits,
            "predictions": logits.max(dim=-1)[1],
            "probabilities": torch.softmax(logits, dim=-1)
        }

    def compute_loss(self, forward_outputs, label_ids, **kwargs):
        # logits_view: (bsz * output_len) * vocab_size
        # label_ids: bsz * output_len
        label_ids = label_ids[:, 1:]
        logits = forward_outputs['logits'][:, :-1]
        logits_view = logits.reshape(-1, logits.size(-1))
        loss = self.criterion(logits_view, label_ids.reshape(-1))
        loss = loss.view(forward_outputs['probabilities'][:, :-1].shape[:-1]).sum(dim=1)
        loss = loss.sum()

        notnull = label_ids.ne(self.PAD_IDX)
        target_tokens = notnull.long().sum()
        loss /= target_tokens
        return {"loss": loss}

    def compute_token_loss(self, forward_outputs, label_ids, **kwargs):
        label_ids = label_ids[:, 1:]
        logits = forward_outputs['logits'][:, :-1]
        logits_view = logits.reshape(-1, logits.size(-1))
        loss = self.criterion(logits_view, label_ids.reshape(-1))
        loss = loss.view(forward_outputs['probabilities'][:, :-1].shape[:-1]).sum(dim=1)
        return {'loss': loss}
    
    def _generate(self, input, beam_size, max_ts):
        # model_input = input['text_vec'].unsqueeze(0)
        self.backbone.eval()
        input_ids = input['text_vec'].view(1, -1)
        input_len = input_ids.shape[-1]
        # return self.backbone._generate(model_input, beam_size, max_ts)
        gen_config = GenerationConfig(min_new_tokens=2, max_new_tokens=max_ts, num_beams=beam_size)
        return self.backbone.generate(inputs=input_ids,
                                      generation_config=gen_config,
                                      pad_token_id=self.PAD_IDX,
                                      eos_token_id=self.EOS_IDX,
                                      do_sample=True,
                                      top_p=0.6)[:, input_len:]