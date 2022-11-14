import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.base import BaseTransformer, ConstraintParser, ConstraintRewardFunction, Evaluator
from ad.airdialogue import InstructionScene
from typing import Any, Dict, Union, List, Optional
from utils.misc import PrecisionRecallAcc, stack_dicts, unstack_dicts
from utils.sampling_utils import *
import numpy as np
from utils.top_k_constraints import top_k_constraints
from utils.torch_utils import get_transformer_logs
from transformers import RobertaModel, RobertaTokenizer
from utils.data_utils import DiscreteFeatures
from utils.instructions import instruction_def

class InstructionPredictorRoberta(BaseTransformer, nn.Module):
    def __init__(self, 
                 roberta_type: str = "roberta-base", 
                 device: Union[torch.device, str] = "cuda", 
                 max_length: Optional[int] = None):
        nn.Module.__init__(self)
        self.roberta_type = roberta_type
        tokenizer = RobertaTokenizer.from_pretrained(self.roberta_type)
        model = RobertaModel.from_pretrained(self.roberta_type)
        BaseTransformer.__init__(self, model, tokenizer, device, max_length)
        ConstraintParser.__init__(self)
        self.h_dim = self.model.config.hidden_size
        # self.ff = nn.Linear(self.h_dim, self.h_dim*2)
        # self.prediction_head = nn.Linear(self.h_dim*2, len(instruction_def))
        self.ffs = nn.ModuleDict({'instruction_head': nn.Linear(self.h_dim, self.h_dim*2)})
        self.prediction_heads = nn.ModuleDict({'instruction_head': nn.Linear(self.h_dim*2, len(instruction_def))})
         
        # self.ffs = nn.ModuleDict({k: nn.Linear(self.h_dim, self.h_dim*2)
        #                           for k, _ in self.discrete_features.get_emb_spec().items()})
        # self.prediction_heads = nn.ModuleDict({k: nn.Linear(self.h_dim*2, num_items)
        #                                        for k, num_items in self.discrete_features.get_emb_spec().items()})
        self.attn_proj = nn.Linear(self.h_dim, 1)
        # self.param_groups = [
        #                      ((self.attn_proj, self.ff, self.prediction_head,), lambda config: {'lr': config['lr']}), 
        #                      ((self.model,), lambda config: {'lr': config['roberta_lr'],}), 
        #                     ]
        self.param_groups = [
                             ((self.attn_proj, self.ffs, self.prediction_heads,), lambda config: {'lr': config['lr']}), 
                             ((self.model,), lambda config: {'lr': config['roberta_lr'],}), 
                            ]
    
    def format(self, scene: InstructionScene):
        # event_list = [str(event) for event in scene.data]
        # if(len(event_list)>=128):
        #     event_list = event_list[:127]
        # return ','.join(event_list)
        # formatted_str = ','.join([str(event) for event in scene.data])
        # print(formatted_str)
        # if len(formatted_str) >= 128:  
        #     formatted_str = formatted_str[:127]
        #     print('###### '+formatted_str)
        # return formatted_str
        return ','.join([str(event) for event in scene.data])
    
    # def _tokenize_customer_scenarios_batch(self, scenarios: List[CustomerScenario]):
    #     return {k: torch.tensor(v).to(self.device) for k, v in stack_dicts([scenario.get_discrete_state(self.discrete_features) 
    #                                                                         for scenario in scenarios]).items()}
    
    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor, **kwargs):
        model_output = self.model(input_ids=tokens, attention_mask=attn_mask, **kwargs)
        hidden_states = model_output.last_hidden_state
        hidden_attn = F.softmax(self.attn_proj(hidden_states).squeeze(2).masked_fill_(attn_mask==0, float('-inf')) / math.sqrt(self.h_dim), dim=1)
        constraint_emb = torch.einsum('btd,bt->bd', hidden_states, hidden_attn)
        # prediction_logit = self.prediction_head(F.relu(self.ff(constraint_emb)))
        prediction_logits = {}
        for k, head in self.prediction_heads.items():
            prediction_logits[k] = head(F.relu(self.ffs[k](constraint_emb)))
        return prediction_logits, model_output
        #return prediction_logit, model_output
    
    def get_loss(self, scenes: List[InstructionScene]):
        #tokens, attn_mask = self._tokenize_batch([scene.data for scene in scenes], scenes)
        # print('self.max_length: ', self.max_length)
        # print(scenes)
        tokens, attn_mask = self._tokenize_batch(scenes)
        # print(tokens.shape, attn_mask.shape)
        prediction_logits, model_output = self(tokens, attn_mask, output_attentions=True)
        transformer_logs = get_transformer_logs(model_output.attentions, self.model, attn_mask)

        #labels = self._tokenize_customer_scenarios_batch([scene.customer_scenario for scene in scenes])
        labels = [float(scene.label) for scene in scenes]
        torch_labels = torch.tensor(labels).to(torch.int64).to(self.device) 
        torch_labels_onehot = F.one_hot(torch_labels, num_classes=len(instruction_def)).to(torch.float32).to(self.device)
        # print(torch_labels_onthot.shape)
        # print(prediction_logits.shape)
        # print(torch_labels_onthot)
        # print(prediction_logits)
        losses = {}
        accuracies = {}
        loss = 0.0
        n = len(scenes)
        exact_match_accuracy = torch.ones((n,)).to(self.device)
        for k in prediction_logits.keys():  # instructionPredictorRoberta에선 prediction_logits.key가 1개뿐.
            losses[k] = F.cross_entropy(prediction_logits[k], torch_labels_onehot)
            accuracies[k] = (torch_labels == torch.argmax(prediction_logits[k], dim=-1)).float().mean()
            # print(loss)
            # print('=================')
            # print(torch.argmax(prediction_logit, dim=-1))
            # print(torch_labels)
            # print(torch_labels == torch.argmax(prediction_logit, dim=-1))
            # accuracy = (torch_labels == torch.argmax(prediction_logit, dim=-1)).float().mean()
            exact_match_accuracy *= (torch_labels == torch.argmax(prediction_logits[k], dim=-1)).float()
            loss += losses[k]

        logs = {**{k+'_loss': (v.item(), n) for k, v in losses.items()}, **{k+'_acc': (v.item(), n) for k, v in accuracies.items()}}
        logs['exact_match_acc'] = (exact_match_accuracy.mean().item(), n)
        logs['loss'] = (loss.item(), n)
        # logs = {}
        # logs['loss'] = loss
        # logs['accuracy'] = accuracy
        logs['transformer'] = transformer_logs

        return loss, logs, []
        # return 0, {}, []
    
    # def top_k_constraint_sets(self, scenes: List[Scene], 
    #                           dialogue_events_list: List[List[Event]], 
    #                           k: int):
    #     tokens, attn_mask = self._tokenize_batch(dialogue_events_list, scenes)
    #     prediction_logits, _ = self(tokens, attn_mask)
    #     constraint_probs = unstack_dicts({k: F.softmax(v, dim=-1).detach().cpu().tolist() for k, v in prediction_logits.items()})
    #     constraints = []
    #     for constraint_prob in constraint_probs:
    #         constraints.append(list(top_k_constraints(constraint_prob, self.discrete_features.idx2val, k_limit=k)))
    #     return constraints

class InstructionPredictorRobertaEvaluator(Evaluator):
    def __init__(self) -> None:
        super(InstructionPredictorRobertaEvaluator, self).__init__()
        # self.k = k
    
    def evaluate(self, model: InstructionPredictorRoberta, scenes: List[InstructionScene]) -> Optional[Dict[str, Any]]:
        _, logs, _ = model.get_loss(scenes)
        
        
        
        # reward_function = ConstraintRewardFunction(model)
        # reward_stats = PrecisionRecallAcc([0, 1])

        # predicted_rewards = reward_function.get_reward(scenes, [scene.events for scene in scenes], k=self.k)
        # true_rewards = [scene.events[-1].em_reward()['reward'] for scene in scenes]
        # assert len(predicted_rewards) == len(true_rewards)
        # for i in range(len(true_rewards)):
        #     reward_stats.add_item(predicted_rewards[i], true_rewards[i], predicted_rewards[i] == true_rewards[i])
        # logs = {'reward': reward_stats.return_summary()}
        return logs

