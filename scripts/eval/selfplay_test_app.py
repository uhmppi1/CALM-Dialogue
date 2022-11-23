from load_objs import load_item
import json
from ad.airdialogue import Scene
from bots.base_bot import BaseBot
from ad.ad_types import Message
from utils.misc import convert_path
import sys
import os

cfg = {'customer_bot': {
    'policy': {
        'lm': {
            'name': 'customer_gpt2_LM', 
            'gpt2_type': 'gpt2', 
            'max_length': None, 
            'checkpoint_path': 'outputs/customer_bot2/model.pkl', 
            'strict_load': True}, 
        'name': 'gpt2_lm_policy', 
        'kind': 'sample'}, 
    'name': 'basic_policy_bot', 
    'generation_kwargs': 
        {'n': 1}
    }, 
    'agent_bot': {
        'policy': {
            'lm': {
                'discrete_features': {
                    'name': 'discrete_features', 
                    'path_to_discrete': 'data/processed_ad/discrete_features.pkl', 
                    'key': 'agent', 
                    'strict': True}, 
                        'name': 'table_gpt2_LM', 
                        'gpt2_type': 'gpt2', 
                        'max_length': None, 
                        'checkpoint_path': 'outputs/instruct_agent_test/model.pkl', 
                        'attn_prior': 'geometric', 
                        'geometric_rate': 0.9, 
                        'cond_reward_key': 'conditioned_reward', 
                        'strict_load': True}, 
                    'name': 'gpt2_lm_policy', 
                    'kind': 'sample'}, 
                'name': 'full_policy_bot', 
                'generation_kwargs': {
                    'n': 1}
                }, 
                'dataset': {
                    'data': {
                        'name': 'airdialogue', 
                        'filepath': 'data/processed_ad/instruction_filtered_small.json', 
                        'limit': 1000, 
                        'heauristic_filter': False, 
                        'filter_with_goal': False}, 
                    'name': 'basic_dataset', 
                    'cond_reward_key': 'conditioned_reward', 
                    'cond_reward': 1.0}, 
                'selfplay': {
                    'load_outputs_file': None, 
                    'outputs_file': 'outputs/selfplay_outputs/selfplay_test1.json', 
                    'max_turns': 40, 
                    'verbose': True}
                }

filepath = convert_path('data/processed_ad/instruction_filtered_small.json')
scenes = []
with open(filepath, "r") as f:
    for i, line in enumerate(f):
        item = Scene.from_json(json.loads(line.strip()))
        scenes.append(item)

scene = scenes[0]
scenes[0].data = {'conditioned_reward' : 0.0}

customer_bot = load_item(cfg['customer_bot'], 'cuda')
customer_bot.eval()
agent_bot = load_item(cfg['agent_bot'], 'cuda')
agent_bot.eval()

def selfplay(agent1: BaseBot, agent2: BaseBot, scene: Scene):
    curr_event = None
    turn = 0

    # def vname(bot):
    #     vnames = [name for name in globals() if globals()[name] is bot]
    #     print(vnames)
    
    while (curr_event is None or not curr_event.is_final_action()) and turn < 50:

        if curr_event is None:
            curr_event = agent1.respond(curr_event, scene)  

        elif str(curr_event.agent) == 'Agent':        
            curr_event = agent1.respond(curr_event, scene)

        elif str(curr_event.agent) == 'Customer':        
            curr_event = agent2.respond(curr_event, scene)        
        elif str(curr_event.agent) == 'Instructor':        
            temp_event = curr_event 


            a = str(input('Instructor : '))
            
            temp_event.event = Message(str(a))
            curr_event = agent2.respond(temp_event, scene)

            

        elif str(curr_event.agent) == 'Data':        
            curr_event = agent2.respond(curr_event, scene)

        print(f'{str(curr_event.agent)}: {str(curr_event.event)}')
        turn += 1
        
    reward = curr_event.em_reward()
    print(reward)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

while True:
    selfplay(customer_bot, agent_bot, scene)