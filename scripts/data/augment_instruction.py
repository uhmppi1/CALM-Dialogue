from tqdm.auto import tqdm
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from load_objs import load_item
import argparse
from datetime import datetime
import os
from ad.airdialogue import InstructionScene
from utils.misc import convert_path
from utils.instructions import instruction_def, get_instruction_from_label


def make_aug_samples(data_dir, out_data_dir, model, limit=0):
    # list to store files
    #res = []
    data_dir = convert_path(data_dir)
    out_data_dir = convert_path(out_data_dir)

    # Iterate directory
    for idx, path in enumerate(os.listdir(data_dir)):
        # check if current path is a file
        if os.path.isfile(os.path.join(data_dir, path)):
            print('----%s----' % os.path.join(data_dir, path))
            print('----%s----' % path.split('.')[0])
            # print('----%s----' % os.path.basename(os.path.join(data_dir, path)))
            augment_instructions_to_file(data_dir, path, out_data_dir, model)
            #res.append(os.path.join(data_dir, path))
            # print('--------')
            #return # TODO : 1줄만 처리하려고 임시로 넣었음. 제거할것. 
        #print(res)
        if limit > 0 and idx >= limit:
            break


def augment_instructions_to_file(data_dir, path, out_data_dir, model):
    file_path = os.path.join(data_dir, path)
    aug_file_path = os.path.join(out_data_dir, path.split('.')[0]+'_aug.json')
    with open(file_path) as f_data:
        example = json.loads(f_data.read())

        prev_customer_line = None
        prev_agent_line = None
        make_augment = False
        scenes = []
        for line in example['events']:
            if make_augment:
                instructions_lines = [prev_customer_line, prev_agent_line, line]
                scene = InstructionScene(instructions_lines, None)
                scene.label = predict_instruction([scene], model)
                scenes.append(scene)
                make_augment = False
            if line['agent'] == 'Customer':
                prev_customer_line = line
            elif line['agent'] == 'Agent':
                prev_agent_line = line
                make_augment = True

        # 3, 8번 라벨(데이터조회 관련)에 대한 예외처리 필요.
        events = []
        #check_data = False
        cur_scene = None
        is_booking = (example['customer_scenario']['goal'] == 'book')
        instruction_event = {}

        for scene in scenes:
            cur_scene = scene
            events.append(cur_scene.data[0])
            instruction_event = {
                "agent": "Instructor",
                "data": get_instruction_from_label(scene.label),
                "action": "message"
            }
            events.append(instruction_event)
            if scene.label in (3, 8):
                data_event = make_dataresult_event(is_booking, example)
                events.append(data_event)
            events.append(cur_scene.data[1])
        
        events.append(cur_scene.data[2])    
        if not cur_scene.data[2]['agent'] == 'Submit':  
            events.append(example['events'][-1])  #Submit 이 빠지지 않도록 예외처리

        example['events'] = events

        with open(aug_file_path, 'w') as f_out:
            f_out.write(json.dumps(example, indent=2))

def make_dataresult_event(is_booking, example):
    instruction_event = {}
    final_event = example['events'][-1]
    print(final_event)
    if is_booking:
        if len(final_event['data']['flight']) > 0:  # kb에서 가져올 매치된 운항정보가 있는경우
            flight = final_event['data']['flight'][0]
            flight_idx = flight-1000
            instruction_event = {
                "agent": "Data",
                "data": example['agent_scenario']['kb'][flight_idx],
                "action": "retrieval"
            }
        else:
            instruction_event = {
                "agent": "Data",
                "data": {
                    "flight": []
                },
                "action": "retrieval"
            }
    else:
        instruction_event = {
            "agent": "Data",
            "data": final_event['data'],
            "action": "retrieval"
        }
    return instruction_event
        


def predict_instruction(scenes, model):

    result = model.predict(scenes)
    print(model.get_loss(scenes))
    #print(model.get_loss(scenes))

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('this is scenes: ',scenes)
    print('this is result: ',result)
    label = result.item()
    print(get_instruction_from_label(label))

    return label

@hydra.main(config_path="../../config", config_name="augment_instruction")
def main(cfg : DictConfig):
    print('instruction augmentation start..')
    cfg = OmegaConf.to_container(cfg)

    model = load_item(cfg['model'], 'cuda')
    data_dir = cfg['data_dir']
    out_data_dir = cfg['out_data_dir']
    limit = cfg['limit']

    make_aug_samples(data_dir, out_data_dir, model, limit)
    print('instruction augmentation completed...')

if __name__ == "__main__":
    main()