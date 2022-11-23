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


def make_aug_samples(data_dir, model):
    # list to store files
    res = []
    data_dir = convert_path(data_dir)

    # Iterate directory
    for path in os.listdir(data_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(data_dir, path)):
            print('----%s----' % os.path.join(data_dir, path))
            print('----%s----' % path.split('.')[0])
            # print('----%s----' % os.path.basename(os.path.join(data_dir, path)))
            augment_instructions_to_file(data_dir, path, model)
            res.append(os.path.join(data_dir, path))
            # print('--------')
            #return # TODO : 1줄만 처리하려고 임시로 넣었음. 제거할것. 
        #print(res)


def augment_instructions_to_file(data_dir, path, model):
    file_path = os.path.join(data_dir, path)
    aug_file_path = os.path.join(data_dir, path.split('.')[0]+'_aug.json')
    with open(file_path) as f_data:
        example = json.loads(f_data.read())

        # print(example)
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

        # 3, 8, 15번 라벨(데이터조회 관련)에 대한 예외처리 필요.
        events = []
        check_data = False
        cur_scene = None
        is_booking = (example['customer_scenario']['goal'] == 'book')
        instruction_event = {}
        label = -1
        for scene in scenes:
            if cur_scene: #두번째 이후의 scene일 때, 
                # 3 or 8 이후 연속되는 data instruction이 들어오지 않을 경우, check table 이후 바로 data result가 이어지는 데이터셋.
                if label in (3, 8) and scene.label not in (3, 8, 15):
                    instruction_event = make_dataresult_event(is_booking, example)
                    events.append(instruction_event)
                events.append(cur_scene.data[1])

            cur_scene = scene
            label = scene.label
            if label in (3, 8, 15):
                if check_data:
                    label = 15   # 15:Data로 취급!
                else:
                    check_data = True
                    if is_booking:  # 8:check flight table로 취급!
                        label = 8
                    else :
                        label = 3
            if label == 15:
                instruction_event = make_dataresult_event(is_booking, example)
            else:
                instruction_event = {
                    "agent": "Instructor",
                    "data": get_instruction_from_label(label),
                    "action": "message"
                }
            events.append(cur_scene.data[0])
            events.append(instruction_event)
        
        events.append(cur_scene.data[1])
        events.append(cur_scene.data[2])
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

    make_aug_samples(data_dir, model)
    print('instruction augmentation completed...')

if __name__ == "__main__":
    main()