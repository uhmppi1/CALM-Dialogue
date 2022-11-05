from tqdm.auto import tqdm
import json
import argparse
from datetime import datetime
import os
from utils.instructions import instruction_def


def make_samples(data_dir, out_path):
    # list to store files
    res = []

    with open(out_path, 'w') as f_out:
        # Iterate directory
        for path in os.listdir(data_dir):
            # check if current path is a file
            if os.path.isfile(os.path.join(data_dir, path)):
                print('----%s----' % os.path.join(data_dir, path))
                instructions_from_file(os.path.join(data_dir, path), f_out)
                res.append(os.path.join(data_dir, path))
                # print('--------')
                # return  # TODO : 1줄만 처리하려고 임시로 넣었음. 제거할것. 
        # print(res)


def instructions_from_file(file_path, f_out):
    with open(file_path) as f_data:
        line_data = f_data.read()
        example = json.loads(line_data)
        prev_line = None

        instructions_lines = []
        for line in example['events']:
            # print('>>>'+str(line))
            if line['agent'] in ('Instructor', 'Data'):
                if len(instructions_lines) <= 0:
                    instructions_lines.append(prev_line)
                    instructions_lines.append(line)
                #else:
                    #이 경우라면 check table 뒤에 data가 따라온 경우라고 생각된다. data는 생략해야!
            else:
                if len(instructions_lines) <= 0:
                    prev_line = line
                else:
                    instructions_lines.append(line)
                    if len(instructions_lines) >= 4:
                        print({'data':instructions_lines})
                        f_out.write(str({'data':instructions_lines})+'\n')
                        instructions_lines.clear()
                        prev_line = line




# def is_sample_valid(example, kb):
#     return example['correct_sample']

# def write_sample_file(data, file_path):
#     with open(file_path, 'w') as f:
#         f.write(json.dumps(data, indent=2))

# def make_samples(data_path, kb_path, limit=None):
#     keys = [
#         'goal', 'name', 'max_price', 'max_connections', 'class',
#         'airline_preference', 'departure_airport', 'departure_month',
#         'departure_day', 'departure_time', 'return_airport', 'return_month',
#         'return_day', 'return_time'
#     ]
#     line_idx = 0
#     with open(data_path) as f_data, open(kb_path) as f_kb:
#         for line_data, line_kb in tqdm(list(zip(f_data, f_kb))):
#             line_idx += 1
#             valid_example_path = 'data_analysis/{:08d}_original_valid.json'.format(line_idx)
#             invalid_example_path = 'data_analysis/{:08d}_original_invalid.json'.format(line_idx)
#             processed_example_path = 'data_analysis/{:08d}_proc01.json'.format(line_idx)
#             event_example_path = 'data_analysis/{:08d}_proc02.json'.format(line_idx)
#             kbsample_path = 'data_analysis/{:08d}_kb.json'.format(line_idx)
#             example = json.loads(line_data)
#             kb = json.loads(line_kb)

#             if is_sample_valid(example, kb):
#                 write_sample_file(example, valid_example_path)
#             else:
#                 write_sample_file(example, invalid_example_path)
#             write_sample_file(kb, kbsample_path)

#             new_dialogue = []
#             for s in example['dialogue']:
#                 parts = s.split(': ')
#                 customer = parts[0]
#                 text = ': '.join(parts[1:])
#                 new_dialogue.append((customer, text))
#             example['dialogue'] = new_dialogue

#             intent = example['intent']
#             for k in keys:
#                 if k not in intent:
#                     intent[k] = 'None'
#             assert set(intent.keys()) == set(keys), f'{intent}'
#             example['customer_state'] = intent
#             example['agent_state'] = kb
#             example.pop('search_info', None)
#             example.pop('timestamps', None)
#             example.pop('correct_sample', None)
#             example.pop('intent', None)
#             example.pop('kb', None)

#             write_sample_file(example, processed_example_path)
#             event_data = factor_into_events(line_idx, example)
#             write_sample_file(event_data, event_example_path)

#             if limit is not None and line_idx >= limit:
#                 break


# def factor_into_events(id, interaction):
#     event_data = {}
#     event_data['uuid'] = id
#     event_data['customer_scenario_uuid'] = 2*id
#     event_data['agent_scenario_uuid'] = 2*id+1
#     event_data['customer_scenario'] = interaction['customer_state']
#     event_data['agent_scenario'] = interaction['agent_state']
#     event_data['expected_action'] = interaction['expected_action']
#     event_data['events'] = []
#     for agent, data in interaction['dialogue']:
#         if agent == 'customer':
#             agent = 'Customer'
#         elif agent == 'agent':
#             agent = 'Agent'
#         else:
#             raise NotImplementedError
#         event_data['events'].append({
#             'agent': agent,
#             'data': data,
#             'action': 'message',
#         })
#     event_data['events'].append({
#         'agent': 'Submit',
#         'data': interaction['action'],
#         'action': interaction['action']['status']
#     })
#     return event_data

def main(data_dir, out_file):
    print('instruction training data generation..')
    make_samples(data_dir, out_file)
    print('generation completed...')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--out_file")
    args = parser.parse_args()
    main(args.data_dir, args.out_file)