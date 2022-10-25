from tqdm.auto import tqdm
import json
import argparse
from datetime import datetime

def is_sample_valid(example, kb):
    return example['correct_sample']

def write_sample_file(data, file_path):
    with open(file_path, 'w') as f:
        f.write(json.dumps(data, indent=2))

def make_samples(data_path, kb_path, limit=None):
    keys = [
        'goal', 'name', 'max_price', 'max_connections', 'class',
        'airline_preference', 'departure_airport', 'departure_month',
        'departure_day', 'departure_time', 'return_airport', 'return_month',
        'return_day', 'return_time'
    ]
    line_idx = 0
    with open(data_path) as f_data, open(kb_path) as f_kb:
        for line_data, line_kb in tqdm(list(zip(f_data, f_kb))):
            line_idx += 1
            valid_example_path = 'data_analysis/{:08d}_original_valid.json'.format(line_idx)
            invalid_example_path = 'data_analysis/{:08d}_original_invalid.json'.format(line_idx)
            processed_example_path = 'data_analysis/{:08d}_proc01.json'.format(line_idx)
            event_example_path = 'data_analysis/{:08d}_proc02.json'.format(line_idx)
            kbsample_path = 'data_analysis/{:08d}_kb.json'.format(line_idx)
            example = json.loads(line_data)
            kb = json.loads(line_kb)

            if is_sample_valid(example, kb):
                write_sample_file(example, valid_example_path)
            else:
                write_sample_file(example, invalid_example_path)
            write_sample_file(kb, kbsample_path)

            new_dialogue = []
            for s in example['dialogue']:
                parts = s.split(': ')
                customer = parts[0]
                text = ': '.join(parts[1:])
                new_dialogue.append((customer, text))
            example['dialogue'] = new_dialogue

            intent = example['intent']
            for k in keys:
                if k not in intent:
                    intent[k] = 'None'
            assert set(intent.keys()) == set(keys), f'{intent}'
            example['customer_state'] = intent
            example['agent_state'] = kb
            example.pop('search_info', None)
            example.pop('timestamps', None)
            example.pop('correct_sample', None)
            example.pop('intent', None)
            example.pop('kb', None)

            write_sample_file(example, processed_example_path)
            event_data = factor_into_events(line_idx, example)
            write_sample_file(event_data, event_example_path)

            if limit is not None and line_idx >= limit:
                break


def factor_into_events(id, interaction):
    event_data = {}
    event_data['uuid'] = id
    event_data['customer_scenario_uuid'] = 2*id
    event_data['agent_scenario_uuid'] = 2*id+1
    event_data['customer_scenario'] = interaction['customer_state']
    event_data['agent_scenario'] = interaction['agent_state']
    event_data['expected_action'] = interaction['expected_action']
    event_data['events'] = []
    for agent, data in interaction['dialogue']:
        if agent == 'customer':
            agent = 'Customer'
        elif agent == 'agent':
            agent = 'Agent'
        else:
            raise NotImplementedError
        event_data['events'].append({
            'agent': agent,
            'data': data,
            'action': 'message',
        })
    event_data['events'].append({
        'agent': 'Submit',
        'data': interaction['action'],
        'action': interaction['action']['status']
    })
    return event_data

def main(data_file, kb_file, limit=None):
    print('sampling data...')
    make_samples(data_file, kb_file, limit)
    print('sampling completed...')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file")
    parser.add_argument("--kb_file")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.data_file, args.kb_file, limit=args.limit)