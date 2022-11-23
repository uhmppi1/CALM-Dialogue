from tqdm.auto import tqdm
import json
import argparse
from datetime import datetime
import os
from utils.instructions import instruction_def


def make_samples(data_dir, out_path):
    # list to store files
    # res = []
    with open(out_path, 'w') as f_out:
        # Iterate directory
        for path in os.listdir(data_dir):
            # check if current path is a file
            if os.path.isfile(os.path.join(data_dir, path)):
                print('----%s----' % os.path.join(data_dir, path))
                instructions_from_file(os.path.join(data_dir, path), f_out)
                # res.append(os.path.join(data_dir, path))
        # print(res)


def instructions_from_file(file_path, f_out):
    with open(file_path) as f_data:
        line_data = f_data.read()
        example = json.loads(line_data)
        prev_line = None

        instructions_lines = []
        label = 0
        for line in example['events']:
            # print('>>>'+str(line))
            if line['agent'] == 'Instructor':
                if len(instructions_lines) <= 0:
                    instructions_lines.append(prev_line)
                    if line['data'] in instruction_def:
                        label = instruction_def[line['data']]
                    else:
                        label = -1  # undefined instruction : 이 경우에는 데이터에 반영하지 않는다. 
                    #instructions_lines.append(line)
                #else:
                    #이 경우라면 check table 뒤에 data가 따라온 경우라고 생각된다. data는 생략해야!
            elif line['agent'] == 'Data':
                continue   # Data event는 고려하지 않고 생략!
                # if len(instructions_lines) <= 0:
                #     instructions_lines.append(prev_line)
                #     label = instruction_def['Data Result']
                    #instructions_lines.append(line)
                #else:
                    #이 경우라면 check table 뒤에 data가 따라온 경우라고 생각된다. data는 생략해야!
            else:
                if len(instructions_lines) <= 0:
                    prev_line = line
                else:
                    instructions_lines.append(line)
                    if len(instructions_lines) >= 3:
                        if label >= 0:
                            print(json.dumps({"data":instructions_lines, "label":label}))
                            f_out.write(json.dumps({"data":instructions_lines, "label":label})+'\n')
                        instructions_lines.clear()
                        prev_line = line


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