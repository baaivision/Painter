import os
import glob
import json
import tqdm
import argparse
import shutil


def get_args_parser():
    parser = argparse.ArgumentParser('get toy dataset for json', add_help=False)
    parser.add_argument('--json_path', type=str, help='path to json file', required=True)
    parser.add_argument('--data_s', type=str, default='datasets')
    parser.add_argument('--data_t', type=str, default='toy_datasets')
    parser.add_argument('--num_sample', type=int, help='number of samples', default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args_parser()

    dataset_full = json.load(open(args.json_path, 'r'))
    dataset = dataset_full[:args.num_sample]

    for data in dataset:
        image_path_src = os.path.join(args.data_s, data['image_path'])
        target_path_src = os.path.join(args.data_s, data['target_path'])
        print(image_path_src)

        image_path_tgt = os.path.join(args.data_t, data['image_path'])
        target_path_tgt = os.path.join(args.data_t, data['target_path'])

        if not os.path.exists(os.path.dirname(image_path_tgt)):
            os.makedirs(os.path.dirname(image_path_tgt))
        if not os.path.exists(os.path.dirname(target_path_tgt)):
            os.makedirs(os.path.dirname(target_path_tgt))

        shutil.copy(image_path_src, image_path_tgt)
        shutil.copy(target_path_src, target_path_tgt)
    
    save_path = args.json_path.replace('datasets', 'toy_datasets')
    json.dump(dataset, open(save_path, 'w'))
    print(save_path)