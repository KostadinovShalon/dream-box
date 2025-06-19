import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument('--id_anns', type=str, default='data/voc/voc0712_train_all.json')
parser.add_argument('--ood_anns', type=str, default='data/voc/ood_voc0712_train_all.json')
args = parser.parse_args()

in_data = json.load(open(args.id_anns, 'rb'))
ood_data = json.load(open(args.ood_anns, 'rb'))

ood_anns = ood_data['annotations']
ood_imgs = ood_data['images']

for ood_img in ood_imgs:
    ood_img['file_name'] = 'outliers/' + ood_img['file_name']
    ood_img['id'] = 'ood_' + str(ood_img['id'])

in_data['images'].extend(ood_imgs)

max_in_id = max([ann['id'] for ann in in_data['annotations']])
for ood_ann in ood_anns:
    ood_ann['image_id'] = 'ood_' + str(ood_ann['image_id'])
    ood_ann['id'] = max_in_id + 1
    max_in_id += 1

in_data['annotations'].extend(ood_anns)
for ann in in_data['annotations']:
    if 'ood' not in ann['image_id']:
        ann['ood'] = 0

json.dump(in_data, open('data/voc/voc0712_train_all_with_ood.json', 'w'))