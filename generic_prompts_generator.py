# Read with Pycocotools and display the image and a segmentation mask with skimage
import argparse
import copy
import json
import tqdm

import numpy as np
import torch
from pycocotools.coco import COCO
import os
from PIL import Image
from diffusers import AutoPipelineForInpainting
parser = argparse.ArgumentParser(description='Generic Prompts Dream-Box generator')
parser.add_argument('--images_dir', type=str, default='data/voc/JPEGImages', help='path to VOC dataset')
parser.add_argument('--ann_file', type=str, default='data/voc/voc0712_train_all.json', help='path to VOC annotation file')
parser.add_argument('--output_dir', type=str, default='output', help='path to output directory')
parser.add_argument('--num_images', type=int, default=10, help='number of images to generate')
args = parser.parse_args()

# Define the path to the annotation file corresponding to the dataset
images_dir = args.images_dir
ann_file = args.ann_file
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

pipeline = AutoPipelineForInpainting.from_pretrained("stabilityai/stable-diffusion-2-inpainting",
                                                     torch_dtype=torch.float16, variant="fp16").to("cuda")

pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()
# pipeline.to("cuda")

# voc_classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
#                'airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train',
#                'bottle', 'chair', 'dining table', 'potted plant', 'couch', 'tv']
# Initialize the COCO api for instance annotations
coco = COCO(ann_file)
ood_coco = copy.deepcopy(coco.dataset)
ood_coco['images'] = []
ood_coco['annotations'] = []

# Get a random image with a random object from voc_classes
catIds = coco.getCatIds()
annIds = coco.getAnnIds(catIds=catIds)
imgIds = coco.getImgIds()
# imgIds = coco.getImgIds()
prompts = [
    "A {} that defies the laws of physics, floating in mid-air with strange edges.",
    "A mechanical {} with organic, plant-like growths intertwining through its structure.",
    "A transparent {} that appears to be both solid and liquid at the same time, emitting soft light.",
    "A {} that changes its shape continuously, with shifting low contrast colors and textures.",
    "A futuristic {} that blends digital and physical elements, glowing with an otherworldly light.",
    "A {} with an impossible texture, smooth like liquid but solid like metal, floating in space.",
    "A {} that merges two unrelated materials, seamlessly integrating them in an abstract form.",
    "A floating {} with intricate geometric patterns constantly changing on its surface.",
    "A {} made of plastic that constantly reconfigures itself into different shapes.",
    "A {} that appears to be in multiple states at once, existing in two places simultaneously.",
    "A strange {} that casts light in ugly but usual colors, transforming its appearance as it moves.",
    "A {} that is both solid and ethereal, with strange veins of energy running through it.",
    "A {} suspended in time, frozen in mid-motion, with particles of light trailing behind it.",
    "A mysterious floating {} with a strange core, surrounded by shifting shadows.",
    "A {} made of multiple contrasting materials that somehow coexist harmoniously.",
    "A complex {} with multiple layers, each one having a different texture and ugly color that shifts over time.",
    "A {} that seems to have multiple dimensions, existing in more than one space at once.",
    "A {} with a constantly rotating surface, covered in strange markings and symbols.",
    "A {} that looks like it's part of the natural world, but is made entirely of artificial materials.",
    "A smooth {} that seems to be melting and reforming simultaneously, surrounded by mist.",
]

ood_img_id = 1
for k in tqdm.tqdm(range(args.num_images)):
    while True:
        idx = np.random.randint(0, len(imgIds))
        valid_anns = []
        non_valid_anns = []
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[imgIds[idx]]))
        for ann in anns:
            if ann['area'] > 2000:
                valid_anns.append(ann)
            else:
                non_valid_anns.append(ann)
        if len(valid_anns) > 0:
            break
    # Sort anns by area, from larger to smaller
    valid_anns = sorted(valid_anns, key=lambda x: x['area'], reverse=True)
    img = coco.loadImgs([anns[0]['image_id']])[0]

    out_ood_img_anns = copy.deepcopy(valid_anns)
    out_in_img_anns = copy.deepcopy(non_valid_anns)
    for img_ann in out_ood_img_anns:
        img_ann['image_id'] = str(ood_img_id)
        img_ann['ood'] = 1
        ood_coco['annotations'].append(img_ann)
    for img_ann in out_in_img_anns:
        img_ann['image_id'] = str(ood_img_id)
        img_ann['ood'] = 0
        ood_coco['annotations'].append(img_ann)

    new_img_name = f'{img["file_name"][:-4]}_ood_{ood_img_id}.png'

    ood_img = copy.deepcopy(img)
    ood_img['id'] = str(ood_img_id)
    ood_img['file_name'] = new_img_name
    ood_coco['images'].append(ood_img)

    # Load the image
    I = Image.open(os.path.join(images_dir, img['file_name']))

    # Load the annotation
    # annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    # anns = coco.loadAnns(annIds)

    # Display the image
    # plt.figure()
    # plt.imshow(I)
    # plt.axis('off')
    for ann in valid_anns:
        # Get category name
        cat = coco.loadCats(ann['category_id'])[0]
        # mask = coco.annToMask(ann)
        bbox = ann['bbox']
        bbox_mask = np.zeros_like(I)
        bbox_mask[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])] = 255

        bbox_mask_image = Image.fromarray(bbox_mask.astype(np.uint8)).convert('1')
        prompt = prompts[k % len(prompts)]
        prompt = prompt.format(cat['name'])
        negative_prompt = "realistic proportions, too bright, well-defined shapes, sharp focus, polished details, unusual colors"

        I = pipeline(prompt=prompt, image=I, mask_image=bbox_mask_image, negative_prompt=negative_prompt,
                     guidance_scale=20.).images[0]

        # Resize
        I = I.resize(I.size)

    # Save image
    I.save(os.path.join(args.output_dir, 'images', new_img_name))
    ood_img_id += 1


# Save outliers in a json file
with open(os.path.join(args.output_dir, 'ood_voc0712_train_all.json'), 'w') as f:
    json.dump(ood_coco, f)
