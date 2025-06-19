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

parser = argparse.ArgumentParser(description='Modified Prompts Embedding Dream-Box generator')
parser.add_argument('--images_dir', type=str, default='data/voc/JPEGImages', help='path to VOC dataset')
parser.add_argument('--ann_file', type=str, default='data/voc/voc0712_train_all.json', help='path to VOC annotation file')
parser.add_argument('--output_dir', type=str, default='output', help='path to output directory')
parser.add_argument('--num_images', type=int, default=5000, help='number of images to generate')
parser.add_argument('--std', type=float, default=2.5, help='std for gaussian noise')
args = parser.parse_args()

# Define the path to the annotation file corresponding to the dataset
images_dir = args.images_dir
ann_file = args.ann_file
std = args.std

# Create output directories
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

pipeline = AutoPipelineForInpainting.from_pretrained("stabilityai/stable-diffusion-2-inpainting",
                                                     torch_dtype=torch.float16, variant="fp16").to("cuda")


pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

coco = COCO(ann_file)
ood_coco = copy.deepcopy(coco.dataset)
ood_coco['images'] = []
ood_coco['annotations'] = []

# Get a random image with a random object from voc_classes
catIds = coco.getCatIds()
annIds = coco.getAnnIds(catIds=catIds)
imgIds = coco.getImgIds()
# imgIds = coco.getImgIds()
with torch.no_grad():
    class_names = [cat['name'] for cat in coco.loadCats(catIds)]
    class_embeddings, _ = pipeline.encode_prompt(class_names, torch.device('cuda:0'), 1, True, None, None, None, None)
    class_embeddings = class_embeddings[:, 1]
    class_emb_dict = {n: emb for n, emb in zip(class_names, class_embeddings)}
    prompt_embeds = {}
    negative_prompt_embeds = {}
    for class_name, emb in class_emb_dict.items():
        pe, npe = pipeline.encode_prompt(
            f"A {class_name}, high quality, photo-realistic",  # prompt
            torch.device('cuda:0'),
            1,
            True,
            None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )
        prompt_embeds[class_name] = pe
        negative_prompt_embeds[class_name] = npe

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
            bbox_mask[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])] = 255

            bbox_mask_image = Image.fromarray(bbox_mask.astype(np.uint8)).convert('1')

            negative_prompt = "unusual colors"

            ee = class_emb_dict[cat['name']]
            prompt_embedding = prompt_embeds[cat['name']]
            negative_prompt_embedding = negative_prompt_embeds[cat['name']]

            # Use the cat emb as a mean and sample from a multivariate gaussian with circular covariance of 1
            # to get the new prompt embedding
            new_prompt_emb = ee + torch.randn_like(ee) * std
            prompt_embedding[0, 2] = new_prompt_emb

            I = pipeline(image=I, mask_image=bbox_mask_image, prompt_embeds=prompt_embedding,
                         negative_prompt_embeds=negative_prompt_embedding,
                         guidance_scale=12.).images[0]

            # Resize
            I = I.resize(I.size)

        # Save image
        I.save(os.path.join(args.output_dir, 'images', new_img_name))
        ood_img_id += 1


# Save outliers in a json file
with open(os.path.join(args.output_dir, 'ood_voc0712_train_all.json'), 'w') as f:
    json.dump(ood_coco, f)
