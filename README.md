# Dream-Box
Implementation of [**Dream-Box: Object-wise Outlier Generation for Out-of-Distribution Detection**](https://kostadinovshalon.github.io/dream-box/) by [Brian Isaac-Medina](https://kostadinovshalon.github.io/) and [Toby Breckon](https://breckon.org/). 
Dream-box is a technique that allows the synthesis of outlier objects in the pixel space that can be used as auxiliary data for object-wise OOD detection. We propose two
mechanisms for the synthesis of these outliers, namely using **Generic Prompts** or **Modified class embeddings**. The former uses a set of class-wise generic prompts to create
outlier objects, while the latter modifies the class-name embedding to generate outliers. More details can be found in the [paper](https://arxiv.org/abs/2504.18746).

## Datasets

This work uses the PASCAL VOC dataset as in-distribution and tests on the MS-COCO dataset as out-of-distribution, removing the classes that are present in the PASCAL VOC dataset.
Following the VOS paper, you can download the dataset from [here](https://drive.google.com/file/d/1n9C4CiBURMSCZy2LStBQTzR17rD_a67e/view?usp=sharing). The dataset should have the following structure:

```
VOC_DATASET_ROOT/
    JPEGImages/
    voc0712_train_all.json
    val_coco_format.json
```

To test OOD performance, please download the MS-COCO dataset and use the annotations file without the overlapping classes with PASCAL VOC, available [here](instances_val2017_ood_rm_overlap.json).

## Requirements

To perform image synthesis, please install [pycocotools](https://pypi.org/project/pycocotools/) and the huggingface diffusers package, following [the official instructions](https://huggingface.co/docs/diffusers/en/installation?install=Python).

To perform OOD training and evaluation, please install MMDetection 3.x following [the official instructions](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation).

## Image Synthesis

We provide two scripts for image synthesis according to our two methods: `generic_prompts_generator.py` and `modified_embeddings_generator.py`. 
Usage is as follows:

```bash
python generic_prompts_generator.py \
    --images_dir VOC_IMAGES_DIR_ROOT [default: data/voc/JPEGImages] \
    --ann_file PATH_TO_VOC_ANNOTATION_FILE [default: data/voc/voc0712_train_all.json] \
    --output_dir OUTPUT_DIR [default: output] \
    --num_images NUM_IMAGES \
```
or

```bash
python modified_embeddings_generator.py \
    --images_dir VOC_IMAGES_DIR_ROOT [default: data/voc/JPEGImages] \
    --ann_file PATH_TO_VOC_ANNOTATION_FILE [default: data/voc/voc0712_train_all.json] \
    --output_dir OUTPUT_DIR [default: output] \
    --num_images NUM_IMAGES \
    --std STD [default: 2.5] \
```

Where `VOC_IMAGES_DIR_ROOT` is the path to the PASCAL VOC images directory, `PATH_TO_VOC_ANNOTATION_FILE` is the path to the PASCAL VOC annotation file, and `OUTPUT_DIR` is the directory where the generated images will be saved. The `NUM_IMAGES` argument specifies how many outlier images to generate, and `STD` is the standard deviation for the Gaussian noise added to the class embeddings in the modified embeddings method.
Both scripts will generate images in `<OUTPUT_DIR>/images` directory and an annotation file with these outliers will be saved in `<OUTPUT_DIR>/ood_voc0712_train_all.json`.

## OOD Training

TBA

## OOD Evaluation

OOD evaluation is performed using the `ood_metrics.py` script. The script needs the predictions file from in-distribution and OOD datasets in order to get the FPR and AUROC metrics. Its usage is as follows:

```bash
python ood_metrics.py \
    id_results PATH_TO_IN_DIST_PREDICTIONS_FILE \
    ood_results PATH_TO_OOD_PREDICTIONS_FILE \
    --det-score-threshold SCORE [default: 0.5] \
```

Where `PATH_TO_IN_DIST_PREDICTIONS_FILE` is the path to the in-distribution predictions file, `PATH_TO_OOD_PREDICTIONS_FILE` is the path to the OOD predictions file, and `SCORE` is the detection score threshold to use for the evaluation. The script will output the FPR@95, AUROC and AUPR metrics for the OOD detection task.

## Citation
If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{isaac-medina2025dream,
  title={Dream-Box: Object-wise Outlier Generation for Out-of-Distribution Detection},
  author={Isaac-Medina, Brian and Breckon, Toby},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Worksohps (CVPRW)},
  year={2025}
}
```
