# ReAlignHDR: Region-Constrained Alignment for Ghost-Free HDR Imaging

ReAlignHDR is a mask-guided HDR reconstruction model built on the AFUNet codebase.

## Environment setup

```sh
conda create -n realignhdr python=3.9
conda activate realignhdr
pip install torch torchvision
pip install numpy opencv-python scikit-image imageio einops timm tqdm
```

## Data preparation

The datasets used in this project are available from:

- [Kalantari's dataset](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/)
- [Tel's dataset](https://drive.google.com/drive/folders/1CtvUxgFRkS56do_Hea2QC7ztzglGfrlB)
- [Hu's dataset](https://github.com/nadir-zeeshan/sensor-realistic-synthetic-data)

Example structure:

```
data_root
└── Kal
    ├── Training
    │   ├── 001
    │   │   ├── 262A0898.tif
    │   │   ├── 262A0899.tif
    │   │   ├── 262A0900.tif
    │   │   ├── 262A0898.npy
    │   │   ├── 262A0899.npy
    │   │   ├── 262A0900.npy
    │   │   ├── exposure.txt
    │   │   ├── HDRImg.hdr
    │   ├── 002
    │   └── ...
    └── Test
        └── Test-set
            ├── 001
            │   ├── 262A2615.tif
            │   ├── 262A2616.tif
            │   ├── 262A2617.tif
            │   ├── 262A2615.npy
            │   ├── 262A2616.npy
            │   ├── 262A2617.npy
            │   ├── exposure.txt
            │   ├── HDRImg.hdr
            └── ...
```

## Generating masks with MLLM and SAM3

Use an MLLM to identify moving foreground categories, then use SAM3 to generate binary foreground/background masks and save them as `.npy` files. For SAM3 installation and model details, please refer to [SAM3](https://github.com/facebookresearch/sam3).

### Mask generation pipeline

The mask-generation pipeline follows the paper workflow:

1. Use GPT-5.4 to identify object categories with clear relative motion across the three exposure frames.
2. Convert the model output into a comma-separated segmentation prompt string.
3. Run SAM3 with that prompt and save one binary `.npy` mask per image, together with optional `.png` previews.

The prompt-generation step requires access to a multimodal model endpoint and the corresponding credentials configured in the running environment.

Prepare the dataset with the project-side preprocessing script:

```sh
python tools/prepare_realignhdr_data.py
```

This script organizes raw HDR scenes into the directory layout used by `train.py`, `train_student.py`, and `test.py`. It can also copy existing masks or create placeholder masks during preprocessing.

Use the dataset-level entry script to generate masks for all scenes sequentially:

```sh
python tools/generate_dataset_masks.py
```

This is the main entry for mask generation. It traverses the scene directories under the dataset root, generates the MLLM motion prompt for each scene, and then runs SAM3 to export the corresponding binary masks.

The entry script calls the following project-side helper scripts:

`tools/generate_motion_prompt.py` reads the three exposure frames in one scene and writes a normalized comma-separated prompt for moving foreground categories.

`tools/generate_sam3_masks.py` runs SAM3 from the generated prompt, merges detected instances into binary foreground masks, and saves `.npy` masks for training and evaluation.

The helper scripts under `sam3/` provide lower-level single-scene prompt and mask utilities for direct SAM3-side debugging.

## Running the model

### Teacher Training

```sh
python train.py
```

### Student Training

```sh
python train_student.py
```

### Testing

`test.py` loads a checkpoint and runs evaluation. It supports both mask-guided testing and mask-free student testing.

```sh
python test.py
```

## Acknowledgments

This project is inspired by [AFUNet](https://github.com/eezkni/AFUNet). We also thank the [SAM3](https://github.com/facebookresearch/sam3) project for providing the segmentation backbone used to generate masks.
