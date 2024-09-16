# Quantitative Evaluations: 2D Pose Estimation
This guide outlines the process to evaluate Sapiens-Pose checkpoints on two datasets.\
- [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody): 133 keypoints (17 kps body, 6 kps feet, 68 kps face, 42 kps hands).
- [COCO](https://cocodataset.org/#home): 17 keypoints

## 📂 1. Data Preparation
Set `$DATA_ROOT` as your training data root directory.\
Download the `val2017` images and 17 kps annotations from [COCO](https://cocodataset.org/#home). Download the 133 kps annotations from [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody). Unzip the images and annotations as subfolders to ```$DATA_ROOT```.\
 Additionally, download the bounding-box detection on the `val2017` set from [COCO_val2017_detections_AP_H_70_person.json](https://huggingface.co/noahcao/sapiens-pose-coco/tree/main/sapiens_host/pose/person_detection_results) and place it under `$DATA_ROOT/person_detection_results`.

The data directory structure is as follows:

      $DATA_ROOT/
      │   └── val2017
      │   │   └── 000000000139.jpg
      │   │   └── 000000000285.jpg
      │   │   └── 000000000632.jpg
      │   └── annotations
      │   │   └── person_keypoints_train2017.json
      │   │   └── person_keypoints_val2017.json
      │   │   └── coco_wholebody_train_v1.0.json
      │   │   └── coco_wholebody_val_v1.0.json
      │   └── person_detection_results
      │   │   └── COCO_val2017_detections_AP_H_70_person.json


## ⚙️ 2. Configuration Update

Let `$DATASET` be either `coco-wholebody` for 133 kps or `coco` for 17 kps.\
Edit `$SAPIENS_ROOT/pose/configs/sapiens_pose/$DATASET/sapiens_1b-210e_$DATASET-1024x768.py`:

1. Update `val_dataloader.dataset.data_root` to your `$DATA_ROOT`. eg. ```data/coco```.
2. Update ```val_evaluator.ann_file``` to also point to validation annotation file under `$DATA_ROOT`.
3. Update `bbox_file` to point to the bounding box detection file under `$DATA_ROOT`.


## 🏋️ 3. Evaluation

The following guide is for Sapiens-1B. You can find other backbones to evaluate under [pose_configs_133](../../pose/configs/sapiens_pose/coco_wholebody/) and [pose_configs_17](../../pose/configs/sapiens_pose/coco/).\
The testing scripts are under: `$SAPIENS_ROOT/pose/scripts/test/$DATASET/sapiens_1b`\
Make sure you have activated the sapiens python conda environment.


### A. 🚀 Single-node Testing
Use `$SAPIENS_ROOT/pose/scripts/test/$DATASET/sapiens_1b/node.sh`.

Key variables:
- `CHECKPOINT`: Absolute path to your checkpoint
- `DEVICES`: GPU IDs (e.g., "0,1,2,3,4,5,6,7")
- `TEST_BATCH_SIZE_PER_GPU`: Default 32
- `OUTPUT_DIR`: Checkpoint and log directory
- `mode=multi-gpu`: Launch multi-gpu testing with multiple workers for dataloading.
- `mode=debug`: (Optional) To debug. Launched single gpu dry run, with single worker for dataloading. Supports interactive debugging with pdb/ipdb.

Launch:
```bash
cd $SAPIENS_ROOT/pose/scripts/test/$DATASET/sapiens_1b
./node.sh
  ```

### B. 🌐 Multi-node Testing (Slurm)

Use `$SAPIENS_ROOT/pose/scripts/test/$DATASET/sapiens_1b/slurm.sh`

Additional variables:
- `CONDA_ENV`: Path to conda environment
- `NUM_NODES`: Number of nodes (default 4, 8 GPUs per node)

Launch:
```bash
cd $SAPIENS_ROOT/pose/scripts/test/$DATASET/sapiens_1b
./slurm.sh
  ```

## 📈  4. Results
Sapiens achieve state-of-the-art results for keypoint estimation on both datasets. Below we compare them with existing methods.

### COCO-WholeBody - 133 Keypoints
|        Model        | Input Size | Body AP | Body AR | Feet AP | Feet AR | Face AP | Face AR | Hand AP | Hand AR | Whole AP | Whole AR | Config | Ckpt |
| :-----------------: | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :--------: | :--------: | :----: | :--------: |
| DeepPose | 384 × 288 | 44.4 | 56.8 | 36.8 | 53.7 | 49.3 | 66.3 | 23.5 | 41.0 | 33.5 | 48.4 | - | - |
| SimpleBaseline | 384 × 288 | 66.6 | 74.7 | 63.5 | 76.3 | 73.2 | 81.2 | 53.7 | 64.7 | 57.3 | 67.1 | - | - |
| HRNet | 384 × 288 | 70.1 | 77.3 | 58.6 | 69.2 | 72.7 | 78.3 | 51.6 | 60.4 | 58.6 | 67.4 | - | - |
| ZoomNAS | 384 × 288 | 74.0 | 80.7 | 61.7 | 71.8 | 88.9 | 93.0 | 62.5 | 74.0 | 65.4 | 74.4 | - | - |
| VitPose+-L | 256 × 192 | 75.3 | - | 77.1 | - | 63.0 | - | 54.2 | - | 60.6 | - | - | - |
| VitPose+-H | 256 × 192 | 75.9 | - | 77.9 | - | 63.6 | - | 54.7 | - | 61.2 | - | - | - |
| RTMPose-x | 384 × 288 | 71.4 | 78.4 | 69.2 | 81.0 | 88.8 | 92.2 | 59.0 | 68.5 | 65.3 | 73.3 | - | - |
| DWPose-m | 256 × 192 | 68.5 | 76.1 | 63.6 | 77.2 | 82.8 | 88.1 | 52.7 | 63.4 | 60.6 | 69.5 | - | - |
| DWPose-l | 384 × 288 | 72.2 | 78.9 | 70.4 | 81.7 | 88.7 | 92.1 | 62.1 | 71.0 | 66.5 | 74.3 | - | - |
|
| Sapiens-0.3B (Ours) | 1024 × 768 | 66.4 | 73.4 | 67.3 | 78.4 | 87.1 | 91.2 | 58.1 | 67.1 | 62.0 | 69.4 | [config](https://github.com/facebookresearch/sapiens/blob/main/pose/configs/sapiens_pose/coco_wholebody/sapiens_0.3b-210e_coco_wholebody-1024x768.py) | [ckpt](https://huggingface.co/noahcao/sapiens-pose-coco/blob/main/sapiens_host/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620.pth) |
| Sapiens-0.6B (Ours) | 1024 × 768 | 74.3 | 80.2 | 79.4 | 87.0 | 89.5 | 92.9 | 65.4 | 74.0 | 69.5 (+3.0) | 76.3 (+2.0) | [config](https://github.com/facebookresearch/sapiens/blob/main/pose/configs/sapiens_pose/coco_wholebody/sapiens_0.6b-210e_coco_wholebody-1024x768.py) | [ckpt](https://huggingface.co/noahcao/sapiens-pose-coco/blob/main/sapiens_host/pose/checkpoints/sapiens_0.6b/sapiens_0.6b_coco_wholebody_best_coco_wholebody_AP_695.pth) |
| Sapiens-1B (Ours) | 1024 × 768 | 77.4 | 82.9 | 83.0 | 89.8 | 90.7 | 93.6 | 69.2 | 77.1 | 72.7 (+6.2) | 79.2 (+4.9) | [config](https://github.com/facebookresearch/sapiens/blob/main/pose/configs/sapiens_pose/coco_wholebody/sapiens_1b-210e_coco_wholebody-1024x768.py) | [ckpt](https://huggingface.co/noahcao/sapiens-pose-coco/blob/main/sapiens_host/pose/checkpoints/sapiens_1b/sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727.pth) |
| Sapiens-2B (Ours) | 1024 × 768 | **79.2** | **84.6** | **84.1** | **90.9** | **91.2** | **93.8** | **70.4** | **78.1** | **74.4 (+7.9)** | **81.0 (+6.7)** | [config](https://github.com/facebookresearch/sapiens/blob/main/pose/configs/sapiens_pose/coco_wholebody/sapiens_2b-210e_coco_wholebody-1024x768.py) | [ckpt](https://huggingface.co/noahcao/sapiens-pose-coco/blob/main/sapiens_host/pose/checkpoints/sapiens_2b/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745.pth) |
