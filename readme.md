File model detect (bỏ vào thư mục model/detect/): https://drive.google.com/file/d/1c1jEyfo4OhFJopOleIpFMRbnpHEbdmPK/view?usp=sharing

## 0. Introduction

Source code used for Zalo AI Challenge track Traffic sign detection.
Wrote by Doanh B C.

## 1. Installation

Install libraries:
`
pip install -r requirements.txt
`

Build source:
`
python -m pip install -e .
`

## 2. Training

run script train.py for training

```
python train.py --train_dir <folder train> \
		--json_dir <folder annotation> \
		--iter <iteration> \
		--config <file config. default: Misc/scratch_mask_rcnn_R_50_FPN_9x_gn.yaml'> \
		--batch <batch size> \
		--lr <learning rate>
```

## 3 Export result via json:

```
bash predict.sh
```

or

```
python detect.py
python submission.py
```

## 4 Visualize:

```
python visualize.py
```

## 5 Calculate map:

```
python test_map.py
```
