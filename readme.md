# SYDE 675 Project
#### Ruoxin Li (20895556)
#### Wei Hao (20942206)

## Required libraries
- tensorflow
- numpy
- opencv-python

## Test
- Copy one of the 4 trained weights to the root directory of this project.
- Run test.py
- Results will output to "results" directory

## Train
- Download training dataset from my [Google Drive](https://drive.google.com/file/d/14eTkC-0xpQUPxVIyftcGq1sbRYHKaX3v/view?usp=sharing).
- Unzip at the root directory
- Change line 8 in train.py
  - "train-set": original DIV2K dataset
  - "train-set-2": bicubic pre-processed
  - "train-set-3": pre-processed based on model prediction
- Run train.py
