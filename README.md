# ML Security Project for ECE-GY 9163

## Group Member:
+ Xujian Quan (xq454)
+ Shaoyong Wang（sw4557）
+ Rui Ren (rr3089)
+ Bo Su (bs3957)

## Environment Requirements
+ Python3.6
+ Colab
+ tempfile
+ tensorflow
+ keras
+ sys
+ h5py
+ numpy
+ tensorflow_model_optimization
+ PIL

## Description:
Filename   |  Discription    
:-----:  | :-------: 
github | [CSAW-HackML-2020](https://github.com/csaw-hackml/CSAW-HackML-2020)
models | 4 original Badnet models + 4 pruned-trained Badnet models
test_input_png | 10 test input images
B1.ipynb | Pruning and training of sunglasses_bd model (output Pruned_B1.h5)
B2.ipynb | Pruning and training of multi_trigger model (output Pruned_B2.h5)
B3.ipynb | Pruning and training of anonymous_1 model (output Pruned_B3.h5)
B4.ipynb | Pruning and training of anonymous_2 model (output Pruned_B4.h5)
eval_sunglasses_bd.py | Repaired B1 model
eval_multi_trigger.py | Repaired B2 model
eval_anonymous_1.py | Repaired B3 model
eval_anonymous_2.py | Repaired B4 model
myEval.py | New eval script for dataset evaluation

## Evaluating Operation:
1. Evaluate valid, test or other label dataset with G1/G2/G3/G4 model by using `myEval.py`, just like the original `eval.py`: <br>
`python3 myEval.py <data directory> <model name: G1/G2/G3/G4>`  <br>
examples: <br>
`python3 myEval.py ./data/clean_validation_data.h5 G1` <br>
`python3 myEval.py ./data/clean_validation_data.h5 G2` <br>
`python3 myEval.py ./data/clean_test_data.h5 G3` <br>
`python3 myEval.py ./data/clean_test_data.h5 G4` <br>
2. Evaluate an image input and print the result in the range [0-1283]: <br>
`python3 <model file> <image path>` <br>
examples: <br>
`python3 eval_sunglasses_bd.py ./test_input_png/image_clean_0.png` <br>
`python3 eval_multi_trigger.py ./test_input_png/image_clean_1.png` <br>
`python3 eval_anonymous_1.py ./test_input_png/image_clean_2.png` <br>
`python3 eval_anonymous_2.py ./test_input_png/image_clean_3.png` <br>
