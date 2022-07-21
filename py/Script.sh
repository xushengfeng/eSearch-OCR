#!/bin/sh

#  Script.sh
#  
#
#  Created by xsf on 2022/4/3.
#  

ifpath="/Users/xsf/Downloads/eSearch/ocr/ppocr/inference"

python3 ppocr/tools/infer/predict_system.py --image_dir="../a.png" --det_model_dir="../assets/ch_PP-OCRv3_det_infer/"  --rec_model_dir="../assets/ch_PP-OCRv3_rec_infer/" --use_angle_cls=False --use_space_char=True --use_gpu=False --rec_char_dict_path="../assets/ppocr_keys_v1.txt"