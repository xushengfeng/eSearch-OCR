#!/bin/sh

#  Script.sh
#  
#
#  Created by xsf on 2022/4/3.
#  

ifpath="/Users/xsf/Downloads/eSearch/ocr/ppocr/inference"
python3.7 /Users/xsf/Downloads/eSearch-service/py/ppocr/tools/infer/predict_system.py --image_dir="/Users/xsf/Pictures/eSearch-2022-04-03-22-25-41-7.png" --det_model_dir="${ifpath}/ch_PP-OCRv2_det_infer" --rec_model_dir="${ifpath}/ch_PP-OCRv2_rec_infer" --use_gpu=False --rec_char_dict_path="${ifpath}/../ppocr_keys_v1.txt"
