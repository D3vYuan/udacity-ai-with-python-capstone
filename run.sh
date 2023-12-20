#!/bin/bash

cd /home/workspace/ImageClassifier
python3 train.py flowers --epochs=1 --gpu
python3 train.py flowers --arch="vgg16_bn" --learning_rate=0.001 --hidden_units=4096 --epochs=1 --gpu
python3 train.py flowers --arch="vgg16_bn" --learning_rate=0.001 --hidden_units=4096 --epochs=1 --save_dir="/opt/models/checkpoint_mine_vgg16_bn" --gpu

# pink primrose
python3 predict.py /home/workspace/ImageClassifier/flowers/valid/1/image_06755.jpg /opt/checkpoint_vgg19_bn_cuda_20220906_1725.pth --gpu
python3 predict.py /home/workspace/ImageClassifier/flowers/valid/1/image_06755.jpg /opt/checkpoint_vgg19_bn_cuda_20220906_1817.pth --arch="vgg19_bn" --top_k=3 --category_names="flower_categories.json" --gpu
python3 predict.py /home/workspace/ImageClassifier/flowers/valid/1/image_06755.jpg /opt/checkpoint_vgg16_bn_cuda_20220906_1809.pth --arch="vgg16_bn" --top_k=3 --category_names="flower_categories.json" --gpu