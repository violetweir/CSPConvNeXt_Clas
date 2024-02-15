import os
file = os.listdir('/tmp/code')
print(file)
os.system('rm /tmp/dataset/Light_ILSVRC2012.zip')
os.system('export CUDA_VISIBLE_DEVICES=0')
os.system('python3 /tmp/code/liming_convnext/tools/train.py \
    -c /tmp/code/liming_convnext/ppcls/configs/ImageNet/ConvNext/CSPConvNext_tiny.yaml \
    -o Global.device=gpu ')