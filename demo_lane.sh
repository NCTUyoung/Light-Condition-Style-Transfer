python2 -u light_condiction_style_transfer.py CULane ERFNet \
                   --gpus 0 \
                   --resume trained/_erfnet_model_best.pth.tar \
                   --img_height 208 \
                   --img_width 976 \
                   -j 10 \
                   --mode 1 \
