#!/bin/bash
pip install torch==1.10.2  torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu102/
pip install git+https://github.com/facebookresearch/detectron2.git
pip install pillow==9.5.0
pip install "dask[distributed]"
pip install cucim==23.8.0 cupy-cuda11x==12.2.0
pip install git+https://github.com/jlevy44/PathPretrain/ --no-deps
# now can update torch to 2.0.1, use this command
# pip install https://github.com/MiroPsota/torch_packages_builder/releases/download/detectron2-0.6/detectron2-0.6%2Bpt2.0.1cu117-cp39-cp39-linux_x86_64.whl
