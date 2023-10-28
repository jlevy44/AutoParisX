#!/bin/bash
pip install torch==1.10.2  torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu102/
pip install git+https://github.com/facebookresearch/detectron2.git
pillow==9.5.0
pip install "dask[distributed]"
pip install cucim==23.8.0 cupy-cuda11x==12.2.0