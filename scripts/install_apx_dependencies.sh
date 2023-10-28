#!/bin/bash
pip install torch==1.10.2  torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu102/
pip install git+https://github.com/facebookresearch/detectron2.git
pillow==9.5.0