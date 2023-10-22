from setuptools import setup
from setuptools.command.install import install
import subprocess
import os

with open("requirements.txt") as f:
      PACKAGES=f.read().splitlines()

with open('README.md','r', encoding='utf-8') as f:
      long_description = f.read()

setup(name='autoparis',
      version='0.1',
      description='Python package for Autoparis Workflow.',
      url='https://github.com/jlevy44/autoparisx',
      author='Joshua Levy',
      author_email='joshualevy44@berkeley.edu',
      license='MIT',
      scripts=[],
      entry_points={
            'console_scripts':['autoparis=autoparis.extract_predict:main',
                               'ap_npy2dzi=autoparis.dzi_writer:main']
      },
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['autoparis'],
      install_requires=PACKAGES,
      extras_require=dict(gpu="cucim==23.8.0 cupy-cuda11x==12.2.0".split()))
