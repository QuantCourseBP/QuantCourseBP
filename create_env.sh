#!/bin/bash

# Create virtual environment folder
python -m venv --clear venv_quant_course_v39

# Activate empty environment
source venv_quant_course_v39/bin/activate

# Install required Python packages
pip install -r requirements.txt
python -m pip install --upgrade pip

# Install Jupyter kernel for virtual environment
ipython kernel install --user --name=venv_quant_course_v39

# Install RISE for notebook presentation mode
jupyter-nbextension install rise --py --sys-prefix

# Deactivate virtual environment
deactivate
