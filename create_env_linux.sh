#!/usr/bin/env bash

# Creating python3.12 virtual environment
python3.12 -m venv venv_quant_course_v312
# Activate empty environment
source ./venv_quant_course_v312/bin/activate
# Installing required packages
pip install -r requirements.txt
python3 -m pip install --upgrade pip
# Install jupyter kernel:
python3 -m ipykernel install --user --name venv_quant_course_v312 --display-name "Quant Course (Python 3.12)"
deactivate
