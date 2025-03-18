#!/bin/bash

# Activate environment
source venv_quant_course_v39/bin/activate

# Enable RISE before notebooks start
jupyter-nbextension enable rise --py --sys-prefix

# Start Jupyter Notebook
jupyter notebook
