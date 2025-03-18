#!/bin/bash

# Activate the virtual environment
source ./venv_quant_course_v39/bin/activate

# Run the Python script passed as an argument
python "$1"

# Deactivate the virtual environment
deactivate
