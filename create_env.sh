#!/bin/bash

pyv=$(python3 -c 'import sys; print(sys.version_info[1])')
echo $pyv

if [[ "$pyv" -eq "9" ]]
then 
    echo "Valid version"
    python3 -m venv --clear venv_quant_course_v39
    source ./venv_quant_course_v39/bin/activate #lehet a Scriptben van bizonyos distrokon
    python3 -m pip install --upgrade pip
    pip install -r QuantCourseBP/requirements.txt #ha más útvonalon van a txt azt adjuk meg
    ipython kernel install --user --nacd me=venv_quant_course_v39
    jupyter-nbextension install rise --py --sys-prefix
    deactivate
else
    echo "Invalid python3 version"
fi

