#!/bin/bash

pyv=$(python3 -c 'import sys; print(sys.version_info[1])')
echo $pyv

if [[ "$pyv" -eq "9" ]]
then 
    echo "Valid version"
    python3 -m venv --clear venv_quant_course_v39
    if [[ -f ./venv_quant_course_v39/bin/activate ]]; then 
        source ./venv_quant_course_v39/bin/activate #lehet a Scriptben van bizonyos distrokon
    else if [[ -f ./venv_quant_course_v39/Scripts/activate ]]; then
        source ./venv_quant_course_v39/Scripts/activate
    else 
        echo "activate Script not found"
        return 1
    fi
    fi
    python3 -m pip install --upgrade pip
    pip install -r requirements.txt #ha más útvonalon van a txt azt adjuk meg
    ipython kernel install --user --nacd me=venv_quant_course_v39
    jupyter-nbextension install rise --py --sys-prefix
    deactivate
else
    echo "Invalid python3 version"
fi

