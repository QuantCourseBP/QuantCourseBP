@echo off
:: create virutal env folder
python -m venv --clear venv_quant_course_v312
:: activate empty env
call .\venv_quant_course_v312\Scripts\activate
:: install required Python packages
pip install -r requirements.txt
python -m pip install --upgrade pip
:: install jupyter kernel for virtual env
ipython kernel install --user --name=venv_quant_course_v312
:: install rise for notebook presentation mode
jupyter-nbextension install rise --py --sys-prefix
deactivate
