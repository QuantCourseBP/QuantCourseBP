@echo off
:: create virutal env folder
python -m venv --clear venv_quant_course_v39
:: activate empty env
call .\venv_quant_course_v39\Scripts\activate
:: install required Python packages
pip install -r requirements.txt
python -m pip install --upgrade pip
:: install jupyter kernel for virtual env
ipython kernel install --user --name=venv_quant_course_v39
deactivate
