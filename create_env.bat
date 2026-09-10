@echo off
:: create virutal env folder
py -3.12 -m venv --clear venv_quant_course_v312
:: activate empty env
call .\venv_quant_course_v312\Scripts\activate
:: install required Python packages
pip install -r requirements.txt
python -m pip install --upgrade pip
:: install jupyter kernel for virtual env
python -m ipykernel install --user --name venv_quant_course_v312 --display-name "Quant Course (Python 3.12)"
deactivate
