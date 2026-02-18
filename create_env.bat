@echo off
:: create virutal env folder
C:\Users\Samyfarha\AppData\Local\Programs\Python\Python39\python.exe -m venv --clear venv_quant_course_v39
:: activate empty env
call .\venv_quant_course_v39\Scripts\activate
:: install required Python packages
pip install -r requirements.txt
C:\Users\Samyfarha\AppData\Local\Programs\Python\Python39\python.exe -m pip install --upgrade pip
:: install jupyter kernel for virtual env
ipython kernel install --user --name=venv_quant_course_v39
:: install rise for notebook presentation mode
jupyter-nbextension install rise --py --sys-prefix
deactivate
