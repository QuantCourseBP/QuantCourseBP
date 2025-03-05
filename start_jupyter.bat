:: activate env
call .\venv_quant_course_v39\Scripts\activate
:: enable rise before notebooks started
jupyter-nbextension enable rise --py --sys-prefix
:: start jupyter
jupyter notebook
