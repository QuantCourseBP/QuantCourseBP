
#!/bin/bash

# Létrehozzuk a virtuális környezetet Python 3.9-cel
/usr/bin/python3 -m venv --clear venv_quant_course_v39

# Aktiváljuk
source ./venv_quant_course_v39/bin/activate

# Frissítjük a pip-et
pip install --upgrade pip

# Telepítjük a csomagokat a requirements.txt alapján
pip install -r requirements.txt

# Jupyter kernel és RISE telepítése
pip install notebook ipykernel rise
python -m ipykernel install --user --name=venv_quant_course_v39
jupyter-nbextension install rise --py --sys-prefix

# Kilépés a virtuális környezetből
deactivate

echo "Virtuális környezet kész: venv_quant_course_v39"
