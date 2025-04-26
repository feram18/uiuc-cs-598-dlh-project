### How to run

- Download the NIHCC ChestXray dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC
- Modify Reproduce_CXR-ML-GZSL.py's "data_root" to point to the images folder from the dataset after unarchiving each individual archive
- working-requirements.txt contains the frozen dependencies. python 3.8.18 is required due to the use of some dependencies.
  - Install pyenv by running curl https://pyenv.run | bash
  - pyenv install 3.8.18
  - pyenv global 3.8.18
  - cd /project/dir/you/chose/
  - pyenv local 3.8.18
  - create an env and install dependenies
    - python -m venv zsl-venv
    - source zsl-venv/bin/activate
    - pip install -r working-requirements.txt
    - 
