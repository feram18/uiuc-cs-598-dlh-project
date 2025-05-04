# CXR-ML-GZSL Replication
> UIUC CS 598 Deep Learning for Healthcare

## Requirements
- Python 3.8.18 is required due to the use of some dependencies
- NIH ChestXray dataset
  - Download from https://nihcc.app.box.com/v/ChestXray-NIHCC

## Execution

- Modify [Reproduce_CXR-ML-GZSL.py](final-report/Reproduce_CXR-ML-GZSL.py)'s `data_root` to point to the images folder 
 from the dataset after unarchiving each individual archive
- `working-requirements.txt` contains the frozen dependencies
- Install `pyenv`
  ```shell
  curl https://pyenv.run | bash
  pyenv install 3.8.18
  pyenv global 3.8.18
  ```
- Switch to the repository directory
- Create a running environment
  ```shell
  pyenv local 3.8.18
  python -m venv zsl-venv
  source zsl-venv/bin/activate
  ```
- Install dependencies
  ```shell
  pip install -r working-requirements.txt
  ```
