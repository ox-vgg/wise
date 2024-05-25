# WISE Installation

The hardware and software requirements for installing WISE are as follows.

- A modern computer with Ubuntu, Debian, or other similar OS
  - There may be installation issues on macOS and Windows. We recommend using a Linux environment (or WSL) for now
- Python version 3.10 (or higher)
- ffmpeg 4.4.2 (or higher)

To install WISE, we first download the WISE source code.

```
## 1. Get the code
git clone -b wise2 https://gitlab.com/vgg/wise/wise.git
cd wise
```

The WISE software depends on several python libraries and there are the
following three ways to install these software dependencies.

- Using the [Conda](https://docs.conda.io/en/latest/) or [Mamba](https://mamba.readthedocs.io/en/latest/index.html) dependency management tool
- Using Python's virtual environment [venv](https://docs.python.org/3/library/venv.html)
- Install WISE to use only CPU (e.g. on machines without a GPU or GPU with insufficient memory)

See the [User Guide](UserGuide.md) to test the visual search capability of the WISE
software tool.


## Option 1: Installation using conda / mamba

Using the conda tool, the WISE software dependencies can be installed as follows. Please note:

- We recommend you to use a recent version of conda (22 or greater) / mamba (1.4+). WISE might not work on lower versions of conda / mamba.
- If you are using WISE on non-Intel platforms, edit `environment.yml` to remove the reference to `mkl`

```
conda env create -f environment.yml
conda activate wise
pip install --no-deps msclap==1.3.3  # avoids installing conflicting version of torch
```

(For mamba, replace conda in the above command accordingly)

## Option 2: Installation using venv

If the conda tool based dependency management option is not suitable, the alternative
is to use Python's virtual environment [venv](https://docs.python.org/3/library/venv.html)
module for installing the dependencies as shown below.

```
python3 --version                  # must be >= 3.10
sudo apt install ffmpeg            # ffmpeg is required to load videos
python3 -m venv wise-dep/          # create virtual environment
source wise-dep/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps msclap==1.3.3
pip install -r torch-faiss-requirements.txt
```

## Option 3: Install WISE to use only CPU (i.e. no GPU)

WISE can be installed on a machine without a GPU. While the processing speed is slow,
all the functionality of the WISE software remains available.

```
git clone -b wise2 https://gitlab.com/vgg/wise/wise.git
cd wise

python3 --version                  # must be >= 3.10
sudo apt install ffmpeg            # ffmpeg is required to load videos
python3 -m venv wise-dep/          # create virtual environment
source wise-dep/bin/activate
python -m pip install --upgrade pip

# Note: the following command is the key to installing CPU only version of WISE
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install --no-deps msclap==1.3.3

pip install faiss-cpu
```