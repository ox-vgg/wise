# WISE Installation

The hardware and software requirements for installing WISE are as follows.

- A modern computer with Ubuntu, Debian, or other similar OS
  - There may be installation issues on macOS and Windows. We recommend using a Linux environment (or WSL) for now
- Python version 3.10 (or higher)

To install WISE, we first download the WISE source code.

The latest stable release of WISE is available at https://gitlab.com/vgg/wise/wise/-/releases .

```
## 1. Download the latest release and extract the WISE code
curl -sLO https://gitlab.com/vgg/wise/wise/-/archive/wise-2.1.0/wise-wise-2.1.0.zip
unzip wise-wise-2.1.0.zip
mv wise-wise-2.1.0 wise-2.1.0
cd wise-2.1.0
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

- If you are using WISE on an Intel platforms, you may install the MKL
  distribution of BLAS for better performance on FAISS by appending
  the `blas=*=mkl` argument to the `conda env create` command.

```
conda env create --name wise -f environment.yml
conda activate wise
```

(For mamba, replace conda in the above command accordingly)

## Option 2: Installation using venv

If the conda tool based dependency management option is not suitable, the alternative
is to use Python's virtual environment [venv](https://docs.python.org/3/library/venv.html)
module for installing the dependencies as shown below.

```
python3 --version                  # must be >= 3.10
sudo apt install ffmpeg            # ffmpeg>=4.4.2,<7.0 is required to load videos
python3 -m venv wise-dep/          # create virtual environment
source wise-dep/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Option 3: Install WISE to use only CPU (i.e. no GPU)

WISE can be installed on a machine without a GPU. While the processing speed is slow,
all the functionality of the WISE software remains available.

```
python3 --version                  # must be >= 3.10
sudo apt install ffmpeg            # ffmpeg>=4.4.2,<7.0 is required to load videos
python3 -m venv wise-dep/          # create virtual environment
source wise-dep/bin/activate
python -m pip install --upgrade pip

# Note: the following command is the key to installing CPU only version of WISE
pip install --index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

pip install -r requirements.txt
```
