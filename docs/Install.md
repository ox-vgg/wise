> See [User Guide](UserGuide.md) for instructions on using WISE for creating audiovisual search engine.

# WISE Installation

The requirements for installing WISE are as follows:

- A computer with a GNU/Linux distribution:
  - We develop WISE in Ubuntu and Debian but other distributions
    should work equally well;
  - There may be installation issues on macOS and Windows (we have
    simply not tested it yet but are interested in knowing about
    failures and successes).

- Python version 3.10 or higher;

- A Nvidia GPU is not required but WISE will be slower without one;

- [FFmpeg](https://ffmpeg.org/) installation compatible with Python's
  [torchaudio](https://docs.pytorch.org/audio/2.8.0/installation.html)

## Quick start installation

For the least amount of grief and the fastest install, we recommend
using [Miniconda](https://docs.anaconda.com/miniconda/) and provide a
Conda environment file.  Using Conda, the current development version
of WISE can be installed like so:

```
git clone https://gitlab.com/vgg/wise/wise.git
cd wise
conda env create --name wise --file environment.yml
conda activate wise
pip install .
```

[Mamba and Micromamba](https://mamba.readthedocs.io/) are also
supported, just replace `conda` with `mamba` or `micromamba` on the
commands above.

To check if WISE is successfully installed run
```bash
wise --help
```

## Full install instructions

WISE is a Python package and can be installed with `pip` as any other
Python package.  However, WISE is not yet distributed on any Python
package repository so needs to be installed from development sources,
the short version of it is:

```
git clone https://gitlab.com/vgg/wise/wise.git
cd wise
pip install .
```

### Dependencies

#### FAISS Python package

The [FAISS](https://faiss.ai/) Python package is named `faiss` but its
official distribution on PyPI is named `faiss-cpu`, i.e., the
`faiss-cpu` distribution installs the package named `faiss`.

Currently, `requirements.txt` lists a requirement on `faiss-cpu`.  If
you build FAISS from source, or if you install it with Conda, you will
end up with the `faiss` package installed from a distribution named
`faiss`.  The package installed from this distribution will then clash
with the package installed from the `faiss-cpu` distribution (see
[issue #232](https://gitlab.com/vgg/wise/wise/-/issues/232) for
details).  To avoid issues, if you do not want to use the `faiss-cpu`
distribution, you can either:

- uninstall `faiss-cpu` after installing wise and then install the
  build of FAISS you want to use, or;

- if you are installing wise from development sources, remove
  `faiss-cpu` from the `requirements.txt`.

Note that WISE does not use the GPU with FAISS even if FAISS was built
with GPU support so `faiss-cpu` is enough.

There is also a `faiss` Python distribution on PyPI.  That is an
"unofficial" distribution of FAISS, unmaintained since 2019 and still
in version 1.5.3.

#### PyTorch (torch, torchvision, and torchaudio)

[PyTorch](https://pytorch.org/) is another dependency of WISE.  There
are PyPI distributions for its torch, torchvision, and torchaudio
packages so typically they require no particular attention.  PyTorch
also provides built distributions for different versions of CUDA,
ROCm, operating systems for both Conda and pip.  Refer to PyTorch
[install page](https://pytorch.org/get-started/locally/) if needed.

#### FFmpeg

FFmpeg is a transitive dependency from Torchaudio.  It is worth of
note simply because it is not a Python package and is only checked at
runtime.  Please refer to the documentation for the Torchaudio version
you have installed for its [FFmpeg version
compatibility](https://docs.pytorch.org/audio/2.8.0/installation.html).

### Running WISE with an AMD GPU

We have not tested running WISE with an AMD GPU.  However, PyTorch
provides distributions built for ROCm.  Please refer to [PyTorch
install page](https://pytorch.org/get-started/locally/).  We are
interested in knowing about any successes or failures with AMD GPUs.

### Running WISE without a GPU

WISE can run without using a GPU.  While the processing speed is
slower, all the functionality of WISE remains available.  The main
performance impact will be during the "extract features" step which
can be done offline and in a different computer.

### Development install

If you plan to make changes to WISE, install the `dev` dependencies
and consider installing in editable mode, i.e., setuptools "develop
mode".  For that, use:

    cd wise
    pip install --editable .[dev]

### Virtual Environments

It is possible to have Python environments without Conda.  [Python's
venv](https://docs.python.org/3/library/venv.html) is a common choice
and comes builtin with Python.  To use it, create and activate the
environment before installing WISE and its dependencies:

    cd wise
    python3 -m venv venv/       # create virtual environment (optional)
    source venv/bin/activate    # activate virtual environment (optional)
    pip install .
