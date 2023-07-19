FROM python:3.10-bookworm
SHELL ["/bin/bash", "-c"]
RUN python3 -m venv /home/env
ENV PATH="/home/env/bin:$PATH"
RUN wget -O requirements.txt https://gitlab.com/vgg/wise/wise/-/raw/main/requirements.txt
RUN wget -O torch-faiss-requirements.txt https://gitlab.com/vgg/wise/wise/-/raw/main/torch-faiss-requirements.txt
RUN pip install -r requirements.txt
RUN pip install -r torch-faiss-requirements.txt
WORKDIR /wise

