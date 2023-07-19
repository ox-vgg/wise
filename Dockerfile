FROM python:3.10-bookworm
RUN wget -O requirements.txt https://gitlab.com/vgg/wise/wise/-/raw/main/requirements.txt
RUN wget -O torch-faiss-requirements.txt https://gitlab.com/vgg/wise/wise/-/raw/main/torch-faiss-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r torch-faiss-requirements.txt
WORKDIR /wise
