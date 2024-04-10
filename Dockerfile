FROM --platform=linux/amd64 python:3.9-slim-buster as build
#FROM python:3.9-slim-buster as build
ADD . .

RUN apt-get -y update

RUN apt-get install -y ffmpeg

RUN pip install --default-timeout=1000 --no-cache-dir -r requirement.txt

#RUN python -m nltk.downloader punkt



CMD ["python3","./init.py"]