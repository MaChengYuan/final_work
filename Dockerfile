
FROM --platform=linux/amd64 python:3.9-slim-buster as build

ADD . .

RUN pip install -r requirement.txt

CMD ["python3","./init.py"]