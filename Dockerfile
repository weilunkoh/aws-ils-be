# FROM python:3.8-slim
# FROM public.ecr.aws/docker/library/python:3.8-slim
FROM public.ecr.aws/docker/library/python:3.10-slim

WORKDIR /app

# add application code to /app
COPY . .

# install dependencies
RUN pip install -r requirements.txt

