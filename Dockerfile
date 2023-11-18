FROM python:3.11.6-slim-bullseye

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN apt-get update
RUN apt-get install unzip
COPY . .
RUN unzip ./test_data.zip

EXPOSE 5000

CMD ["python3", "-u", "./server.py"]
