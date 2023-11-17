FROM python:3.11.6-slim-bullseye

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 5000

CMD ["python3", "./app.py"]
