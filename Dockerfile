FROM python:3.11
WORKDIR /app
VOLUME /app/Data
VOLUME /app/wandb
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/
CMD ["python", "main.py"]
