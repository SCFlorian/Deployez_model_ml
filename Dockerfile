FROM python:3.10

WORKDIR /home/user/app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app:fastapi_app", "--host", "0.0.0.0", "--port", "7860"]