FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10
EXPOSE 8000
WORKDIR /app
COPY ./app /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . . 





