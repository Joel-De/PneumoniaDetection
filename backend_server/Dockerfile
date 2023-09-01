FROM python:3.11
WORKDIR /app

copy ./backend_server/requirements.txt ./

RUN pip install  -r ./requirements.txt

COPY ./backend_server ./
COPY ./common ./common

CMD ["python", "main_api.py"]