FROM python:3.6

LABEL authors="Sanjeev Kumar Karn and Mark Buckley"

WORKDIR /app

COPY ./app/requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY ./app /app

RUN pip install . --no-deps

RUN mkdir /data /auth

CMD /app/bin/run_with_restart_docker 
