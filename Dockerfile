FROM python:3.9-slim

RUN apt-get update && apt-get upgrade -y
# RUN apt install libgl1-mesa-glx libglib2.0-0 -y

WORKDIR /opt

# https://docs.python.org/3/using/cmdline.html#envvar-PYTHONDONTWRITEBYTECODE
# Prevents Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE 1

# ensures that the python output is sent straight to terminal (e.g. your container log)
# without being first buffered and that you can see the output of your application (e.g. django logs)
# in real time. Equivalent to python -u: https://docs.python.org/3/using/cmdline.html#cmdoption-u
ENV PYTHONUNBUFFERED 1
# ENV ENVIRONMENT prod
# ENV TESTING 0


COPY ./requirements.txt /opt/app/requirements.txt

RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /opt/app/requirements.txt

COPY ./app /opt/app
RUN mkdir -p /opt/data
RUN mkdir -p /opt/mean_image
RUN mkdir -p /opt/histograms

EXPOSE 8080

# Start app
ENTRYPOINT ["gunicorn", "-c", "app/gunicorn.py", "-k", "uvicorn.workers.UvicornWorker", "app.main:app"]

# ENTRYPOINT ["uvicorn", "app.main:app"]
# CMD ["--host", "0.0.0.0", "--port", "8080"]
