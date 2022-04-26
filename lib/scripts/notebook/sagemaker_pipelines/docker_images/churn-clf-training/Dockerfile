FROM python:3.7.6-slim-buster

RUN rm -rf /usr/share/man/man1/; mkdir /usr/share/man/man1/ 

RUN apt-get update -y && apt-get -y install python3-dev gcc --no-install-recommends default-jdk 

RUN pip3 install --no-cache-dir --upgrade pip 

RUN pip3 install retrying \
    pandas==1.2.4 \
    numpy==1.20.2 \
    scikit-learn==0.24.2 \
    sagemaker-training==3.9.2


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8