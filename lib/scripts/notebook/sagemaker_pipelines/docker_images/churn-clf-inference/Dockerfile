FROM python:3.7.6-slim-buster

RUN rm -rf /usr/share/man/man1/; mkdir /usr/share/man/man1/ 

RUN apt-get update -y && apt-get -y install python3-dev gcc --no-install-recommends default-jdk 

RUN pip3 install --no-cache-dir --upgrade pip 

RUN pip3 install retrying \
    pandas==1.2.4 \
    numpy==1.20.2 \
    scikit-learn==0.24.2 \
    multi-model-server==1.1.8 \
    sagemaker-inference==1.5.11 \
    boto3==1.21.43 \
    itsdangerous==2.0.1

COPY serving/ /opt/ml/serving

RUN pip install -e /opt/ml/serving 

LABEL com.amazonaws.sagemaker.capabilities.multi-models=false 

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true 

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

ENTRYPOINT ["python", "/opt/ml/serving/custom_inference/serving.py"]

CMD ["serve"]