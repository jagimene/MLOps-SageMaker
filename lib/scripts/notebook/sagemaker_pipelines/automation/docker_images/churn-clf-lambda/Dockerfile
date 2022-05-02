FROM 829825986145.dkr.ecr.us-east-1.amazonaws.com/churn-clf-processing:latest

RUN rm -rf /usr/share/man/man1/; mkdir /usr/share/man/man1/ 

RUN apt-get update -y && apt-get -y install python3-dev gcc --no-install-recommends default-jdk 

RUN pip3 install --no-cache-dir --upgrade pip 

RUN pip3 install retrying \
    awslambdaric==1.0.0 \
    boto3==1.17.77

COPY lambda_function /function

WORKDIR /function 

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

ENTRYPOINT ["python", "-m", "awslambdaric"]

CMD ["app.lambda_handler"]