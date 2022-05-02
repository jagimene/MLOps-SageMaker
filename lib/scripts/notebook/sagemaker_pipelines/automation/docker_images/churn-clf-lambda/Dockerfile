FROM public.ecr.aws/lambda/python:3.8

RUN pip3 install retrying --target "${LAMBDA_TASK_ROOT}" \
    boto3==1.17.77

COPY lambda/register_model/ ${LAMBDA_TASK_ROOT} 



CMD ["app.lambda_handler"]