#FROM ubuntu:latest
FROM python:3.8.1
COPY ./api/hello.py /exp/
COPY ./svm_Gamma=0.01_C=0.5.joblib /exp/
COPY ./requirements.txt /exp/requirements.txt
RUN pip install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
ENV FLASK_APP=hello
#CMD ["flask run"]
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
EXPOSE 5000
