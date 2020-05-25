FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3
ADD . /competition
WORKDIR /competition
RUN chmod 777 -R /competition
RUN pip --no-cache-dir install  -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
CMD ["sh", "run.sh"]
