FROM bempp/cl-notebook
LABEL description="Docker image for pbj"

COPY requirements.txt /root
WORKDIR /root
#RUN /usr/bin/python -m pip install --upgrade pip
RUN pip3 install -r requirements.txt 
EXPOSE 8888/tcp
ENV SHELL /bin/bash
ENTRYPOINT ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]

