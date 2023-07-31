FROM pytorch/torchserve:latest-cpu

USER root
# run and update some basic packages software packages, including security libs
RUN apt-get update &&     apt-get install -y software-properties-common &&     add-apt-repository -y ppa:ubuntu-toolchain-r/test &&     apt-get update &&     apt-get install -y gcc-9 g++-9 apt-transport-https ca-certificates gnupg curl

# Install gcloud tools for gsutil as well as debugging
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" |     tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&     curl https://packages.cloud.google.com/apt/doc/apt-key.gpg |     apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - &&     apt-get update -y &&     apt-get install google-cloud-sdk -y

USER model-server

# install dependencies
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install google-cloud-storage

# COPY src /home/model-server/src
COPY serve/model-store/ViT-model.mar /home/model-server/model-store/
COPY token/alberto-playground-bc631f2fcabd.json /home/model-server/

ENV GOOGLE_APPLICATION_CREDENTIALS=/home/model-server/alberto-playground-bc631f2fcabd.json

ARG MODEL_NAME=ViT-model
ENV MODEL_NAME="${MODEL_NAME}"

ARG MAR_URI=gs://alberto-playground/serve
ENV MAR_URI="${AIP_STORAGE_URI}"

# health and prediction listener ports
ARG AIP_HTTP_PORT=7080
ENV AIP_HTTP_PORT="${AIP_HTTP_PORT}"

ARG MODEL_MGMT_PORT=7081

# expose health and prediction listener ports from the image
EXPOSE "${AIP_HTTP_PORT}"
EXPOSE "${MODEL_MGMT_PORT}"
EXPOSE 8080 8081 8082 7070 7071

# create torchserve configuration file
USER root
RUN echo "service_envelope=json\n"     "inference_address=http://0.0.0.0:${AIP_HTTP_PORT}\n"     "management_address=http://0.0.0.0:${MODEL_MGMT_PORT}" >>     /home/model-server/config.properties
USER model-server

# run Torchserve HTTP serve to respond to prediction requests
CMD ["echo", "MAR_URI=${MAR_URI}", ";",    "ls", "-ltr", "/home/model-server/model-store/", ";",     "torchserve", "--start", "--ts-config=/home/model-server/config.properties",     "--models", "${MODEL_NAME}=${MODEL_NAME}.mar",     "--model-store", "/home/model-server/model-store"]
