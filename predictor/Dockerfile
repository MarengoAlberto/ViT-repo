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
#COPY token/alberto-playground-bc631f2fcabd.json /home/model-server/

#ENV GOOGLE_APPLICATION_CREDENTIALS=/home/model-server/alberto-playground-bc631f2fcabd.json

# Accept the environment variable
ARG MODEL_NAME
# Set the environment variable
ENV MODEL_NAME="${MODEL_NAME}"
# Accept the environment variable
ARG PROJECT_ID
# Set the environment variable
ENV PROJECT_ID=${PROJECT_ID}
# Accept the environment variable
ARG REGION
# Set the environment variable
ENV REGION=${REGION}
# Accept the environment variable
ARG BUCKET_NAME
# Set the environment variable
ENV BUCKET_NAME=${BUCKET_NAME}
# Accept the environment variable
ARG EMBED_DIM
# Set the environment variable
ENV EMBED_DIM=${EMBED_DIM}
# Accept the environment variable
ARG HIDDEN_DIM
# Set the environment variable
ENV HIDDEN_DIM=${HIDDEN_DIM}
# Accept the environment variable
ARG NUM_HEADS
# Set the environment variable
ENV NUM_HEADS=${NUM_HEADS}
# Accept the environment variable
ARG NUM_LAYERS
# Set the environment variable
ENV NUM_LAYERS=${NUM_LAYERS}
# Accept the environment variable
ARG PATCH_SIZE
# Set the environment variable
ENV PATCH_SIZE=${PATCH_SIZE}
# Accept the environment variable
ARG NUM_CHANNELS
# Set the environment variable
ENV NUM_CHANNELS=${NUM_CHANNELS}
# Accept the environment variable
ARG NUM_PATCHES
# Set the environment variable
ENV NUM_PATCHES=${NUM_PATCHES}
# Accept the environment variable
ARG NUM_CLASSES
# Set the environment variable
ENV NUM_CLASSES=${NUM_CLASSES}
# Accept the environment variable
ARG DROPOUT
# Set the environment variable
ENV DROPOUT=${DROPOUT}
# Accept the environment variable
ARG MAP_CLASSES_PATH
# Set the environment variable
ENV MAP_CLASSES_PATH=${MAP_CLASSES_PATH}

ARG MAR_URI=gs://${PROJECT_ID}/serve
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
