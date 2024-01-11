#use the latest version of XLA enabled Pytorch
FROM gcr.io/tpu-pytorch/xla:nightly
WORKDIR /root
COPY . .
# Installs Tensorflow to resolve the TPU name to IP Address
RUN pip install tensorflow
# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
RUN pip install cloudml-hypertune
RUN pip install -r requirement.txt
# Installs google cloud sdk, this is mostly for using gsutil to
RUN wget -nv https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz &&  \
    mkdir /root/tools && tar xvzf google-cloud-sdk.tar.gz -C /root/tools && rm google-cloud-sdk.tar.gz &&  \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false  \
    --path-update=false --bash-completion=false --disable-installation-options &&  \
    rm -rf /root/.config/* &&  \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates \
    rm -rf /root/tools/google-cloud-sdk/.install/.backup
# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg
# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["sh", "tpu/tpu_launcher.sh"]