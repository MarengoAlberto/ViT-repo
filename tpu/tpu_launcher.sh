# dump all the environment variables so that we can see them in the
# execution log
export
# use an auxiliary python script to get the IP Address and Port of
# the TPU VM
tpu_address=`python /root/tpu/launcher.py`
echo "tpu_address is $tpu_address"
# export the TPU address and port
export XRT_TPU_CONFIG="tpu_worker;0;$tpu_address"
# invoke the trainer code
python -m trainer.task