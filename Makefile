run-locally:
				scripts/run_locally.sh

build:
				scripts/build.sh

run:
				scripts/run.sh

push:
				scripts/push.sh

deploy:
				scripts/deploy.sh

train:
				python -m trainer.task

train-distributed:
				scripts/train_distributed.sh

train-tpu:
				scripts/train_tpu.sh
