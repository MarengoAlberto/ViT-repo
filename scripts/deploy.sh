#!/bin/bash

# Load ENV variables
set -a
source .env

python deployment/deploy_gcp.py