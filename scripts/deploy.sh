#!/bin/bash

# Load ENV variables
set -a
source environments/.env

python deployment/deploy_gcp.py