#!/usr/bin/env bash

sudo easy_install pip
pip install --upgrade virtualenv

virtualenv --system-site-packages venv

. ./venv/bin/activate

easy_install -U pip

#pip install --upgrade tensorflow
pip install -U -r ./src/requirements.txt