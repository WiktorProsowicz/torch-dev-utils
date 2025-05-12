#!/bin/bash

# This script should be run after the devcontainer is created.
# It's role is to establish the necessary environment for the
# user to start developing.

deactivate 2> /dev/null
rm -rf venv

python3 setup_scripts.py setup_venv && source venv/bin/activate

python setup_scripts.py install_package

cat .devcontainer/res/.bash_aliases >> ~/.bash_aliases
cat .devcontainer/res/.bashrc >> ~/.bashrc
