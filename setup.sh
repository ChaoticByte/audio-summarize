#!/usr/bin/env bash

# init
oldcwd=$(pwd)
function cleanup {
    cd ${oldcwd}
}
trap cleanup EXIT

export root_dir=$(realpath $(dirname $0))
export vendor_dir=${root_dir}/vendor

# Prepare installation of dependencies

mkdir -p ${vendor_dir}
cd ${vendor_dir}

# Install whisper.cpp

if [ ! -d ./whisper.cpp ]; then
    git clone -b v1.6.2 https://github.com/ggerganov/whisper.cpp.git
fi
cd whisper.cpp
make
cd ${vendor_dir}

# Install python packages

if ! python3 -m pip install -r "${root_dir}/requirements.txt"; then
    echo
    echo "Make shure to run this script in a python virtual environment!"
fi
