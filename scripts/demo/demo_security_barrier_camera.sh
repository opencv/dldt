#!/usr/bin/env bash

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

printf "\e[0;33mWARNING: If you get an error when running the demo in the Docker container, you may need to install additional packages. To do this, run the container as root (-u 0) and run install_openvino_dependencies.sh script. If you get a package-independent error, try setting additional parameters using -sample-options.\e[0m\n"

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]-$0}" )" && pwd )"

. "$ROOT_DIR/utils.sh"

usage() {
    echo "Security barrier camera demo that showcases three models coming with the product"
    echo "-d name     specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD are acceptable. Sample will look for a suitable plugin for device specified"
    echo "-help            print help message"
    echo "-sample-options   specify command line arguments for the sample"
    exit 1
}

trap 'error ${LINENO}' ERR

target="CPU"

# parse command line options
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -h | -help | --help)
    usage
    ;;
    -d)
    target="$2"
    echo target = "${target}"
    shift
    ;;
    -sample-options)
    sampleoptions="$2 $3 $4 $5 $6"
    echo sample-options = "${sampleoptions}"
    shift
    ;;
    *)
    # unknown option
    ;;
esac
shift
done


target_image_path="$ROOT_DIR/car_1.bmp"

run_again="Then run the script again\n\n"
dashes="\n\n###################################################\n\n"

if [[ -f /etc/centos-release ]]; then
    DISTRO="centos"
elif [[ -f /etc/lsb-release ]]; then
    DISTRO="ubuntu"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    DISTRO="macos"
fi

if [[ $DISTRO == "centos" ]]; then
    # check installed Python version
    if command -v python3.5 >/dev/null 2>&1; then
        python_binary=python3.5
    fi
    if command -v python3.6 >/dev/null 2>&1; then
        python_binary=python3.6
    fi
elif [[ $DISTRO == "ubuntu" ]]; then
    python_binary=python3
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # check installed Python version
    if command -v python3.8 >/dev/null 2>&1; then
        python_binary=python3.8
    elif command -v python3.7 >/dev/null 2>&1; then
        python_binary=python3.7
    elif command -v python3.6 >/dev/null 2>&1; then
        python_binary=python3.6
    elif command -v python3.5 >/dev/null 2>&1; then
        python_binary=python3.5
    else
        python_binary=python3
    fi
fi

if ! command -v $python_binary &>/dev/null; then
    printf "\n\nPython 3.5 (x64) or higher is not installed. It is required to run Model Optimizer, please install it. ${run_again}"
    exit 1
fi

"$python_binary" -m pip install -U pip
"$python_binary" -m pip install -r "$ROOT_DIR/../open_model_zoo/tools/downloader/requirements.in"

if [ -e "$ROOT_DIR/../../bin/setupvars.sh" ]; then
    setupvars_path="$ROOT_DIR/../../bin/setupvars.sh"
else
    printf "Error: setupvars.sh is not found\n"
fi
if ! . "$setupvars_path" ; then
    printf "Unable to run ./setupvars.sh. Please check its presence. ${run_again}"
    exit 1
fi

# Step 1. Downloading Intel models
printf "${dashes}"
printf "Downloading Intel models\n\n"


target_precision="FP16"

printf "target_precision = ${target_precision}\n"

downloader_dir="${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo/tools/downloader"

downloader_path="$downloader_dir/downloader.py"
models_path="$HOME/openvino_models/ir"
models_cache="$HOME/openvino_models/cache"

declare -a model_args

while read -r model_opt model_name; do
    model_subdir=$("$python_binary" "$downloader_dir/info_dumper.py" --name "$model_name" |
        "$python_binary" -c 'import sys, json; print(json.load(sys.stdin)[0]["subdirectory"])')

    model_path="$models_path/$model_subdir/$target_precision/$model_name"

    print_and_run "$python_binary" "$downloader_path" --name "$model_name" --output_dir "$models_path" --cache_dir "$models_cache"

    model_args+=("$model_opt" "${model_path}.xml")
done < "$ROOT_DIR/demo_security_barrier_camera.conf"

# Step 2. Build samples
printf "${dashes}"
printf "Build Inference Engine demos\n\n"

demos_path="${INTEL_OPENVINO_DIR}/deployment_tools/open_model_zoo/demos"

if ! command -v cmake &>/dev/null; then
    printf "\n\nCMAKE is not installed. It is required to build Inference Engine demos. Please install it. ${run_again}"
    exit 1
fi

OS_PATH=$(uname -m)
NUM_THREADS="-j2"

if [ "$OS_PATH" == "x86_64" ]; then
  OS_PATH="intel64"
  NUM_THREADS="-j8"
fi

build_dir="$HOME/inference_engine_demos_build"
if [ -e "$build_dir/CMakeCache.txt" ]; then
  rm -rf "$build_dir/CMakeCache.txt"
fi
mkdir -p "$build_dir"
cd "$build_dir"
cmake -DCMAKE_BUILD_TYPE=Release "$demos_path"
make $NUM_THREADS security_barrier_camera_demo

# Step 3. Run samples
printf "${dashes}"
printf "Run Inference Engine security_barrier_camera demo\n\n"

binaries_dir="${build_dir}/${OS_PATH}/Release"
cd "$binaries_dir"

print_and_run ./security_barrier_camera_demo -d "$target" -d_va "$target" -d_lpr "$target" -i "$target_image_path" "${model_args[@]}" ${sampleoptions}

printf "${dashes}"
printf "Demo completed successfully.\n\n"
