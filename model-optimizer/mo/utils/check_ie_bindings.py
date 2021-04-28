#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import platform
import re
import sys

try:
    import mo
    execution_type = "mo"
except ModuleNotFoundError:
    mo_root_path = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    sys.path.insert(0, mo_root_path)
    execution_type = "install_prerequisites.{}".format("bat" if platform.system() == "Windows" else "sh")

import mo.utils.version as v
try:
    import openvino_telemetry as tm  # pylint: disable=import-error,no-name-in-module
except ImportError:
    import mo.utils.telemetry_stub as tm
from mo.utils.error import classify_error_type


def send_telemetry(mo_version: str, message: str, event_type: str):
    t = tm.Telemetry(app_name='Model Optimizer', app_version=mo_version)
    t.start_session('mo')
    t.send_event(execution_type, event_type, message)
    t.end_session('mo')
    t.force_shutdown(1.0)


def import_core_modules(silent: bool, path_to_module: str):
    try:
        from openvino.inference_engine import IECore, get_version  # pylint: disable=import-error,no-name-in-module
        from openvino.offline_transformations import ApplyMOCTransformations, CheckAPI  # pylint: disable=import-error,no-name-in-module

        import openvino # pylint: disable=import-error

        if silent:
            return True

        ie_version = str(get_version())
        mo_version = str(v.get_version()) # pylint: disable=no-member

        print("\t- {}: \t{}".format("Inference Engine found in", os.path.dirname(openvino.__file__)))
        print("{}: \t{}".format("Inference Engine version", ie_version))
        print("{}: \t    {}".format("Model Optimizer version", mo_version))

        versions_mismatch = False
        # MO and IE version have a small difference in the beginning of version because
        # IE version also includes API version. For example:
        #   Inference Engine version: 2.1.custom_HEAD_4c8eae0ee2d403f8f5ae15b2c9ad19cfa5a9e1f9
        #   Model Optimizer version:      custom_HEAD_4c8eae0ee2d403f8f5ae15b2c9ad19cfa5a9e1f9
        # So to match this versions we skip IE API version.
        if not re.match(r"^([0-9]+).([0-9]+).{}$".format(mo_version), ie_version):
            versions_mismatch = True
            extracted_mo_release_version = v.extract_release_version(mo_version)
            mo_is_custom = extracted_mo_release_version == (None, None)

            print("[ WARNING ] Model Optimizer and Inference Engine versions do no match.")
            print("[ WARNING ] Consider building the Inference Engine Python API from sources or reinstall OpenVINO (TM) toolkit using", end=" ")
            if mo_is_custom:
                print("\"pip install openvino\" (may be incompatible with the current Model Optimizer version)")
            else:
                print("\"pip install openvino=={}.{}\"".format(*extracted_mo_release_version))

        simplified_mo_version = v.get_simplified_mo_version()
        message = str(dict({
            "platform": platform.system(),
            "mo_version": simplified_mo_version,
            "ie_version": v.get_simplified_ie_version(version=ie_version),
            "versions_mismatch": versions_mismatch,
        }))
        send_telemetry(simplified_mo_version, message, 'ie_version_check')

        return True
    except Exception as e:
        # Do not print a warning if module wasn't found or silent mode is on
        if "No module named 'openvino'" not in str(e) and not silent:
            print("[ WARNING ] Failed to import Inference Engine Python API in: {}".format(path_to_module))
            print("[ WARNING ] {}".format(e))

            # Send telemetry message about warning
            simplified_mo_version = v.get_simplified_mo_version()
            message = str(dict({
                "platform": platform.system(),
                "mo_version": simplified_mo_version,
                "ie_version": v.get_simplified_ie_version(env=os.environ),
                "python_version": sys.version,
                "error_type": classify_error_type(e),
            }))
            send_telemetry(simplified_mo_version, message, 'ie_import_failed')

        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--path_to_module")
    args = parser.parse_args()

    if not import_core_modules(args.silent, args.path_to_module):
        exit(1)
