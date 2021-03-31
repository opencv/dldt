# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import re
import sys
import subprocess


def get_version_file_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "version.txt")


def generate_mo_version():
    """
    Function generates version like in cmake
    custom_{branch_name}_{commit_hash}
    """
    try:
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode()
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
        return "custom_{}_{}".format(branch_name, commit_hash)
    except Exception as e:
        return "unknown version"


def get_version():
    version_txt = get_version_file_path()
    if not os.path.isfile(version_txt):
        return generate_mo_version()
    with open(version_txt) as f:
        version = f.readline().replace('\n', '')
    return version


def extract_release_version(version: str):
    patterns = [
        # captures release version set by CI for example: '2021.1.0-1028-55e4d5673a8'
        r"^([0-9]+).([0-9]+)*",
        # captures release version generated by MO from release branch, for example: 'custom_releases/2021/1_55e4d567'
        r"_releases/([0-9]+)/([0-9]+)_*"
    ]

    for pattern in patterns:
        m = re.search(pattern, version)
        if m and len(m.groups()) == 2:
            return m.group(1), m.group(2)
    return None, None


def simplify_version(version: str):
    release_version = extract_release_version(version)
    if release_version == (None, None):
        return "custom"
    return "{}.{}".format(*release_version)


def get_simplified_mo_version():
    return simplify_version(get_version())


def get_simplified_ie_version(env=dict(), version=None):
    if version is None:
        try:
            version = subprocess.check_output([sys.executable, os.path.join(os.path.dirname(__file__), "ie_version.py")], timeout=2, env=env).strip().decode()
        except:
            return "ie not found"
    m = re.match(r"^([0-9]+).([0-9]+).(.*)", version)
    if m and len(m.groups()) == 3:
        return simplify_version(m.group(3))
    return "custom"
