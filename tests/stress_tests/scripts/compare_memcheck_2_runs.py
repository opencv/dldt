#!/usr/bin/env python3
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
Create comparison table based on MemCheckTests results from 2 runs
Usage: ./scrips/compare_memcheck_2_runs.py cur_source ref_source \
       --db_collection collection_name --out_file file_name
"""
# pylint:disable=line-too-long

import argparse
import json
import os
import sys
from collections import OrderedDict
from glob import glob
from operator import itemgetter
from pathlib import Path
import logging as log

from pymongo import MongoClient
from memcheck_upload import create_memcheck_records

# Database arguments
from memcheck_upload import DATABASE, DB_COLLECTIONS


class HashableDict(dict):
    """Dictionary class with defined __hash__ to make it hashable
       (e.g. use as key in another dictionary)"""
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def get_db_memcheck_records(query, db_collection, db_name, db_url):
    """Request MemCheckTests records from database by provided query"""
    client = MongoClient(db_url)
    collection = client[db_name][db_collection]
    items = list(collection.find(query))
    return items


def get_memcheck_records(source, db_collection=None, db_name=None, db_url=None):
    """provide MemCheckTests records"""
    if os.path.isdir(source):
        logs = list(glob(os.path.join(source, '**', '*.log'), recursive=True))
        items = create_memcheck_records(logs, build_url=None, artifact_root=source)
    else:
        assert db_collection and db_name and db_url
        query = json.loads(source)
        items = get_db_memcheck_records(query, db_collection, db_name, db_url)

    return items


def compare_memcheck_2_runs(cur_values, references, output_file=None):
    """Compares 2 MemCheckTests runs and prepares a report on specified path"""
    import pandas                   # pylint:disable=import-outside-toplevel
    from scipy.stats import gmean   # pylint:disable=import-outside-toplevel

    returncode = 0

    # constants
    metric_name_template = "{} {}"
    GEOMEAN_THRESHOLD = 0.9

    # Fields should be presented in both `references` and `cur_values`.
    # Some of metrics may be missing for one of `references` and `cur_values`.
    # Report will contain data with order defined in `required_fields` and `required_metrics`
    required_fields = [
        # "metrics" should be excluded because it will be handled automatically
        "model", "device", "test_name"
    ]
    required_metrics = [
        "vmrss", "vmhwm",
        # "vmsize", "vmpeak"    # temporarily disabled as unused
    ]
    # `Ops` is a template applied for every metric defined in `required_metrics`
    ops = OrderedDict([
        # x means ref, y means cur
        ("ref", lambda x, y: x),
        ("cur", lambda x, y: y),
        ("cur-ref", lambda x, y: y - x if (x is not None and y is not None) else None),
        ("ref/cur", lambda x, y: x / y if (x is not None and y is not None) else None)
    ])
    # `Comparison_ops` is a template applied for metrics columns
    # generated by applied `ops` to propagate status of function
    comparison_ops = {
        # format: {metric_col_name: (operation, message)}
        metric_name_template.format("vmrss", "ref/cur"):
            lambda x: (gmean(x) > GEOMEAN_THRESHOLD,
                       "geomean={} is less than threshold={}".format(gmean(x), GEOMEAN_THRESHOLD)),
        metric_name_template.format("vmhwm", "ref/cur"):
            lambda x: (gmean(x) > GEOMEAN_THRESHOLD,
                       "geomean={} is less than threshold={}".format(gmean(x), GEOMEAN_THRESHOLD))
    }

    filtered_refs = {}
    for record in references:
        filtered_rec = {key: val for key, val in record.items() if key in required_fields}
        filtered_rec_metrics = {key: val for key, val in record["metrics"].items() if key in required_metrics}
        filtered_refs[HashableDict(filtered_rec)] = filtered_rec_metrics

    filtered_cur_val = {}
    for record in cur_values:
        filtered_rec = {key: val for key, val in record.items() if key in required_fields}
        filtered_rec_metrics = {key: val for key, val in record["metrics"].items() if key in required_metrics}
        filtered_cur_val[HashableDict(filtered_rec)] = filtered_rec_metrics

    comparison_data = []
    for data in [filtered_refs, filtered_cur_val]:
        for record in data:
            rec = OrderedDict()
            for field in required_fields:
                rec.update({field: record[field]})
                rec.move_to_end(field)
            if rec not in comparison_data:
                # Comparison data should contain unique records combined from references and current values
                comparison_data.append(rec)
    comparison_data = sorted(comparison_data, key=itemgetter("model"))

    for record in comparison_data:
        metrics_rec = OrderedDict()
        for metric in required_metrics:
            ref = filtered_refs.get(HashableDict(record), {}).get(metric, None)
            cur = filtered_cur_val.get(HashableDict(record), {}).get(metric, None)
            for op_name, op in ops.items():
                op_res = op(ref, cur)
                metric_name = metric_name_template.format(metric, op_name)
                metrics_rec.update({metric_name: op_res})
                metrics_rec.move_to_end(metric_name)
        # update `comparison_data` with metrics
        for metric_name, op_res in metrics_rec.items():
            record.update({metric_name: op_res})
            record.move_to_end(metric_name)

    # compare data using `comparison_ops`
    orig_data = pandas.DataFrame(comparison_data)
    data = orig_data.dropna()
    
    devices = data["device"].unique()
    for device in devices:
        frame = data[data["device"] == device]
        for field, comp_op in comparison_ops.items():
            status, msg = comp_op(frame.loc[:, field])
            if not status:
                log.error('Comparison for field="%s" for device="%s" failed: %s', field, device, msg)
                returncode = 1

    # dump data to file
    if output_file:
        if os.path.splitext(output_file)[1] == ".html":
            orig_data.to_html(output_file)
        else:
            orig_data.to_csv(output_file)
        log.info('Created memcheck comparison report %s', output_file)

    return returncode


def cli_parser():
    """parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Compare 2 runs of MemCheckTests')
    parser.add_argument('cur_source',
                        help='Source of current values of MemCheckTests. '
                             'Should contain path to a folder with logs or '
                             'JSON-format query to request data from DB.')
    parser.add_argument('ref_source',
                        help='Source of reference values of MemCheckTests. '
                             'Should contain path to a folder with logs or '
                             'JSON-format query to request data from DB.')
    parser.add_argument('--db_url',
                        help='MongoDB URL in a for "mongodb://server:port".')
    parser.add_argument('--db_collection',
                        help=f'Collection name in "{DATABASE}" database to query'
                             f' data using current source.',
                        choices=DB_COLLECTIONS)
    parser.add_argument('--ref_db_collection',
                        help=f'Collection name in "{DATABASE}" database to query'
                             f' data using reference source.',
                        choices=DB_COLLECTIONS)
    parser.add_argument('--out_file', dest='output_file', type=Path,
                        help='Path to a file (with name) to save results. '
                             'Example: /home/.../file.csv')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = cli_parser()
    references = get_memcheck_records(args.ref_source, args.ref_db_collection, DATABASE, args.db_url)
    cur_values = get_memcheck_records(args.cur_source, args.db_collection, DATABASE, args.db_url)
    exit_code = compare_memcheck_2_runs(cur_values, references, output_file=args.output_file)
    sys.exit(exit_code)
