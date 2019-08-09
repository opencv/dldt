"""
Copyright (C) 2018-2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from .aggregated_statistics import AggregatedStatistics
from .calibrator import Calibrator
from .calibration_configuration import CalibrationConfiguration, CalibrationConfigurationHelper
from .calibrator_configuration import CalibratorConfiguration
from .calibrator_factory import CalibratorFactory
from .command_line_reader import CommandLineReader
from .command_line_processor import CommandLineProcessor
from .accuracy.metric_factory import MetricFactory

__version__ = "0.0.1"
__all__ = [
    'AggregatedStatistics',
    'Calibrator',
    'CalibrationConfiguration',
    'CalibrationConfigurationHelper',
    'CalibratorConfiguration',
    'CalibratorFactory',
    'CommandLineReader',
    'CommandLineProcessor',
    'MetricFactory'
]
