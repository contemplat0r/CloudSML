# encoding: utf-8
"""
All numerical column statistics computing.
Numerical column format-specific merge functions.
-------------------------------------------------
"""

import math

import numpy

def compute_column_stats_for_numerical_format(column):
    column_min = float(column.min())
    column_max = float(column.max())
    if math.isnan(column_min) and math.isnan(column_max):
        return {}
    return {
            'min': column_min,
            'max': column_max,
            'sum': float(column.sum())
        }


def merge_numerical_column_stats(numerical_stats):
    numerical_stats = [stats for stats in numerical_stats if stats]
    if numerical_stats:
        return {
                'min': min(stats['min'] for stats in numerical_stats),
                'max': max(stats['max'] for stats in numerical_stats),
                'sum': sum(stats['sum'] for stats in numerical_stats)
            }
    return {}


def merge_numerical_column_partitions_stats(per_column_partitions_stats):
    """
    Collect all merged stats computed by other functions.
    """
    return merge_numerical_column_stats(
            stats['format_specific_statistics'] for stats in per_column_partitions_stats
        )

