# encoding: utf-8

"""
All text column statistics computing
Text column format-specific merge functions.
-------------------------------------------------
"""


def compute_column_stats_for_text_format(column):
    return {
            'min_len': len(min(column, key=len)),
            'max_len': len(max(column, key=len))
        }


def merge_text_column_min_len_max_len(text_stats):
    """
    Merge min and max for whole column (for all partitions).
    """
    return (
            min(stats['min_len'] for stats in text_stats),
            max(stats['max_len'] for stats in text_stats)
    )


def merge_text_column_partitions_stats(per_column_partitions_stats):
    """
    Collect all merged stats computed by other functions.
    """
    text_stats = [stats['format_specific_statistics']\
        for stats in per_column_partitions_stats]
    merged_min_len, merged_max_len = merge_text_column_min_len_max_len(text_stats)
    return {'min_len': merged_min_len, 'max_len': merged_max_len}
