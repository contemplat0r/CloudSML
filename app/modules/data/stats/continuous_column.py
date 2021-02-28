import numpy

def get_binned_data(column, bins):
    stats, ranges = numpy.histogram(column, bins=bins) 
    return {
            'binning_stats': {
                    'ranges': ranges,
                    'stats': stats
                }
        }


def merge_binned_data(per_column_partitions_stats):
    return {
            'binning_stats': {
                'stats': sum(stats['binning_stats']['stats'] for \
                        stats in per_column_partitions_stats
                    ),
                'ranges': per_column_partitions_stats[0]['binning_stats']['ranges']
            }
        }
