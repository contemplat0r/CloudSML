import numpy
import pandas

from app.modules.data.stats import continuous_column

def test_get_binned_data():
    numpy.random.seed(0)
    column = pandas.Series(numpy.random.normal(loc=40, scale=60, size=2000))
    bins = numpy.linspace(column.min(), column.max(), 20)
    binned_data = continuous_column.get_binned_data(column, bins=bins)
    assert all(binned_data['binning_stats']['ranges'] == bins)
    assert all(binned_data['binning_stats']['stats'] == numpy.array(
                [
                    7, 8, 17, 39, 88, 112, 190, 210, 256, 284, 250,
                    178, 142, 96, 61, 32, 21, 8, 1
                ]
            )
        )
    
def test_merge_binned_data_not_intersect():
    loc = 40
    scale = 60
    size = 4000
    num_bins = 20
    column_samples = numpy.random.normal(loc=loc, scale=scale, size=size)
    one = numpy.random.choice(column_samples, size=int(3 * size / 4), replace=False)
    two = numpy.array([item for item in column_samples if item not in one])
    column = pandas.Series(column_samples)
    part_one = pandas.Series(one)
    part_two = pandas.Series(two)
    bins = numpy.linspace(column.min(), column.max(), num_bins)
    column_binned_data = continuous_column.get_binned_data(column, bins=bins)
    binned_data_one = continuous_column.get_binned_data(part_one, bins=bins)
    binned_data_two = continuous_column.get_binned_data(part_two, bins=bins)
    merged_binned_data = continuous_column.merge_binned_data(
            [
                binned_data_one,
                binned_data_two
            ]
        )
    assert all(
            column_binned_data['binning_stats']['stats'] == merged_binned_data['binning_stats']['stats']
        )

def test_merge_binned_data_intersect():
    loc = 40
    scale = 60
    size = 4000
    num_bins = 20
    numpy.random.seed(0)
    part_one_samples = numpy.random.normal(loc=loc, scale=scale, size=int(size / 2))
    numpy.random.seed(10)
    part_two_samples = numpy.random.normal(loc=loc, scale=scale, size=int(size / 2))
    part_one = pandas.Series(part_one_samples)
    part_two = pandas.Series(part_two_samples)
    column = pandas.concat([part_one, part_two])
    bins = numpy.linspace(column.min(), column.max(), num_bins)
    column_binned_data = continuous_column.get_binned_data(column, bins=bins)
    binned_data_one = continuous_column.get_binned_data(part_one, bins=bins)
    binned_data_two = continuous_column.get_binned_data(part_two, bins=bins)
    merged_binned_data = continuous_column.merge_binned_data(
            [
                binned_data_one,
                binned_data_two
            ]
        )
    assert all(
            column_binned_data['binning_stats']['stats'] == merged_binned_data['binning_stats']['stats']
        )
