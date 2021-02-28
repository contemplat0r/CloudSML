"""
Utils for data module
=====================
"""
import operator

import dask.bag
import dask.dataframe
from sklearn.model_selection import train_test_split


class SplitSampling(object):
    """
    Split the input Dask DataFrame into non-overlapping samples.
    """
    __slots__ = (
            'split_ratio',
            'random_state',
        )

    def __init__(self, split_ratio, random_state=None):
        """
        Arguments:
            split_ratio (float): ratio at which the samples will be splitted;
                if should be in range of [0.0; 1.0].
            random_state (int): Pseudo-random number generator state used for
                random sampling.
        """
        assert 0.0 <= split_ratio <= 1.0
        self.split_ratio = split_ratio
        self.random_state = random_state

    def _split_partition(self, partition_df):
        return train_test_split(
                partition_df,
                test_size=self.split_ratio,
                random_state=self.random_state
            )

    def _split(self, delayed_partitions):
        return [
                dask.delayed(self._split_partition)(delayed_partition_df) \
                    for delayed_partition_df in delayed_partitions
            ]

    def _extract(self, splitted_partitions, partition_index, meta):
        return dask.dataframe.from_delayed(
                [dask.delayed(operator.itemgetter(partition_index))(splitted_partition) \
                    for splitted_partition in splitted_partitions],
                meta=meta
            )

    def split(self, dask_df):
        """
        Arguments:
            dask_df (dask.dataframe.DataFrame): input dataframe.

        Returns:
            list: a list of Dask DataFrames
        """
        if self.split_ratio == 0.0:
            return (None, dask_df)
        if self.split_ratio == 1.0:
            return (dask_df, None)

        splitted_partitions = self._split(dask_df.to_delayed())
        return (
                self._extract(splitted_partitions, partition_index=0, meta=dask_df._meta),
                self._extract(splitted_partitions, partition_index=1, meta=dask_df._meta),
            )
