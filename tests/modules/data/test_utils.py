import string

import dask.dataframe
import numpy
import pandas

from app.modules.data import utils


def test_SplitSampling_split():
    loc = 40
    scale = 60
    size = 10000
    
    numpy.random.seed(0)
    samples = numpy.random.normal(loc=loc, scale=scale, size=int(size))

    categorical_samples = numpy.random.choice(
            numpy.array(['a', 'ab', 'c', 'dce', 'b', '1', '2pq']),
            size=size
        )
    
    alphabet = numpy.array(list(string.ascii_lowercase))
    text_continuous_samples = numpy.array(
            [
                ''.join(numpy.random.choice(alphabet, size=8).tolist()) \
                    for _ in range(0, size)
            ]
        )
    
    pandas_df = pandas.DataFrame(
            {
                'A': samples,
                'B': categorical_samples,
                'C': text_continuous_samples
            }
        )

    dask_df = dask.dataframe.from_pandas(pandas_df, npartitions=1)
    learn, test = utils.SplitSampling(split_ratio=0.6, random_state=0).split(dask_df)
    assert pandas_df.shape[0] == learn.compute().shape[0] + test.compute().shape[0]

    dask_df = dask.dataframe.from_pandas(pandas_df, npartitions=3)
    learn, test = utils.SplitSampling(split_ratio=0.7, random_state=0).split(dask_df)
    assert pandas_df.shape[0] == learn.compute().shape[0] + test.compute().shape[0]

    dask_df = dask.dataframe.from_pandas(pandas_df, npartitions=4)
    learn, test = utils.SplitSampling(split_ratio=0.85, random_state=0).split(dask_df)
    assert pandas_df.shape[0] == learn.compute().shape[0] + test.compute().shape[0]

    dask_df = dask.dataframe.from_pandas(pandas_df, npartitions=7)
    learn, test = utils.SplitSampling(split_ratio=0.6, random_state=0).split(dask_df)
    assert pandas_df.shape[0] == learn.compute().shape[0] + test.compute().shape[0]
