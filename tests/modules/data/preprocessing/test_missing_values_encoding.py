import math
import dask.dataframe
import pandas

from app.modules.data.preprocessing import missing_values_encoding
from cloudsml.models.base_data_transformation_column import BaseDataTransformationColumn


def test_missing_values_decoder():
    pandas_df = pandas.DataFrame(
            {
                1: [1, 2, 2, 4, math.nan, 3.0, math.nan],
                2: ['t', 't', 'O', math.nan, 'a', 'b', 'C']
            }
        )
    dask_df = dask.dataframe.from_pandas(pandas_df, npartitions=1)
    columns_info = {
            1: BaseDataTransformationColumn(
                title=None,
                data_type=None,
                name='A',
                id=1,
                data_transformation=None,
                statistics={ 'missing_values_count' : 2}
            ),
            2: BaseDataTransformationColumn(
                title=None,
                data_type=None,
                name='B',
                id=2,
                data_transformation=None,
                statistics={ 'missing_values_count' : 1},
            )
        }
    dask_df, missing_values_substitution_map = (
            missing_values_encoding.missing_values_encoder(dask_df, columns_info)
        )
    assert dask_df.compute()[1][4] == missing_values_encoding.FILL_MISSING_CONST
    assert dask_df.compute()[2][3] == missing_values_encoding.FILL_MISSING_CONST
    assert missing_values_substitution_map == {
            1: missing_values_encoding.FILL_MISSING_CONST,
            2: missing_values_encoding.FILL_MISSING_CONST
        }
