# encoding: utf-8

import functools

import dask.dataframe
import numpy
import pandas

from cloudsml_computational_backend_common.data.consts import DataFormats, DataTypes


class CategoriesUnifier(object):
    """
    Updates all the partitions in the Dask DataFrame to have the same category
    codes everywhere.
    """

    def __init__(self, categories_by_column):
        """
        Args:
            categories_by_column (dict): a dictionary holding categories mappings
                by column names.
        """
        self.categories_by_column = categories_by_column

    def update_partition(self, df):
        for column_name, all_column_categories in self.categories_by_column.items():
            column = df[column_name]
            new_column = column.cat.codes.astype(numpy.int16, copy=False)
            new_column.replace(
                    to_replace={
                        code: all_column_categories.get_loc(category) \
                            for code, category in enumerate(column.cat.categories)
                    },
                    inplace=True
                )
            df[column_name] = new_column
        return df

    def update(self, dask_df):
        """
        Args:
            dask_df (dask.dataframe.DataFrame): input DataFrame with native
                categorical cloumns.

        Returns:
            dask.dataframe.DataFrame: output DataFrame with all the categorical
                columns replaced with plain ``int16`` columns where the
                values (category codes) are unified across all the paritions.
        """
        new_meta = dask_df._meta.copy()
        for column in new_meta:
            if isinstance(new_meta[column].dtype, pandas.core.dtypes.dtypes.CategoricalDtype):
                new_meta[column] = pandas.Series(dtype=numpy.int16)

        return dask_df.map_partitions(
                self.update_partition,
                meta=new_meta
            )


def dask_universal_read(dataset_url, columns_info):
    """
    Read any dataset by a given URL (``dataset_url``).

    Args:
        dataset_url (str): a URL pointing to the dataset, e.g.
            * ``s3://my-bucket/my_data.csv``
            * ``file:///home/user/my_data.csv``
            * ``hdfs://127.0.0.1:1234/user/my_data.csv``
        columns_info (list): a list of :class:`BaseDataTransformationColumn`-like
            objects containing the column ``id``, ``name``, ``data_type``, and
            ``data_format``.

    Returns:
        dask.dataframe.DataFrame: an evaluated Dask DataFrame holding the
        requested dataset.
    """
    print(columns_info)
    if dataset_url.lower().endswith('.csv'):
        reader = functools.partial(dask.dataframe.read_csv)
    else:
        raise NotImplementedError(
                "Cannot read the file '%s'. The file extension is not supported"
                % dataset_url
            )

    categories_by_column = {}
    kwargs = {}
    if columns_info is None:
        kwargs['dtype'] = str
    else:
        kwargs['names'] = [column.id for column in columns_info]
        kwargs['usecols'] = [
                column_index for column_index, column in enumerate(columns_info) \
                    if column.data_format in (DataFormats.numerical, DataFormats.character)
            ]
        kwargs['header'] = 0
        def get_column_dtype(column):
            if column.data_format == DataFormats.numerical:
                return numpy.float32
            if column.data_type in (DataTypes.categorical, DataTypes.binary):
                return pandas.core.dtypes.dtypes.CategoricalDtype()

        dtypes = {}
        for column in columns_info:
            dtypes[column.id] = column_dtype = get_column_dtype(column)
            print('\n', column.name, column_dtype)
            if isinstance(column_dtype, pandas.core.dtypes.dtypes.CategoricalDtype):
                print('\n', column.name, 'isinstance pandas.core.dtypes.dtypes.CategoricalDtype')
                categories_by_column[column.id] = pandas.Index(
                        unique_value for unique_value, _ in column.statistics['uniques_stats']
                    )
            else:
                print('\n', column.name, 'notinstance pandas.core.dtypes.dtypes.CategoricalDtype')

        kwargs['dtype'] = dtypes
        print('\n kwargs: ', kwargs)
    dask_dataframe = reader(dataset_url, **kwargs)
    print(dask_dataframe.compute().head())

    if categories_by_column:
        categories_unifier = CategoriesUnifier(categories_by_column)
        dask_dataframe = categories_unifier.update(dask_dataframe)
    return dask_dataframe
