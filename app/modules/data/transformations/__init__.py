# encoding: utf-8
"""
Data Transformations Helpers Module
"""
from io import StringIO
import ctypes

import inline
import numpy

from cloudsml_computational_backend_common.analysis.consts import PredictiveModelExportFormats

from app.modules.translator import translator as pfa_translator


def transform_dask_dataframe(dataset_dask_df, transformations):
    """
    Transform Dask DataFrame dataset following PFA-described list of
    transformations.

    Args:
        dataset_dask_df (dask.DataFrame): input dataset.
        transformations (list): dataset transformations in PFA format.

    Returns:
        dataset_dask_df (dask.DataFrame): a modified input dataset with applied
        transformations.
    """
    for transformation in transformations:
        if transformation:
            new_meta = dataset_dask_df._meta.copy(deep=False)
            new_column_name = transformation['output']['name']
            try:
                new_column_name = int(new_column_name)
            except ValueError:
                pass
            new_meta[new_column_name] = 0.0
            dataset_dask_df = dataset_dask_df.map_partitions(
                    transform_pandas_dataframe, new_column_name, transformation, meta=new_meta
                )

    return dataset_dask_df


def transform_pandas_dataframe(pandas_df, new_column_name, transformation):
    """
    Helper function to perform transformation
    Args:
        pandas_df (pandas.DataFrame): input dataframe (dask partition)
        transformation (dict): transformation in PFA-format

    Returns:
        pandas_df (pandas.DataFrame): a modified input pandas dataframe with applied
        transformation.
    """
    document_fields_description = transformation['input']['fields']
    assert all(
            field_description['type'] == 'double' or field_description['type'] == ['double', 'null']
                for field_description in document_fields_description
                ), (
                        "Some PFA field records are not of double type:\n%s" %
                        document_fields_description
                   )

    transformation_source_code = StringIO()

    translator = pfa_translator.Translator()
    translator.translate_pfa_document(
            lang=PredictiveModelExportFormats.c,
            output_file=transformation_source_code,
            pfa_document=transformation
        )
    transformation_source_code.seek(0)

    action = inline.c(transformation_source_code.read()).action
    action.restype = ctypes.c_float

    def transform_pandas_row(pandas_row):
        """
        Applies the ``action`` on the given ``pandas_row``
        """
        return numpy.float32(action(ctypes.c_char_p.from_param(pandas_row.values.tobytes())))

    pandas_df = pandas_df.copy(deep=False)

    original_columns = pandas_df.columns
    pandas_df.columns = ['var%s' % column for column in pandas_df.columns]

    input_pandas_df = pandas_df[[field['name'] for field in document_fields_description]]
    assert (input_pandas_df.dtypes == numpy.float32).all(), (
            "Some pandas dataframe columns are not in numpy.float32 dtype:\n%s" \
                % input_pandas_df.dtypes[input_pandas_df.dtypes != numpy.float32]
        )

    new_column = input_pandas_df.apply(
            func=transform_pandas_row,
            axis='columns',
            reduce=True
        ).astype(numpy.float32)

    pandas_df.columns = original_columns
    pandas_df[new_column_name] = new_column
    return pandas_df


def fetch_data_transformations_by_id_in_pfa(data_transformations_list):
    """
    Helper iterator to fetch DataTransformation by it's ID via API
    Args:
        data_transformations_list (list): list of DT ids
    Returns:
        yields DataTransformation in PFA
    """
    from app.extensions import cloudsml

    for data_transformation_id in data_transformations_list:
        data_transformation = cloudsml.data_api.get_export_data_transformation_by_id(
                export_format='pfa',
                data_transformation_id=data_transformation_id
            )
        yield data_transformation
