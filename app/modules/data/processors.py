# encoding: utf-8
"""
Data processors
---------------
"""
import logging

from cloudsml_computational_backend_common.data.consts import DatasetExportFormats

from app.extensions import seaweedfs
from app.utils import json
from .readers import dask_universal_read
from .stats import extract_dataframe_info
from .transformations import (
        fetch_data_transformations_by_id_in_pfa,
        transform_dask_dataframe,
    )


log = logging.getLogger(__name__)


def extract_dataset_info(
        dataset_url,
        data_transformation_id,
        dataset_transformations=None
    ):
    """
    Select reader function. Data loaded in dask DataFrame.
    Retrieve info from dask DataFrame
    """
    from app.extensions import cloudsml
    data_transformation = cloudsml.data_api.get_data_transformation_by_id(data_transformation_id)
    if data_transformation.transformation_type == 'blank':
        columns_info = None
    elif data_transformation.transformation_type == 'add_column':
        columns_info = cloudsml.data_api.get_data_transformation_columns(
                data_transformation_id,
                initial=True,
                # XXX: We must read the list in batches when the number of columns exceeds 1000
                limit=1000
            )
    else:
        raise NotImplementedError(
                "Transformation type '%s' cannot be handled"
                % data_transformation.transformation_type
            )

    dataset_df = dask_universal_read(dataset_url, columns_info=columns_info)
    if dataset_transformations:
        dataset_df = transform_dask_dataframe(
                dataset_df,
                fetch_data_transformations_by_id_in_pfa(dataset_transformations)
            )

    if data_transformation.transformation_type == 'add_column':
        dataset_df = dataset_df[['dt%s' % (data_transformation.id)]]

    dataframe_info = extract_dataframe_info(dataset_df)
    per_column_statistic = dataframe_info['per_column_statistic']

    for index, column_name in enumerate(dataset_df.columns):
        assert column_name in per_column_statistic
        if data_transformation.transformation_type == 'blank':
            column_title = column_name
        else:
            if len(per_column_statistic) == 1:
                column_title = data_transformation.name
            else:
                column_name = '%s__%d' % (column_name, index)
                column_title = '%s #%d' % (data_transformation.name, index)
        cloudsml.data_api.create_data_transformation_column(
                data_transformation_id,
                name=column_name,
                title=column_title,
                statistics=json.dumps(per_column_statistic[column_name]),
                data_type=per_column_statistic[column_name]['type'].name,
                data_format=per_column_statistic[column_name]['format'].name,
            )
    cloudsml.data_api.patch_data_transformation_by_id(
            data_transformation_id,
            [
                {'op': 'replace', 'path': '/status', 'value': 'ready'},
                {'op': 'replace', 'path': '/rows_count', 'value': dataframe_info['rows_count']}
            ]
        )


def export_dataset(
        dataset_url,
        dataset_transformations,
        export_format,
        export_dataset_id,
        **extra_options
    ):
    """
    Exports dataset by ``dataset_url`` to specified ``export_format``
    """
    from app.extensions import cloudsml

    columns_info = cloudsml.data_api.get_data_transformation_columns(
            dataset_transformations[-1],
            initial=True,
            # XXX: We must read the list in batches when the number of columns exceeds 1000
            limit=1000
        )
    exporting_df = dask_universal_read(dataset_url, columns_info=columns_info)
    exporting_df = transform_dask_dataframe(
            exporting_df,
            fetch_data_transformations_by_id_in_pfa(dataset_transformations)
        )

    columns = extra_options.pop('columns', None)
    if columns:
        exporting_df = exporting_df[columns]
    offset = extra_options.pop('offset', 0)
    if offset > 0:
        # We don't have any "index" column to actually perform offsetting
        # correctly (receiving the same result every time we query Dask).
        # Dask does not guarantee any order of the records outside of
        # partitions unless you set an index column, which is an expensive
        # operation.
        raise NotImplementedError("offset > 0 is not supported yet")
    limit = extra_options.pop('limit', None)
    if limit:
        exporting_df = exporting_df.head(limit, compute=False)

    # XXX Implement a stream instead of .compute()
    if export_format is DatasetExportFormats.JSON:
        exported_dataset = exporting_df.compute().to_json(orient='records')
    elif export_format is DatasetExportFormats.CSV:
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    dataframe_file_id = seaweedfs.upload_file(
            stream=exported_dataset,
            name="exported_dataset_{export_dataset_id}.{export_format}".format(
                export_dataset_id=export_dataset_id,
                export_format=export_format.value
            )
        )

    cloudsml.data_api.patch_materialized_data_transformation_by_id(
            materialized_data_transformation_id=export_dataset_id,
            body=[
                {
                    "op": "replace",
                    "path": "/data_transformation_seaweed_id",
                    "value": dataframe_file_id
                },
                {
                    "op": "replace",
                    "path": "/status",
                    "value": "finished"
                }
            ]
        )
