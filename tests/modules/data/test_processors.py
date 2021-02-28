# encoding: utf-8

import tempfile

import mock
import pytest

from cloudsml_computational_backend_common.data.consts import DatasetExportFormats, DataFormats

from app.modules.data import processors
from app.extensions import seaweedfs, cloudsml


def test_extract_dataset_info_first_load(
        cb_app,
        boston_dataset
    ):
    def patched_call_api(*args, **kwargs):
        patched_call_api.call_count = getattr(patched_call_api, 'call_count', 0) + 1
        if args[0] == '/data/transformations/{data_transformation_id}':
            return cloudsml.models.DetailedDataTransformation(
                    transformation_type='blank'
                )
        return True
    with mock.patch.object(
                cloudsml.data_api.api_client, 'call_api', patched_call_api
            ) as mocked_call:
        processors.extract_dataset_info('file://' + boston_dataset['path'], 1)
        assert mocked_call.call_count == 16


@pytest.mark.parametrize('generate_dataframe', [('pandas', None)], indirect=True)
def test_extract_dataset_info_uniques_stats_key(
        cb_app,
        generate_dataframe
    ):
    def patched_call_api(*args, **kwargs):
        patched_call_api.call_count = getattr(patched_call_api, 'call_count', 0) + 1
        if args[0] == '/data/transformations/{data_transformation_id}':
            return cloudsml.models.DetailedDataTransformation(
                    transformation_type='blank'
                )
        return True
    with mock.patch.object(
                cloudsml.data_api.api_client, 'call_api', patched_call_api
            ) as mocked_call:
        csv_file = tempfile.NamedTemporaryFile(mode='w', suffix=".csv")
        generate_dataframe.to_csv(csv_file.name)

        processors.extract_dataset_info('file://' + csv_file.name, 1)
        assert mocked_call.call_count == 6


@mock.patch.object(seaweedfs, 'upload_file')
def test_export_dataset(
        mock_upload_file,
        cb_app,
        boston_dataset
    ):
    mock_upload_file.return_value = True

    def patched_call_api(*args, **kwargs):
        patched_call_api.call_count = getattr(patched_call_api, 'call_count', 0) + 1
        if args[0] == '/data/transformations/{data_transformation_id}/columns/':
            return [
                    cloudsml.models.BaseDataTransformationColumn(
                            name=column_name,
                            data_format=DataFormats.numerical
                        ) for column_name in boston_dataset['dataframe'].columns
                ]
        elif args[0] == '/data/transformations/{data_transformation_id}.{export_format}':
            return {}
        return True

    with mock.patch.object(
                cloudsml.data_api.api_client, 'call_api', patched_call_api
            ) as mocked_call:
        processors.export_dataset(
                'file://' + boston_dataset['path'],
                [1],
                DatasetExportFormats.JSON,
                1
            )
        assert mocked_call.call_count == 3
        assert mock_upload_file.called
