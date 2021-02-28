# encoding: utf-8
"""
Analysis Helper Processors
"""
import logging
from io import StringIO

from cloudsml_computational_backend_common.analysis.schemas import ModelInfoShema

from app.extensions import cloudsml
from app.extensions import seaweedfs
from app.modules.data.readers import dask_universal_read
from app.modules.data.transformations import (
        fetch_data_transformations_by_id_in_pfa,
        transform_dask_dataframe,
    )
from app.modules.data.utils import SplitSampling
from app.modules.translator import translator
from app.modules.data.preprocessing import one_hot_encoding, missing_values_encoding
from app.utils import json
from .postprocessing.pfa.one_hot_pfa_decoder import OneHotPFADecoder
from .postprocessing.pfa.missing_values_pfa_decoder import MissingValuesPFADecoder
from .predictive_analysis import PREDICTIVE_ANALYSIS_METHODS


log = logging.getLogger(__name__)


def build_predictive_model(
        learn_dataset_url,
        dataset_transformations,
        predictive_analysis_method_name,
        predictive_analysis_options,
        predictive_model_id
    ):
    """
    This function is building predictive model, saves it to SeaWeedFS and patching through API
    the predictive model by it's id, putting there a seaweedfs' id to model and to model info.

    Args:
        learn_dataset_url (str): link to a learn dataset.
        dataset_transformations (list): a list of data transformation ids of
            the learn dataset in PFA format.
        predictive_analysis_method_name (str): name of the predictive method
            that is requested to be used.
        predictive_analysis_options (dict): kwargs to predictive analysis
            method.
        predictive_model_id (int): id of the model in API to patch it
    """
    log.info(
            "New %s model #%d is going to be built...",
            predictive_analysis_method_name,
            predictive_model_id
        )

    initial_columns_info = cloudsml.data_api.get_data_transformation_columns(
            dataset_transformations[-1],
            initial=True,
            # XXX: We must read the list in batches when the number of columns exceeds 1000
            limit=1000
        )

    learn_dataset_df = dask_universal_read(learn_dataset_url, columns_info=initial_columns_info)
    learn_dataset_df = transform_dask_dataframe(
            learn_dataset_df,
            fetch_data_transformations_by_id_in_pfa(dataset_transformations)
        )

    target_column_id = predictive_analysis_options['target_column_id']
    feature_column_ids = predictive_analysis_options['feature_column_ids']
    selected_columns_info = {
            column.id: column \
                for column in cloudsml.data_api.get_data_transformation_columns(
                        dataset_transformations[-1],
                        id=([target_column_id] + feature_column_ids),
                        # XXX: API server limits the maximum possible limit of columns per single
                        # request at 1000 to avoid too long response times. Thus, we must implement
                        # querying in the columns info in batches. Yet, this might be hidden behind
                        # a convinient wrapper.
                        limit=1000
                    )
        }
    learn_dataset_df = learn_dataset_df[sorted(selected_columns_info.keys())]

    missing_values_encoder = missing_values_encoding.missing_values_encoder
    learn_dataset_df, missing_values_substitution_map = missing_values_encoder(
            learn_dataset_df,
            selected_columns_info
        )

    learn_dataset_df, selected_columns_info = one_hot_encoding.OneHotEncoder(
            categorical_columns_ids=predictive_analysis_options['categorical_column_ids'],
            columns_info=selected_columns_info
        ).update(learn_dataset_df, selected_columns_info)

    test_partition_ratio = predictive_analysis_options.get('test_partition_ratio', 0.4)
    test_learn_splitter = SplitSampling(test_partition_ratio, random_state=0)
    test_dataset_df, learn_dataset_df = test_learn_splitter.split(learn_dataset_df)

    predictive_analysis_method = PREDICTIVE_ANALYSIS_METHODS[predictive_analysis_method_name]
    log.info('Model #%d is being fitted with data...', predictive_model_id)
    model = predictive_analysis_method(
            learn_dataset_df,
            columns_info=selected_columns_info,
            **predictive_analysis_options
        )

    log.info('Model #%d is being exported to PFA...', predictive_model_id)
    one_hot_pfa_decoder = OneHotPFADecoder({
            column.id: column.virtual_columns \
                for column in selected_columns_info.values() \
                    if hasattr(column, 'virtual_columns')
        })
    missing_values_pfa_decoder = MissingValuesPFADecoder(missing_values_substitution_map)

    translated_model = missing_values_pfa_decoder.transform(
            one_hot_pfa_decoder.transform(model.to_pfa())
        )

    model_file_id = seaweedfs.upload_file(
        stream=json.dumps(translated_model),
        name='model_%s.pfa' % predictive_model_id
    )

    log.info('Model #%d information is being collected...', predictive_model_id)
    model_info = {
            'learn': model.get_info(learn_dataset_df),
        }
    if test_partition_ratio > 0.0:
        model_info['test'] = model.get_info(test_dataset_df)

    model_info = ModelInfoShema().load(
            {'performance_stats': model_info}
        ).data

    model_info_id = seaweedfs.upload_file(
        stream=json.dumps(model_info),
        name='model_info_%s.json' % predictive_model_id
    )

    cloudsml.predictive_analytics_api.patch_predictive_model_by_id(
        predictive_model_id,
        [
            {"op": "replace", "path": "/model_seaweed_id", "value": model_file_id},
            {"op": "replace", "path": "/model_info_seaweed_id", "value": model_info_id},
            {"op": "replace", "path": "/status", "value": "fitted"},  # TODO use constant here
        ]
    )


def export_model(predictive_model_id, export_format, dataset_transformations, exported_model_id):
    """
    Exports model to specified ``export_format``, saves it to SeaweedFS and updated
    ExportedPredictiveModel by ``exported_model_id`` with SeaweedFS file id.

    Args:
        predictive_model_id (int): ID of PredictiveModel to export
        export_format (str): Format to export to
        dataset_transformations (list): List of DataTransformations' IDs
        exported_model_id (int): ID of instance ExportedPredictiveModel
            to update info about exporting
    """
    #TODO Implement real model export in PFA
    #TODO Implement usage of ``dataset_transformations``
    predictive_model_pfa = cloudsml.predictive_analytics_api.get_predictive_model_export_by_id(
            predictive_model_id=predictive_model_id,
            export_format='model.pfa'
        )

    translator_instance = translator.Translator()
    model = StringIO()
    translator_instance.translate_pfa_document(
            lang=export_format,
            output_file=model,
            pfa_document=predictive_model_pfa
        )
    model.seek(0)
    model_file_id = seaweedfs.upload_file(
        stream=model,
        name="exported_model_{export_task_id}.{export_format}".format(
            export_task_id=exported_model_id,
            export_format=export_format.value
        )
    )
    cloudsml.predictive_analytics_api.patch_exported_predictive_model_by_id(
        exported_model_id,
        [
            {"op": "replace", "path": "/status", "value": "finished"},
            {"op": "replace", "path": "/model_seaweed_id", "value": model_file_id}
        ]
    )


def build_predictive_pipeline(
        pipeline_id,
        pipeline_options,
        learn_dataset_url,
        dataset_transformations,
    ):
    """
    Reads ``pipeline_options`` to retrieve parameters for building a bunch of models.
    Updates Pipeline through the API to change the status and models_count
    """
    models_count = (
            len(pipeline_options['target_column_ids'])
            * len(pipeline_options['predictive_analysis_options'])
        )
    pipeline = cloudsml.predictive_analytics_api.patch_pipeline_by_id(
            pipeline_id=pipeline_id,
            body=[
                    {'op': 'replace', 'path': '/models_count', 'value': models_count}
                ]
        )
    predictive_models_to_build = []
    pipeline_model_index = 0
    static_feature_column_ids = [
            int(feature_column_id) \
                for feature_column_id, feature_column_options in (
                        pipeline_options['features'].items()
                    ) if feature_column_options['type'] == 'static'
        ]
    # TODO: extend the support for cross-validation
    test_partition_ratio = pipeline_options['testing_settings']['test_partition_ratio']
    for target_column_id in pipeline_options['target_column_ids']:
        for (
                method_name,
                method_parameters
            ) in pipeline_options['predictive_analysis_options'].items():
            pipeline_model_index += 1
            categorical_column_ids = (
                    set(pipeline_options['categorical_column_ids'])
                    & set(static_feature_column_ids)
                )
            predictive_analysis_options = dict(
                    target_column_id=target_column_id,
                    # TODO: implement optional and random predictors selection.
                    feature_column_ids=static_feature_column_ids,
                    categorical_column_ids=categorical_column_ids,
                    # TODO: extend the support for cross-validation
                    test_partition_ratio=test_partition_ratio,
                    method_parameters=method_parameters
                )
            predictive_model = cloudsml.predictive_analytics_api.fit_predictive_model(
                    predictive_analysis_method=method_name,
                    name='{pipeline_name} #{pipeline_model_index}'.format(
                            pipeline_name=pipeline.name,
                            pipeline_model_index=pipeline_model_index
                        ),
                    predictive_analysis_options=json.dumps(predictive_analysis_options),
                    data_transformation_id=dataset_transformations[-1],
                    pipeline_id=pipeline_id
                )
            predictive_models_to_build.append(
                    {
                        'predictive_analysis_options': predictive_analysis_options,
                        'predictive_analysis_method_name': method_name,
                        'predictive_model_id': predictive_model.id,
                    }
                )

    for predictive_model_params in predictive_models_to_build:
        build_predictive_model(
                learn_dataset_url=learn_dataset_url,
                dataset_transformations=dataset_transformations,
                **predictive_model_params
            )

    cloudsml.predictive_analytics_api.patch_pipeline_by_id(
            pipeline_id=pipeline_id,
            body=[
                {'op': 'replace', 'path': '/status', 'value': 'ready'}
            ]
        )
