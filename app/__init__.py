# encoding: utf-8
"""
CloudSML Computational Backend
==============================
"""
import logging
import sys

from kuyruk.importer import import_object_str
from kuyruk.worker import Worker


current_app = None  # pylint: disable=invalid-name

CONFIG_NAME_MAPPER = {
    'development': 'config.DevelopmentConfig',
    'production': 'config.ProductionConfig',
    'local': 'local_config.LocalConfig',
}


class Args(object):
    """
    Dummy class to pass it to start kuyruk.Worker from invoke task
    """
    local = False
    queues = ['kuyruk']
    logging_level = None
    max_run_time = None
    max_load = None


def create_app(config_name='local', queue_name=''):
    """
    Entry point to the CloudSML Computational Backend worker.
    """
    from cloudsml_computational_backend_common import kuyruk

    assert config_name in CONFIG_NAME_MAPPER

    try:
        config = import_object_str(CONFIG_NAME_MAPPER[config_name])
    except ImportError:
        if config_name == 'local':
            logging.error(
                "You have to have `local_config.py` or `local_config/__init__.py` in order to use "
                "the default 'local' Flask Config."
            )
            sys.exit(1)
        raise

    kuyruk.config.from_dict(config.KUYRUK_CONFIG)

    args = Args()
    args.queues = [queue_name]

    global current_app  # pylint: disable=global-statement,invalid-name
    current_app = app = Worker(kuyruk, args)
    app.config = config

    from . import extensions
    extensions.init_app(app)

    return app
