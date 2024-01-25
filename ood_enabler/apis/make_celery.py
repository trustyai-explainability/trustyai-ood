"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20230824
© Copyright IBM Corp. 2024 All Rights Reserved.
"""
import os
from celery import Celery

CELERY_BROKER_URL = os.environ.get('BROKER_URL') or 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = os.environ.get('RESULT_BACKEND') or 'redis://localhost:6379/0'


def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=CELERY_RESULT_BACKEND,
        broker=CELERY_BROKER_URL
    )
    celery.conf.update(app.config,
                       accept_content=['json'])

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
