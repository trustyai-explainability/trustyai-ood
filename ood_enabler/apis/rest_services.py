/*
 * © Copyright IBM Corp. 2024, and/or its affiliates. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the “License”);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import os
import sys
import argparse
import requests
import ssl
import threading
from tempfile import TemporaryDirectory
from flask import Flask, request, jsonify, abort, url_for
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin

ood_path = os.path.abspath('.')
if ood_path not in sys.path:
    sys.path.append(ood_path)

from ood_enabler.ood_enabler import OODEnabler
from ood_enabler.apis.swagger_generation import generate_api_spec
from ood_enabler.apis.make_celery import make_celery
from ood_enabler.storage.model_store import ModelStore
from ood_enabler.storage.cos_storage import COSStorage
from ood_enabler.storage.local_storage import FileSystemStorage
from ood_enabler.data.factory import get_image_data_handler
from ood_enabler.util.constants import SavedModelFormat

#log = logging.getLogger('werkzeug')
#log.setLevel(logging.ERROR)
#log.disabled = True

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
celery = make_celery(app)
CELERY_ACCEPT_CONTENT = ['pickle']
CELERY_TASK_SERIALIZER = 'pickle'
CELERY_RESULT_SERIALIZER = 'pickle'

spec = APISpec(
    title="OOD Enabler",
    version="1.0",
    openapi_version="2.0",
    info=dict(description="REST APIs for Enabling OOD for Pytorch/Tensorflow models"),
    plugins=[FlaskPlugin(), MarshmallowPlugin()],
)


@app.route('/ood_enable', methods=['POST'])
def ood_enable():
    """Embeds OOD layer into provided model for for producing model certainty score
    ---
    post:
      description: Embeds OOD layer into provided model for for producing model certainty score
      parameters:
        - in: body
          schema: OODEnableSchema
          name: OOD Enable Params
      responses:
        202:
          description: Returns task ID and status url for monitoring ood enable request
        500:
            description: Returns error message logged by server

    """
    if request.method == 'POST':
        args = request.get_json()
        model_ref = args['model_ref']
        output_ref = args['output_ref']
        data_ref = args.get('data_ref')

        task = ood_enable_long_task.delay(model_ref, output_ref, data_ref)

        return jsonify({"task_id": task.id, 'status_url': '/ood_enable/status/{}'.format(task.id)}), 202, \
               {'Location': url_for('task_status', task_id=task.id)}


@celery.task(name="ood_enable.long_task", bind=True)
def ood_enable_long_task(self, model_ref, output_ref, data_ref):
    """
    Task for embedding OOD layer into provided model ref, and saving to output ref

    :param model_ref: reference to model storage location and metadata
    :type model_ref: `dict`
    :param output_ref: reference where to store ood enabled model
    :type output_ref: `dict`
    :param data_ref: reference to in-distribution dataset's storage location for normalizing ood layer (optional)
    :type data_ref: `dict`
    :return: Location of where output model was stored
    :rtype: `dict`
    """
    self.update_state(state='PROGRESS', meta={'status': "enabling ood model..."})
    model_wrapper = get_model_ref(model_ref)
    data_handler = None

    with TemporaryDirectory() as tmpdir:
        if data_ref is not None:
            data_handler = get_data_ref(data_ref, model_ref['metadata']['type'], tmpdir)

        # TODO: check for kserve settings in environment to use for inference service
        OODEnabler.ood_enable(model_wrapper, data_handler)

        self.update_state(state='PROGRESS', meta={'status': "saving ood enabled model..."})

        output = save_ood_model(output_ref, model_wrapper)

        return {'status': 'Task completed! model saved at {}'.format(output)}


def get_model_ref(model_ref):
    """
    Gets model from provided provided model refeference

    :param model_ref: reference to model storage location and metadata
    :type model_ref: `dict`
    :return: model wrapper
    :rtype: `model_wrapper.ModelWrapper`
    """
    model_metadata = model_ref['metadata']

    if 'ibm_cos' in model_ref['location']:
        print("getting cos")
        cos_credentials = model_ref['location']['ibm_cos']
        model_store = ModelStore.from_cos(bucket=cos_credentials['bucket'],
                                          api_key=cos_credentials['api_key'],
                                          service_instance_id=cos_credentials['resource_instance_id'],
                                          endpoint=cos_credentials['service_endpoint'],
                                          auth_endpoint=cos_credentials['auth_endpoint'])

        return model_store.load(model_metadata, cos_credentials['file_path'])

    else:
        # uri is given
        uri = model_ref['location']['uri']
        r = requests.get(uri)
        print("getting the uri since no cos")
        if r.ok:
            with TemporaryDirectory() as tmpdir:
                filename = os.path.join(tmpdir, uri.split('/')[-1])
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=128):
                        f.write(chunk)

                model_store = ModelStore.from_filesystem()

                return model_store.load(model_metadata, filename)

        else:
            abort(500, "Error retrieving model from URI")


def get_data_ref(data_ref, model_type, destination):
    """
    Gets datahandler from provided dataset reference

    :param data_ref: reference to dataset storage location and metadata
    :type data_ref: `dict`
    :param model_type: the type of ml backend to use for data
    :type model_type: `str`
    :param destination: local path to store retrieved data for loading
    :type destination: `str`
    :return: datahandler
    :rtype: `data.DataHandler`
    """
    ds_metadata = data_ref['metadata']
    print('getting data')
    if 'ibm_cos' in data_ref['location']:
        cos_credentials = data_ref['location']['ibm_cos']
        cos = COSStorage(bucket=cos_credentials['bucket'],
                         api_key=cos_credentials['api_key'],
                         service_instance_id=cos_credentials['resource_instance_id'],
                         endpoint=cos_credentials['service_endpoint'],
                         auth_endpoint=cos_credentials['auth_endpoint'])

        return get_image_data_handler(model_type, ds_metadata, cos, cos_credentials['file_path'], destination)

    else:
        # uri is given
        uri = data_ref['location']['uri']
        r = requests.get(uri)

        if r.ok:
            filename = os.path.join(destination, uri.split('/')[-1])
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk)

            local_store = FileSystemStorage()

            return get_image_data_handler(model_type, ds_metadata, local_store, filename, destination)

        else:
            abort(500, "Error retrieving dataset from URI")


def save_ood_model(output_ref, ood_model):
    """
    Saves OOD enabled model to location specified in output reference
    :param output_ref: reference to output storage location
    :type output_ref: `dict`
    :param ood_model: wrapper around ood enabled model
    :type ood_model: `model_wrapper.Model`
    :return: location where ood model was saved
    :rtype: `str`
    """
    save_format = SavedModelFormat.NATIVE
    if 'save_format' in output_ref:
        if output_ref['save_format'] == 'onnx':
            save_format = SavedModelFormat.ONNX
            print('using onnx save format')

        else:
            print('using native')

    if 'ibm_cos' in output_ref['location']:
        print("saving with cos")
        cos_credentials = output_ref['location']['ibm_cos']
        model_store = ModelStore.from_cos(bucket=cos_credentials['bucket'],
                                          api_key=cos_credentials['api_key'],
                                          service_instance_id=cos_credentials['resource_instance_id'],
                                          endpoint=cos_credentials['service_endpoint'],
                                          auth_endpoint=cos_credentials['auth_endpoint'])

        return model_store.upload(ood_model, cos_credentials['file_path'], saved_model_format=save_format)

    # TODO: check out to upload object with presigned URL; should be PUT request


@app.route('/ood_enable/status/<task_id>', methods=['GET'])
def task_status(task_id):
    """Checks the status of an initiated ood_enable operation
        ---
        get:
          description: Checks the status of an initiated ood_enable operation
          parameters:
            - in: path
              name: task_id
              type: string
          responses:
            200:
              description: Returns a json of the state and status of the task
            500:
                description: Returns error message logged by server

        """
    task = ood_enable_long_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }

    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'status': task.info.get('status', '')
        }

    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'status': str(task.info),  # this is the exception raised
        }

    return jsonify(response)


@app.route("/connection_check", methods=["GET"])
def connection_check():
    """
        Returns 'ok' status to ensure server is running
        :return: 200 status code
    """
    return jsonify({'status': 'ok'}, 200)


def run_flask_app(cert=None, key=None, port=8443):
    if cert is not None and key is not None:
        context = ssl.SSLContext()
        context.load_cert_chain(cert, key)
        app.run(host='0.0.0.0', port=port, debug=False,  ssl_context=context)

    else:
        app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cert', help='Path to cert file for ssl')
    parser.add_argument('--key', help='Path to key file for ssl')
    parser.add_argument('--api-spec', help='Path to generate a swagger json file for the APIs')
    parser.add_argument('--port', help='port to start rest server', default=8443)
    args = parser.parse_args()

    if args.api_spec is not None:
        generate_api_spec(args.api_spec, app, spec, views=[ood_enable])

    else:
        t1 = threading.Thread(target=run_flask_app, args=(args.cert, args.key, args.port))
        t1.start()
