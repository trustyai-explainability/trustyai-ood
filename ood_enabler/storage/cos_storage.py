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
from ibm_botocore.client import Config
import ibm_boto3
import ood_enabler.settings as s
from ood_enabler.storage.storage import Storage
import uuid


class COSStorage(Storage):
    """
    Class to store/retrieve assets from the IBM's Cloud Object Store (COS)

    """
    def __init__(self, bucket=None, api_key=None, service_instance_id=None, endpoint=None, auth_endpoint=None,
                 access_key_id=None, secret_access_key=None ):
        """
        Initializes COS connection set to provided bucket and prefix
        Will set credentials from environment variables or passed through args

        :param bucket: COS store bucket
        :type bucket: `str`
        :param api_key: IBM_API_KEY_ID to access COS
        :type api_key: `str`
        :param service_instance_id: IAM_SERVICE_ID to access COS
        :type service_instance_id: `str`
        :param endpoint: ENDPOINT to access COS
        :type endpoint: `str`
        :param auth_endpoint: IBM_AUTH_ENDPOINT to access COS
        :type auth_endpoint: `str`
        :param access_key_id: AWS_ACCESS_KEY_ID - paired with `secret_access_key`
        :type access_key_id: `str`
        :param secret_access_key: AWS_SECRET_ACCESS_KEY_ID - paired with `access_key_id`
        :type secret_access_key: `str`
        """
        self.bucket = s.COS_BUCKET if bucket is None else bucket
        self.api_key = s.COS_API_KEY if api_key is None else api_key
        self.service_instance_id = s.COS_SERVICE_INSTANCE_ID if service_instance_id is None else service_instance_id
        self.endpoint = s.COS_ENDPOINT if endpoint is None else endpoint
        self.auth_endpoint = s.COS_AUTH_ENDPOINT if auth_endpoint is None else auth_endpoint
        self.access_key = s.AWS_ACCESS_KEY_ID if access_key_id is None else access_key_id
        self.secret_access_key = s.AWS_SECRET_ACCESS_KEY if secret_access_key is None else secret_access_key

        if not self.bucket:
            raise ValueError("Bucket must not be None")

        if not self.access_key and not self.secret_access_key:
            self.cos_client = ibm_boto3.client(service_name='s3',
                                               ibm_api_key_id=self.api_key,
                                               ibm_service_instance_id=self.service_instance_id,
                                               ibm_auth_endpoint=self.auth_endpoint,
                                               config=Config(signature_version='oauth'),
                                               endpoint_url=self.endpoint)

        else:
            self.cos_client = ibm_boto3.client(service_name='s3',
                                               aws_access_key_id=self.access_key,
                                               aws_secret_access_key=self.secret_access_key,
                                               endpoint_url=self.endpoint)

    def _apply_all_files_in_cos_with_prefix(self, prefix, func):
        objects = self.cos_client.list_objects(Bucket=self.bucket, Prefix=prefix)

        if 'Contents' not in objects:
            raise ValueError('Not matching object found in COS')

        keys = [i['Key'] for i in objects['Contents']]

        file_name = None
        for k in keys:
            f = k.split(prefix)[1]
            if f.startswith('/'):
                f = f[1:]
            elif f == '':
                f = k.split('/')[-1]  # Support the case if the source path is provided as a file
                file_name = f
            elif not prefix.endswith('/'):
                continue   # Do not do anything otherwise, can happen if source matches a substring of a longer folder/file name
            func(k, f)

        return file_name

    def retrieve(self, source, destination):
        """
        Retrieves asset from COS

        :param source: path to asset
        :type source: `str`
        :param destination: path to store asset
        :return: path to saved file
        """

        def _func(k, f):
            d = os.path.join(destination, f)
            if not os.path.exists(os.path.dirname(d)):
                os.makedirs(os.path.dirname(d))
            try:
                self.cos_client.download_file(Bucket=self.bucket, Key=k, Filename=d)
            except IsADirectoryError:
                pass

        file_name = self._apply_all_files_in_cos_with_prefix(source, _func)

        if file_name is not None:
            destination = os.path.join(destination, file_name)
        return destination

    def store(self, source, destination):
        """
        Stores asset to COS

        :param source: path to retrieve file
        :type source: `str`
        :param destination: prefix to store file
        :type destination: `str`
        :return: path to exported file
        """

        if destination.startswith('/'):
            destination = destination[1:]

        if os.path.isdir(source):
            for (dir_path, dir_names, file_names) in os.walk(source):
                for f in file_names:
                    p = dir_path.split(source)[1]
                    if p != '':
                        key = destination + '/' + p + '/' + f
                    else:
                        key = destination + '/' + f
                    while '//' in key:
                        key = key.replace('//', '/')
                    self.cos_client.upload_file(Filename=os.path.join(dir_path, f), Bucket=self.bucket, Key=key)
            return '{}/{}'.format(self.bucket, destination)
        else:
            if os.path.splitext(destination)[-1] == '':
                key = destination + '/' + os.path.basename(source)
            else:
                key = destination
            while '//' in key:
                key = key.replace('//', '/')
            self.cos_client.upload_file(Filename=source, Bucket=self.bucket, Key=key)
            return '{}/{}'.format(self.bucket, key)

    def store_temporary(self, source, destination=''):
        """
        Stores asset to COS in a temporary folder

        :param source: path to retrieve file
        :type source: `str`
        :param destination: prefix to store file
        :type destination: `str`
        :return: a TemporaryCOS object, to be called with 'with' statement
        """

        class TemporaryCOS(object):
            def __init__(self, storage_class, source, destination):
                self.source = source
                self.destination = destination
                self.storage_class = storage_class

                hash_str = str(uuid.uuid1())
                short_hash_str = hash_str.split('-')[0]

                model_path = destination + '/tmp-{}'.format(hash_str)
                if model_path.startswith('/'):
                    model_path = model_path[1:]
                while '//' in model_path:
                    model_path = model_path.replace('//', '/')

                self.model_path = model_path
                self.short_hash_str = short_hash_str

                self.full_path = self.storage_class.store(source, model_path)

            def __enter__(self):
                return self.full_path, self.model_path, self.short_hash_str

            def __exit__(self, *args):
                def _func(k, f):
                    self.storage_class.cos_client.delete_object(Bucket=self.storage_class.bucket, Key=k)

                self.storage_class._apply_all_files_in_cos_with_prefix(self.model_path, _func)

        return TemporaryCOS(self, source, destination)

