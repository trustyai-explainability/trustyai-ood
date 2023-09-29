import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

COS_BUCKET = os.environ.get('OOD_COS_BUCKET')
COS_API_KEY = os.environ.get('OOD_COS_API_KEY')
COS_SERVICE_INSTANCE_ID = os.environ.get('OOD_COS_SERVICE_INSTANCE_ID')
COS_ENDPOINT = os.environ.get('OOD_COS_ENDPOINT')
COS_AUTH_ENDPOINT = os.environ.get('OOD_COS_AUTH_ENDPOINT')
AWS_ACCESS_KEY_ID = os.environ.get('OOD_COS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('OOD_COS_SECRET_ACCESS_KEY')
KSERVE_IP_PORT = os.environ.get('OOD_KSERVE_IP_PORT')
K8S_CONTEXT = os.environ.get('OOD_K8S_CONTEXT')
K8S_NAMESPACE = os.environ.get('OOD_K8S_NAMESPACE')
K8S_SA_NAME = os.environ.get('OOD_K8S_SA_NAME')
RHODS_PROJECT = os.environ.get('OOD_RHODS_PROJECT')
RHODS_RUNTIME = os.environ.get('OOD_RHODS_RUNTIME')
RHODS_STORAGE_KEY = os.environ.get('OOD_RHODS_STORAGE_KEY')

DEFAULT_OOD_THRESH_PERCENTILE = 20

# set empty string env to none
if not COS_BUCKET:
    COS_BUCKET = None

if not COS_API_KEY:
    COS_API_KEY = None

if not COS_SERVICE_INSTANCE_ID:
    COS_SERVICE_INSTANCE_ID = None

if not COS_ENDPOINT:
    COS_ENDPOINT = None

if not COS_AUTH_ENDPOINT:
    COS_AUTH_ENDPOINT = None

if not AWS_ACCESS_KEY_ID:
    AWS_ACCESS_KEY_ID = None

if not AWS_SECRET_ACCESS_KEY:
    AWS_SECRET_ACCESS_KEY = None

if not KSERVE_IP_PORT:
    KSERVE_IP_PORT = None

if not K8S_CONTEXT:
    K8S_CONTEXT = None

if not K8S_NAMESPACE:
    K8S_NAMESPACE = None

if not K8S_SA_NAME:
    K8S_SA_NAME = None

if not RHODS_PROJECT:
    RHODS_RUNTIME = None

if not RHODS_RUNTIME:
    RHODS_RUNTIME = None

if not RHODS_STORAGE_KEY:
    RHODS_STORAGE_KEY = None
