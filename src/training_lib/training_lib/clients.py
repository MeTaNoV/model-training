import os

from training_lib.errors import MissingEnvironmentVariableException

from google.cloud import secretmanager
from google.cloud import storage
import labelbox


def load_secret(secret_id: str) -> str:
    secret_client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{os.environ['GOOGLE_PROJECT']}/secrets/{secret_id}/versions/1"
    response = secret_client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


def load_service_secret() -> str:
    """
    Checks if the service secret is available in the environment and
        if not it fetches the key from google secret manager.

    Returns:
         the service secret
    """
    service_secret = os.environ.get('SERVICE_SECRET')
    if service_secret is None:
        deployment_name = os.environ['DEPLOYMENT_NAME']
        secret_id = f"{deployment_name}_service_secret"
        service_secret = load_secret(secret_id)
    return service_secret


def load_labelbox_api_key() -> str:
    """
    Checks if the labelbox api key is available in the environment and
        if not it fetches the key from google secret manager.

    Returns:
         the labelbox api key
    """
    labelbox_api_key = os.environ.get('LABELBOX_API_KEY')
    if labelbox_api_key is None:
        deployment_name = os.environ['DEPLOYMENT_NAME']
        secret_id = f"{deployment_name}_labelbox_api_key"
        labelbox_api_key = load_secret(secret_id)
    return labelbox_api_key


def get_gcs_client() -> storage.Client:
    """
    Returns:
        google cloud storage client.
    """
    google_project = os.environ.get('GOOGLE_PROJECT')
    if not google_project:
        raise MissingEnvironmentVariableException(f"Must set GOOGLE_PROJECT env var")
    return storage.Client(project=google_project)

def get_lb_client() -> labelbox.Client:
    """
    Returns:
         labelbox client.
    """
    labelbox_api_key = load_labelbox_api_key()
    return labelbox.Client(api_key=labelbox_api_key,
                       endpoint='https://api.labelbox.com/_gql', enable_experimental=True)





