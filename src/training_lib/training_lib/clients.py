import os

from training_lib.errors import MissingEnvironmentVariableException

from google.cloud import secretmanager
from google.cloud import storage
import labelbox


def load_labelbox_api_key() -> str:
    """
    Checks if the labelbox api key is available in the environment and
        if not it fetches the key from google secret manager.

    Returns:
         the labelbox api key
    """
    _labelbox_api_key = os.environ.get('LABELBOX_API_KEY')
    if _labelbox_api_key is None:
        deployment_name = os.environ['DEPLOYMENT_NAME']
        secret_client = secretmanager.SecretManagerServiceClient()
        secret_id = f"{deployment_name}_labelbox_api_key"
        name = f"projects/{os.environ['GOOGLE_PROJECT']}/secrets/{secret_id}/versions/1"
        response = secret_client.access_secret_version(request={"name": name})
        _labelbox_api_key = response.payload.data.decode("UTF-8")
    return _labelbox_api_key


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





