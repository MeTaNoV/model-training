## ETL / Train / Deploy / Analyze using GCP and Labelbox


### Overview

Run ETL jobs, train models, deploy models, and track model performance all from a single service. The code deploys a service called the `coordinator` to google cloud. It exposes a rest api for launching various pipelines. The coordinator only has to be deployed once and then will be controllable via the labelbox web app (WIP). This project is designed to be easily extended for custom workflows. However, we will support the following models with no additional configuration required:

1. Image Radio Classification
2. Image Checklist / Dropdown Classification
3. Image Bounding Box Detection
4. Text Radio Classification
5. Text Checklist / Dropdown Classification
6. Text Named Entity Recognition

### Requirements

1. Create a service account in the google cloud UI. This account must have the following permissions:
    * Basic editor permissions
    * Secret manager admin
2. Download the private key for the service account
    * Put it anywhere on your computer
    * Set the GOOGLE_APPLICATION_CREDENTIALS to point to it. `export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/model-training-credentials.json`
3. docker and docker compose v2 installed
    * https://docs.docker.com/compose/install/
    * https://docs.docker.com/get-docker/ (not required for mac since docker and docker compose v2 are available together in the first link).
4. gcloud cli installed (and configured for the proper service account)
    * Run `curl https://sdk.cloud.google.com | bash` to install
    * `source ~/.<bash_profile/zshrc/bashrc>` to load env vars
    * `gcloud auth login` to login (You also can login from the service account directly with `gcloud auth activate-service-account SERVICE_ACCOUNT_ID@PROJECT_ID.iam.gserviceaccount.com --key-file=$GOOGLE_APPLICATION_CREDENTIALS`)
    * Set the correct google project with `gcloud config set project PROJECT_ID`
5. Connect docker to GCR by running `gcloud auth configure-docker`


### Deployment

Deploy the coordinator service on port 80

1. Create a .env file to keep track of the following env vars (copy .env.example to get started):
    - `DEPLOYMENT_NAME`
        - This is the name that all of the google resources will use. This will enable multiple deployments. E.g. prod-training-service or dev-training-service.
    - `GCS_BUCKET`
        - GCS bucket to store all of the artifacts. If the bucket doesn't exist it will automatically be created.
    - `GOOGLE_PROJECT`
        - Google cloud project name.
    - `SERVICE_SECRET`
        - This can be anything. You will have to use the same secret when making a request to the service.
    - `GOOGLE_APPLICATION_CREDENTIALS`
        - Path to the application credentials.
    - `GOOGLE_SERVICE_ACCOUNT`
        - Google service account. Will have the following format: `<name>@<project>.iam.gserviceaccount.com`.
    - `LABELBOX_API_KEY`
2. Once the .env file has the correct values, load the env vars by running `source .env`
    - If you update the .env file, make sure to re-run `source .env` to load the latest env vars.
3. Deploy the service
    - To the cloud: `./deployment/deploy.sh`
    - Locally: `./run.sh`
4. Test that it is running with:
    - curl http://ip/ping
    - ip will be `0.0.0.0` for a local deployment and the remote ip will be printed to the console when you run the deployment script.
    - The server will respond with pong if the deployment was successful


### Managing training deployments


* Delete the training deployment
    - Set `DEPLOYMENT_NAME` env vars
    - Run the `./deployment/teardown.sh` script
        - Deletes the firewall, static ip, coordinator instance and the secrets
        - The gcs bucket and gcr training artifacts, the gcr images will not be updated, and all vertex artifacts will remain.

* Updating secrets
    - Set `LABELBOX_API_KEY`, `SERVICE_SECRET`, and `DEPLOYMENT_NAME` env vars
    - Run `./deployment/reload_secrets.sh`

* Updating the coordinator
    - Change your code locally
    - Make sure the `DEPLOYMENT_NAME` env var is set, and run `deployment/reload_coordinator.sh`

* Updating containerized jobs
    - The training pipeline always uses the latest images. This means that anytime you build and push, the coordinator will use the pushed code automatically. <b>Do not push containers to GCR for a deployment that is being used in production unless you want those changes to be used</b>/.
    - Update your code locally, run `docker compose build <container_name>`, and then `docker compose push <container_name>`.



### Design

The coordinator is an api for managing etl/training/deployment jobs. The coordinator doesn't do any data processing. Instead it runs various pipelines by running containers in vertex ai. The code is organizationed such that all logic for coordinating the workflows are under `src/coordinator` and any custom jobs are defined under `src/jobs`.


Key terms:
* `Job`:
    - A job is a single task such as etl, training, deployment, or model evaluation.
    - It exposes a run method that executes the job.
* `Pipeline`:
    - A pipeline contains the logic for running a series of jobs.
    - It exposes three functions.
        1. parse_args: Used to validate the dict payload that contains the pipeline parameters
        2. run: A function that defines the behavior of the pipeline when run from the local machine


### Custom Pipelines / Extending

#### Creating Custom Jobs
* A custom job can be used to run any arbitrary job on gcp. This is not necessary if you already have a container on gcs that defines the job or the job can be run from a remote client (in this case run from the pipeline). To create the custom job do the following:
1. Create a new directory under the `jobs` directory.
2. Write the logic for the job. Include a cli to pass in any arguments.
3. Add a Dockerfile
5. Add to the docker compose
6. Add tests

#### Extending a Pipeline
1. Find the pipeline you want to extend under `coordinator/pipelines/...`
2. Create a new class that inherits from the base job class. Define an implementation for `run`.
3. Add the new job to the pipeline and update the pipeline's `run` function


#### Creating a New Pipeline
1. Copy a pipeline found under `coordinator/pipelines/...`
2. Update the logic for your new job
3. Add the job to a pipeline in `coordinator/config.py`
    * Update `pipelines` to include the new workflow
    * Add the new pipeline name `PipelineName` type as a Literal


### Testing

* To run any of the seed scripts you can use the `seed.sh` script. It will run any of the python script in a docker container
    * Run: `./seed.sh <script>.py`
    * Once the data is seeded you can test against a local or live deployment of the coordinator


### HTTPS

* By default, the service uses http. To support https, you must have a domain name and an SSL certificate from a certificate authority. Once you have the certificate you can either directly update the service to use https or configure a reverse proxy with https. To update the service to use https, simply provide the ssl_keyfile and ssl_certfile params to app.run() at the bottom of coordinator.py. 
 If you directly update the server to use https you might want to change the default port to 443. You can change this in app.run(), the deployment script (update the firewall), and docker compose (for local usage)


### M1 Mac
* Images built for arm64 architectures cannot be run on google cloud. To fix this issue, you must build your containers to run on amd64 architectures. Before building any containers or running the deploy script run the following:
`export DOCKER_DEFAULT_PLATFORM=linux/amd64`
