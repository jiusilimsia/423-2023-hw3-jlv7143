# Cloud Classification App
Jiusi Li

This is a simple [Streamlit](https://docs.streamlit.io/) application that loads a trained machine learning classifier on the [Cloud dataset](https://archive.ics.uci.edu/ml/datasets/cloud), and provides an webpage for performing inference on new data.





## Run the app on local machine

To set up the environment to run this project on a local machine:

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(Set up the AWS credentials if need)
```bash
export AWS_PROFILE=<your_aws_profile_name>
```

To use this application, you can run the `app.py` script with Streamlit, which will start the app server.

```bash
streamlit run src/app.py
```

The script loads the classifier and preprocessed data from the `cloud_classifier.pkl` and `cloud_data.csv` files located in the model folders that correspond to the chosen model version on the webpage. All the model folders are located within the artifacts folder.

Once the server is running, you can access the Swagger UI by visiting `http://localhost` in your web browser. From there, you can making predicitons with the `IR Min`, `IR Max`, `IR Mean`, `Visible entropy` and `Visible contrast` fields.



## Run the unit tests on local machine

Set up the environment:

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_unittest.txt
```

Run the unit tests: in the root directory of this project, run the following command:

```shell
pytest
```





## Docker (Build Docker image and run Docker container)

To run the Streamlit application in a Docker container, you can use the following commands:

### Build the Docker image
(image is platform (linux/amd64))
```bash
docker build -f dockerfiles/Dockerfile -t <image_name> .
```

### Run the app
Set the AWS_PROFILE environment variable to the AWS profile you want to use; 
Build the Docker container; 
Run the web app; 
```bash
docker run -v ~/.aws:/root/.aws -e AWS_PROFILE=<your_aws_profile_name> <image_name>
```

This will build a Docker image for the Streamlit application, and then run a container based on that image. 



### Build the Docker image for tests

```bash
docker build -f dockerfiles/Dockerfile_unittest -t <image_name> .
```

### Run the tests

```bash
docker run -it  <image_name>
```





## Run the app on AWS

If the AWS ECS task for this web app is still running, then you can access this webpage by logging into the Northwestern University VPN and clicking a link. However, since the task is stopped, the link no longer works.

If you want to start the app, you have to log into my account and start it in ECS hw3-cloud-app Cluster hw3-cloud-app Service and update service with Desired tasks=1 and start a task (However, I'm not going to tell you my account and password). After the task start to run, you can click the link under DNS names in Networking section of ECS hw3-cloud-app Service in hw3-cloud-app Cluster.





## Instructions on making predictions with the web app

- On the left side of the webpage, you can use the sliders to set the parameters of cloud to make predictions. The explanation of the parameters are listed under the parameter name.
- On the main page of the webpage, you can use the drop-down menu to select the version of the model you want to use to make the predictions.
- You can find the prediction for the desired cloud class, along with its corresponding probability, under the "Prediction" section of the webpage.