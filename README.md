# Distributed Trainer with ViT example

## 🔓 Local Setup

### Install dependencies

In terminal run the command

```
pip install -r requirements
```

### GCP Authentication

Step 1: Setup Gcloud cli

Follow the guide [here](https://cloud.google.com/sdk/docs/install-sdk)

Step 2: Authenticate using your user account

In terminal run the command
```
gcloud auth login
```
And sign into your google account in the browser tab that opens automatically.

In terminal run the command
```
gcloud auth application-default login
```

And sign into your google account in the browser tab that opens automatically.

Step 3: Set project to [YOUR PROJECT NAME]

In terminal run the command
```
gcloud config set project [YOUR PROJECT NAME]
```

### Environment Setup

Make sure that the ENV file environments/.env is filled in with the correct parameters to rain the model.
Most important variables to fill in are `PROJECT_ID`, `BUCKET_NAME` and `REGION`

## 🚀 Train Locally

**In terminal run**

```
make train
```


## 🚀 Train distributed on Vertex AI

**In terminal run**

```
make train-distributed
```



## 🐳 Model Deployment (locally)

**In terminal run**

```
make run-locally
```

OR (to run locally on Docker)

```
make build run
```

The local endpoint is: `localhost:8080/predictions/vit`

The raw body for the POST request is in json format and should be following this template:

```
{
    "instances":
    [{"link": "gs://[BUCKET_NAME]/path/to/image.jpg"}]
}
```


If successful returns response:

```
{
    "predictions":[
                {"response": "dog",
                 "probabilities": {
                        "plane": 0.0078,
                        "dog": 0.98,
                        "...": ...,
                        }
                 },
                 {...}
    ]
}
```

## 🐳 Deploy latest version to Endpoint

In terminal run the command:
```
make build push deploy

```

