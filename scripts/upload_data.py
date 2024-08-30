import json
import os
import sys

import boto3
import joblib
from dotenv import load_dotenv
from torch import load as torch_load
from torch.nn import Sequential
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizerFast,
    CLIPVisionModelWithProjection,
)

# from torchvision.models import resnet50

# These functions only work when you are in the project folder.
# Once there, call `python exploration/explore_s3.py`.
sys.path.append(os.getcwd())

load_dotenv()

s3_resource = boto3.resource("s3")
aws_ils_auth_bucket = s3_resource.Bucket(os.getenv("BUCKET_AUTH"))
aws_ils_model_bucket = s3_resource.Bucket(os.getenv("BUCKET_MODEL"))


def put_auth(record, key):
    json_record = json.dumps(record, indent=4)
    aws_ils_auth_bucket.put_object(Key=key, Body=json_record)


def upload_rights_mapping(rights_mapping):
    print("Uploading mapping of API endpoints to access rights.")
    rights_mapping = rights_mapping
    for key in rights_mapping:
        bucket_key = f"rights_mapping/{key}"
        put_auth(rights_mapping[key], bucket_key)
    print("Uploaded mapping of API endpoints to access rights.")


def upload_roles(roles):
    print("Uploading roles.")
    roles = roles
    for key in roles:
        bucket_key = f"roles/{key}"
        put_auth(roles[key], bucket_key)
    print("Uploaded roles.")


def upload_user_base(users):
    print("Uploading sample users.")
    users = users
    for key in users:
        bucket_key = f"users/{key}"
        put_auth(users[key], bucket_key)
    print("Uploaded sample users.")


def upload_model():
    model_paths = [
        "clip_image_feature_extractor",
        "clip_image_processor",
        "clip_text_feature_extractor",
        "clip_text_tokenizer",
    ]
    for model_path in model_paths:
        for file in os.listdir(f"model/{model_path}"):
            with open(os.path.join("model", model_path, file), "rb") as data:
                aws_ils_model_bucket.put_object(
                    Key=f"model/{model_path}/{file}", Body=data
                )
        print(f"Uploaded model/{model_path}.")

    classifier_path = "model/model_with_class_names.joblib"
    with open(classifier_path, "rb") as data:
        aws_ils_model_bucket.put_object(Key="model/knn_centroids_classifier", Body=data)
    print("Uploaded model/knn_centroids_classifier.")


def save_clip_model_to_local():
    embed_model_name = "openai/clip-vit-large-patch14"
    image_processor_path_name = "model/clip_image_processor"
    image_model_path_name = "model/clip_image_feature_extractor"
    text_tokenizer_path_name = "model/clip_text_tokenizer"
    text_model_path_name = "model/clip_text_feature_extractor"

    # Load image model
    image_processor = CLIPImageProcessor.from_pretrained(embed_model_name)
    image_model = CLIPVisionModelWithProjection.from_pretrained(embed_model_name)

    # Save image model
    image_processor.save_pretrained(f"{image_processor_path_name}")
    image_model.save_pretrained(f"{image_model_path_name}")
    print("Saved image model to local.")

    # Load text model
    tokenizer = CLIPTokenizerFast.from_pretrained(embed_model_name)
    text_model = CLIPTextModelWithProjection.from_pretrained(embed_model_name)

    # Save text model
    tokenizer.save_pretrained(text_tokenizer_path_name)
    text_model.save_pretrained(text_model_path_name)
    print("Saved text model to local.")


if __name__ == "__main__":
    rights_mapping = {
        "password_change_post": "administration",
        "password_change_put": "individual_admin",
        "inference_post": "inference",
        "inference_put": "inference",
        "all_records_post": "all_records",  # for supervisor to see list of records
        "all_records_get": "all_records",  # for supervisor to see details of record (e.g. for batch evaluation or training)
        "individual_records_post": "individual_records",  # for individual user to see their own list of records for inference, batch evaluation, training
        "evaluation_get": "training",  # getting info of submitted batch evaluation job (for individual user)
        "evaluation_post": "training",  # uploading images for evaluation
        "training_get": "training",  # getting info of submitted training job (for individual user)
        "training_post": "training",  # uploading images for training for new class or existing class
        "training_put": "training_update",  # updating list of images to use for training (e.g. removing blacklist images)
        "validation_get": "training",  # getting info of submitted validation feature extraction job (for individual user)
        "text_features_get": "training",  # getting info of text features available (i.e. feature vectors for each class name)
        "text_features_post": "training",  # getting list of text features available (i.e. list of class names)
        "visualisation_get": "visualise",  # getting available classes for 2d visualisation
        "visualisation_post": "visualise",  # triggering pca computations for selected classes and options
        "visualisation_put": "visualise",  # updating pca computation job triggered by batch training robustness check that manual review is triggered
        "image_get": "visualise",  # getting image for training image visualisation
        "user_admin_get": "administration",
        "user_admin_post": "administration",
        "user_admin_put": "administration",
        "user_admin_delete": "administration",
    }

    roles = {
        "user": {
            "inference": True,
            "training": False,
            "training_update": False,
            "visualise": False,
            "individual_records": True,
            "all_records": False,
            "individual_admin": True,
            "administration": False,
        },
        "trainer": {
            "inference": True,
            "training": True,
            "training_update": False,
            "visualise": False,
            "individual_records": True,
            "all_records": False,
            "individual_admin": True,
            "administration": False,
        },
        "trainer_supervisor": {
            "inference": True,
            "training": True,
            "training_update": True,
            "visualise": True,
            "individual_records": True,
            "all_records": True,
            "individual_admin": True,
            "administration": False,
        },
        "administrator": {
            "inference": False,
            "training": False,
            "training_update": False,
            "visualise": False,
            "individual_records": False,
            "all_records": False,
            "individual_admin": True,
            "administration": True,
        },
    }

    # Key in users and credentials for testing purposes
    users = {}

    # Uncomment where appropriate to run the desired function.

    upload_rights_mapping(rights_mapping)
    # upload_roles(roles)
    # upload_user_base(users)

    # save_clip_model_to_local()
    # upload_model()
