import json
import os
import sys

import boto3
from dotenv import load_dotenv

# These functions only work when you are in the project folder.
# Once there, call `python exploration/explore_s3.py`.
sys.path.append(os.getcwd())
from src import helper_model

load_dotenv()

s3_client = boto3.client("s3")
s3_resource = boto3.resource("s3")
aws_ils_records_bucket = s3_resource.Bucket("aws-ils-records")
aws_ils_auth_bucket = s3_resource.Bucket("aws-ils-auth")
aws_ils_model_bucket = s3_resource.Bucket("aws-ils-model")


def get_prediction_records():
    file_path = "data/prediction_records.joblib"
    if os.path.exists(file_path):
        prediction_records = joblib.load(file_path)
    else:
        prediction_records = []
    return prediction_records


def list_local_records():
    prediction_records = get_prediction_records()
    for x in prediction_records:
        print(x["userID"], x["timestamp_ms"], x["predictions"])


def list_buckets():
    response = s3_client.list_buckets()
    buckets = [x["Name"] for x in response["Buckets"]]
    print(buckets)


def put_record(record):
    json_record = json.dumps(record, indent=4)
    aws_ils_records_bucket.put_object(
        Key=f"{record['userID']}/{record['timestamp_ms']}", Body=json_record
    )


def put_records():
    for record in helper_model.get_prediction_records():
        put_record(record)


def list_bucket_objects():
    for object_record in aws_ils_records_bucket.objects.all():
        response = object_record.get()
        json_record = response["Body"].read().decode()
        record = json.loads(json_record)
        print(record["userID"], record["timestamp_ms"], type(record))
        if "feedback_correct" in record:
            print(record["feedback_correct"], record["feedback_remarks"])
            print()


def update_records_with_new_format():
    for object_record in aws_ils_records_bucket.objects.all():
        response = object_record.get()
        json_record = response["Body"].read().decode()
        record = json.loads(json_record)
        record["feedback_correct"] = True
        record["feedback_remarks"] = None
        put_record(record)


def get_one_record():
    record = helper_model.get_prediction_record_from_bucket(
        aws_ils_records_bucket, "<insert username of uploader>", "<insert timestamp>"
    )
    print(record["userID"], record["timestamp_ms"], type(record))


def put_auth(record, key):
    json_record = json.dumps(record, indent=4)
    aws_ils_auth_bucket.put_object(Key=key, Body=json_record)


# def upload_user_base():
#     users = helper_auth.sample_users
#     for key in users:
#         bucket_key = f"users/{key}"
#         put_auth(users[key], bucket_key)

# def upload_roles():
#     roles = helper_auth.sample_roles
#     for key in roles:
#         bucket_key = f"roles/{key}"
#         put_auth(roles[key], bucket_key)

# def upload_rights_mapping():
#     # put_auth(helper_auth.sample_rights_mapping, "rights_mapping")
#     rights_mapping = helper_auth.sample_rights_mapping
#     for key in rights_mapping:
#         bucket_key = f"rights_mapping/{key}"
#         put_auth(rights_mapping[key], bucket_key)


def upload_model():
    # model_path = 'model/food_model.pth'
    model_path = "model/resnet50_car_damage.pth"
    with open(model_path, "rb") as data:
        aws_ils_model_bucket.put_object(Key="model", Body=data)


def upload_class_names():
    # class_names = ["Apple Pie", "Cheesecake", "Chocolate Cake", "French Toast", "Garlic Bread"]
    class_names = ["Broken Light", "Dent", "Glass Shatter"]
    json_class_names = json.dumps(class_names, indent=4)
    aws_ils_model_bucket.put_object(Key="class_names", Body=json_class_names)


def upload_model_and_class_names():
    upload_model()
    upload_class_names()


if __name__ == "__main__":
    if int(sys.argv[1]) == 0:
        list_local_records()
    elif int(sys.argv[1]) == 1:
        list_buckets()
    elif int(sys.argv[1]) == 2:
        put_records()
    elif int(sys.argv[1]) == 3:
        list_bucket_objects()
    elif int(sys.argv[1]) == 4:
        update_records_with_new_format()
    elif int(sys.argv[1]) == 5:
        get_one_record()
    # elif int(sys.argv[1]) == 6:
    #     upload_user_base()
    # elif int(sys.argv[1]) == 7:
    #     upload_roles()
    # elif int(sys.argv[1]) == 8:
    #     upload_rights_mapping()
    elif int(sys.argv[1]) == 9:
        upload_model()
    elif int(sys.argv[1]) == 10:
        upload_class_names()
    elif int(sys.argv[1]) == 11:
        upload_model_and_class_names()
