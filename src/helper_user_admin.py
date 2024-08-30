import hashlib
import json
import random
import string

import boto3
from botocore.exceptions import ClientError

from .helper_misc import DEFAULT_PAGE_SIZE, check_offset_limit

characterList = string.ascii_letters + string.digits + string.punctuation


def get_salt():
    return "sg_mgmt_uni_mitb_2023"


def get_user(
    bucket,
    username: str,
    return_record: bool = False,
):
    try:
        object_record = bucket.Object(f"users/{username}")
        response = object_record.get()
        if return_record:
            json_record = response["Body"].read().decode()
            user_record = json.loads(json_record)
            return True, user_record
        else:
            return True
    except Exception as err:
        print(str(err))
        return False


def get_all_users(bucket, offset=None, limit=None):
    offset, limit = check_offset_limit(offset, limit)

    users = []
    records = bucket.objects.filter(Prefix="users/").page_size(limit)
    page_idx = offset // limit
    pages = list(records.pages())
    for object_record in pages[page_idx]:
        response = object_record.get()
        json_record = response["Body"].read().decode()
        user_record = json.loads(json_record)
        user_record["username"] = object_record.key.replace("users/", "")
        del user_record["password"]
        del user_record["password_change_required"]
        users.append(user_record)
    page_num = page_idx + 1
    return users, len(pages), page_num


def generate_password() -> str:
    password = []
    for i in range(12):
        randomchar = random.choice(characterList)
        password.append(randomchar)

    return "".join(password)


def hash_password(password: str):
    salted_password = password + get_salt()
    return hashlib.sha256(salted_password.encode("utf-8")).hexdigest()


def add_user(s3_bucket, username: str, role: str, password: str, timestamp_ms: int):
    hashed_password = hash_password(password)
    upload_user(
        s3_bucket, username, role, hashed_password, True, False, True, timestamp_ms
    )


def update_user(s3_bucket, username: str, user_record: dict):
    upload_user(
        s3_bucket,
        username,
        user_record["role"],
        user_record["password"],
        user_record["password_change_required"],
        user_record["blacklist"],
        user_record["active"],
        user_record["timestamp_ms"],
    )


def delete_user(s3_bucket, username: str, user_record: dict):
    upload_user(
        s3_bucket,
        username,
        user_record["role"],
        user_record["password"],
        user_record["password_change_required"],
        user_record["blacklist"],
        False,
        user_record["timestamp_ms"],
    )


# Can be used for adding, updating, or soft deleting a user
def upload_user(
    s3_bucket,
    username: str,
    role: str,
    hashed_password: str,
    password_change_required: bool,
    blacklist: bool,
    active: bool,
    timestamp_ms: int = None,
):
    try:
        user_record = {
            "password": hashed_password,
            "role": role,
            "password_change_required": password_change_required,
            "blacklist": blacklist,
            "active": active,
            "timestamp_ms": timestamp_ms,
        }
        json_record = json.dumps(user_record, indent=4)
        bucket_key = f"users/{username}"
        s3_bucket.put_object(
            Key=bucket_key,
            Body=json_record,
        )
        print("Uploaded user to database.")
    except Exception as err:
        print("Unable to upload user to database.")
        print(str(err))
        raise


def send_email(
    ses_client,
    source_email: str,
    destination_email: str,
    template_name: str,
    username: str,
    password: str,
):
    destination = {"ToAddresses": [destination_email]}

    send_args = {
        "Source": source_email,
        "Destination": destination,
        "Template": template_name,
        "TemplateData": json.dumps(
            {
                "username": username,
                "password": password,
            }
        ),
    }

    try:
        response = ses_client.send_templated_email(**send_args)
        message_id = response["MessageId"]
        print(f"Sent mail {message_id} from {source_email} to {destination}.")
    except ClientError:
        print(f"Couldn't send mail from {source_email} to {destination}.")
        raise
