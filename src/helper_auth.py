import json

import flask_login
from werkzeug.local import LocalProxy


# Used by authenticate_request()
class User(flask_login.UserMixin):
    pass


no_password_change_required = ["password_change_put"]


def get_all_roles(bucket):
    roles = {}
    bucket_records = bucket.objects.filter(Prefix="roles/")
    for bucket_record in bucket_records:
        role_name = bucket_record.key.replace("roles/", "")
        response = bucket_record.get()
        json_details = response["Body"].read().decode()
        role_details = json.loads(json_details)
        roles[role_name] = role_details
    return roles


def get_all_rights_mapping(bucket):
    rights_mapping = {}
    bucket_records = bucket.objects.filter(Prefix="rights_mapping/")
    for bucket_record in bucket_records:
        rights_mapping_name = bucket_record.key.replace("rights_mapping/", "")
        response = bucket_record.get()
        json_details = response["Body"].read().decode()
        rights_mapping_details = json.loads(json_details)
        rights_mapping[rights_mapping_name] = rights_mapping_details
    return rights_mapping


def check_user(
    bucket,
    username: str,
    password: str,
):
    try:
        object_record = bucket.Object(f"users/{username}")
        response = object_record.get()
        json_record = response["Body"].read().decode()
        user_record = json.loads(json_record)
        user_exists = (
            user_record["password"] == password
            and user_record["active"]
            and not user_record["blacklist"]
        )

        return user_exists, user_record
    except Exception as err:
        print(str(err))
        return False, None


def get_access_rights(user_object, roles):
    return roles[user_object["role"]]


def get_mapping_right(mapping_key, rights_mapping):
    return rights_mapping[mapping_key]


def authenticate_request(
    request: LocalProxy, bucket, roles: dict, rights_mapping: dict
):
    username = request.headers.get("Username")
    password = request.headers.get("Password")

    user_exists, user_object = check_user(
        bucket,
        username,
        password,
    )
    if not user_exists:
        return
    access_rights = get_access_rights(user_object, roles)

    endpoint = request.url.split("/")[-1].split("?")[0]
    method = request.method.lower()
    mapping_key = f"{endpoint}_{method}"
    mapping_right = get_mapping_right(mapping_key, rights_mapping)

    if not user_object["password_change_required"] and access_rights[mapping_right]:
        user = User()
        user.id = username
        return user
    elif mapping_key in no_password_change_required and access_rights[mapping_right]:
        user = User()
        user.id = username
        return user
    else:
        return


# def authenticate_request(
#     request: LocalProxy, bucket, roles: dict, rights_mapping: dict
# ):
#     user = User()
#     user.id = "<insert username>"
#     return user
