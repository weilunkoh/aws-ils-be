from flask_restx import Api
from werkzeug.datastructures import FileStorage


def get_login_parser(api: Api):
    login_parser = api.parser()
    login_parser.add_argument("userID", location="form", type=str, required=True)
    login_parser.add_argument("password", location="form", type=str, required=True)
    return login_parser


def get_inference_parser(api: Api):
    inference_parser = api.parser()
    inference_parser.add_argument(
        "file", location="files", type=FileStorage, required=True
    )
    inference_parser.add_argument("top_k", location="form", type=int, required=False)
    return inference_parser


def get_background_timestamp_parser(api: Api):
    background_get_parser = api.parser()
    # Query string location for timestamp argument
    background_get_parser.add_argument(
        "timestamp", location="args", type=int, required=True
    )
    return background_get_parser


def get_page_parser(api: Api):
    page_parser = api.parser()
    page_parser.add_argument("offset", location="args", type=int)
    page_parser.add_argument("limit", location="args", type=int)
    return page_parser


def get_background_record_user_timestamp_parser(api: Api):
    background_get_parser = api.parser()

    # Query string location for record_type, user and timestamp argument
    background_get_parser.add_argument(
        "record_type", location="args", type=str, required=True
    )
    background_get_parser.add_argument(
        "username", location="args", type=str, required=True
    )
    background_get_parser.add_argument(
        "timestamp", location="args", type=int, required=True
    )
    return background_get_parser


def get_training_post_parser(api: Api):
    training_post_parser = api.parser()
    training_post_parser.add_argument(
        "training_file", location="files", type=FileStorage, required=False
    )
    training_post_parser.add_argument(
        "validation_file", location="files", type=FileStorage, required=False
    )
    return training_post_parser


def get_training_update_parser(api: Api):
    training_update_parser = api.parser()
    training_update_parser.add_argument(
        "update_to_used", location="form", type=str, action="append", required=False
    )
    training_update_parser.add_argument(
        "update_to_unused", location="form", type=str, action="append", required=False
    )
    training_update_parser.add_argument(
        "original_userID", location="form", type=str, required=False
    )
    training_update_parser.add_argument(
        "original_timestamp_ms", location="form", type=int, required=False
    )
    training_update_parser.add_argument(
        "target_class_name", location="form", type=str, required=False
    )

    return training_update_parser


def get_feedback_parser(api: Api):
    feedback_parser = api.parser()
    feedback_parser.add_argument("timestamp", location="form", type=int, required=True)
    feedback_parser.add_argument(
        "feedback_correct", location="form", type=str, required=True
    )
    feedback_parser.add_argument(
        "feedback_remarks", location="form", type=str, required=True
    )
    return feedback_parser


def get_records_parser(api: Api):
    records_parser = api.parser()
    records_parser.add_argument("record_type", location="form", type=str, required=True)
    records_parser.add_argument("offset", location="form", type=int)
    records_parser.add_argument("limit", location="form", type=int)
    return records_parser


def get_new_user_parser(api: Api):
    new_user_parser = api.parser()
    new_user_parser.add_argument("username", location="form", type=str, required=True)
    new_user_parser.add_argument("role", location="form", type=str, required=True)
    return new_user_parser


def get_update_user_parser(api: Api):
    update_user_parser = api.parser()
    update_user_parser.add_argument(
        "username", location="form", type=str, required=True
    )
    update_user_parser.add_argument("role", location="form", type=str, required=False)
    update_user_parser.add_argument(
        "blacklist", location="form", type=str, required=False
    )
    update_user_parser.add_argument("active", location="form", type=str, required=False)
    return update_user_parser


def get_delete_user_parser(api: Api):
    delete_user_parser = api.parser()
    delete_user_parser.add_argument(
        "username", location="form", type=str, required=True
    )
    return delete_user_parser


def pw_reset_parser(api: Api):
    pw_reset_parser = api.parser()
    pw_reset_parser.add_argument("username", location="form", type=str, required=True)
    return pw_reset_parser


def pw_change_parser(api: Api):
    pw_change_parser = api.parser()
    pw_change_parser.add_argument(
        "old_password", location="form", type=str, required=True
    )
    pw_change_parser.add_argument(
        "new_password", location="form", type=str, required=True
    )
    return pw_change_parser


def get_visualisation_get_parser(api: Api):
    visualisation_get_parser = api.parser()

    # Query string location for user, timestamp, and return_type argument
    visualisation_get_parser.add_argument(
        "username", location="args", type=str, required=True
    )
    visualisation_get_parser.add_argument(
        "timestamp", location="args", type=int, required=True
    )
    visualisation_get_parser.add_argument(
        "return_type", location="args", type=str, required=True
    )
    return visualisation_get_parser


def get_visualisation_post_parser(api: Api):
    vis_post_parser = api.parser()
    # Arguments for list of centroids, list of used, list of unused.
    vis_post_parser.add_argument(
        "centroids", location="form", type=str, action="append", required=False
    )
    vis_post_parser.add_argument(
        "used_training_images",
        location="form",
        type=str,
        action="append",
        required=False,
    )
    vis_post_parser.add_argument(
        "unused_training_images",
        location="form",
        type=str,
        action="append",
        required=False,
    )
    vis_post_parser.add_argument(
        "validation_images", location="form", type=str, action="append", required=False
    )
    vis_post_parser.add_argument(
        "class_names", location="form", type=str, action="append", required=False
    )
    return vis_post_parser


def get_visualisation_put_parser(api):
    vis_put_parser = api.parser()
    vis_put_parser.add_argument("pca_userID", location="form", type=str, required=True)
    vis_put_parser.add_argument(
        "pca_timestamp_ms", location="form", type=int, required=True
    )
    return vis_put_parser


def get_image_get_parser(api: Api):
    get_parser = api.parser()
    # Query string location for image class and timestamp argument
    get_parser.add_argument("folder", location="args", type=str, required=True)
    get_parser.add_argument("image_class", location="args", type=str, required=True)
    get_parser.add_argument("timestamp", location="args", type=int, required=True)
    return get_parser


def get_class_name_parser(api: Api):
    class_name_parser = api.parser()
    # Query string location for class name argument
    class_name_parser.add_argument(
        "class_name", location="args", type=str, required=True
    )
    return class_name_parser


def get_text_features_parser(api: Api):
    text_features_parser = api.parser()
    text_features_parser.add_argument("offset", location="form", type=int)
    text_features_parser.add_argument("limit", location="form", type=int)
    return text_features_parser


def get_all_parsers(api: Api):
    return {
        "login_parser": get_login_parser(api),
        "pw_reset_parser": pw_reset_parser(api),
        "pw_change_parser": pw_change_parser(api),
        "inference_parser": get_inference_parser(api),
        "feedback_parser": get_feedback_parser(api),
        "all_records_get_parser": get_background_record_user_timestamp_parser(api),
        "records_parser": get_records_parser(api),
        "background_get_parser": get_background_timestamp_parser(api),
        "evaluation_parser": get_inference_parser(api),  # same as inference_parser
        "training_post_parser": get_training_post_parser(api),
        "training_update_parser": get_training_update_parser(api),
        "visualisation_get_parser": get_visualisation_get_parser(api),
        "visualisation_post_parser": get_visualisation_post_parser(api),
        "visualisation_put_parser": get_visualisation_put_parser(api),
        "image_get_parser": get_image_get_parser(api),
        "all_users_parser": get_page_parser(api),
        "new_user_parser": get_new_user_parser(api),
        "update_user_parser": get_update_user_parser(api),
        "delete_user_parser": get_delete_user_parser(api),
        "class_name_parser": get_class_name_parser(api),
        "text_features_parser": get_text_features_parser(api),
    }
