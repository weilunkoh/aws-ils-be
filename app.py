import json
import logging
import os
from threading import Thread

import boto3
import flask_login
import torch
from dotenv import load_dotenv
from flask import Flask, send_file
from flask_cors import CORS
from flask_restx import Api, Resource, fields, reqparse
from src import helper_auth, helper_misc, helper_model, helper_parser, helper_user_admin

load_dotenv()

app: Flask = Flask(__name__)
login_manager = flask_login.LoginManager()
login_manager.init_app(app)

logging.basicConfig(level=logging.INFO)
app.config.SWAGGER_UI_DOC_EXPANSION = "list"

# S3 Stuff
app.s3_client = boto3.client("s3")
app.s3_resource = boto3.resource("s3")
app.aws_ils_records_bucket = app.s3_resource.Bucket(os.getenv("BUCKET_RECORDS"))
app.aws_ils_auth_bucket = app.s3_resource.Bucket(os.getenv("BUCKET_AUTH"))
app.aws_ils_model_bucket = app.s3_resource.Bucket(os.getenv("BUCKET_MODEL"))
app.aws_ils_background_bucket = app.s3_resource.Bucket(os.getenv("BUCKET_BACKGROUND"))
app.auth_roles = helper_auth.get_all_roles(app.aws_ils_auth_bucket)
app.auth_rights_mapping = helper_auth.get_all_rights_mapping(app.aws_ils_auth_bucket)

# S3 models
helper_model.load_models_to_local(
    app.aws_ils_model_bucket,
    load_image=os.getenv("IMAGE_MODEL_HOST") == "local",
    load_text=os.getenv("TEXT_MODEL_HOST") == "local",
)

if os.getenv("IMAGE_MODEL_HOST") == "local":
    app.image_processor = helper_model.load_image_processor()
    app.image_feature_extractor = helper_model.load_image_feature_extractor()
else:
    app.image_processor = None
    app.image_feature_extractor = os.getenv("IMAGE_SAGEMAKER_ENDPOINT_NAME")

if os.getenv("TEXT_MODEL_HOST") == "local":
    app.text_tokenizer = helper_model.load_text_tokenizer()
    app.text_feature_extractor = helper_model.load_text_feature_extractor()
else:
    app.text_tokenizer = None
    app.text_feature_extractor = os.getenv("TEXT_SAGEMAKER_ENDPOINT_NAME")

# SES Stuff
app.ses_client = boto3.client("ses", region_name=os.getenv("AWS_DEFAULT_REGION"))
app.source_email = "no-reply@smuils.jklmgroup.com"
app.welcome_template = "welcome_email_change_pw"
app.reset_pw_template = "reset_pw_email"
app.allowed_files = [".png", ".jpg", ".jpeg"]


@login_manager.request_loader
def request_loader(request):
    return helper_auth.authenticate_request(
        request, app.aws_ils_auth_bucket, app.auth_roles, app.auth_rights_mapping
    )


CORS(app)

api = Api(
    app,
    version="1.0",
    title="REST APIs for Model Prediction",
    description="Documentation and quick testing page for REST APIs that are created for model prediction.",
)
app.parsers = helper_parser.get_all_parsers(api)


def handle_error_message(err: Exception):
    print(err)
    return {"status": "error", "type": err.__class__.__name__, "message": str(err)}


@api.route("/host_message")
@api.doc(description=f"Host message regarding deployment platform.")
class HostMessage(Resource):
    def get(self):
        try:
            host_message = os.getenv("HOST_MSG")
            return {"status": "success", "host_message": host_message}, 200
        except Exception as err:
            return handle_error_message(err), 500


@api.route("/login")
@api.doc(description=f"Authenticates a user to access ILS.")
class Login(Resource):
    @api.expect(app.parsers["login_parser"])
    def post(self):
        args = app.parsers["login_parser"].parse_args()
        try:
            userID = helper_misc.parse_arg_str(args["userID"])
            password = helper_misc.parse_arg_str(args["password"])

            user_exists, user_object = helper_auth.check_user(
                app.aws_ils_auth_bucket, userID, password
            )
            access_rights_with_password_change_required = {
                "password_change_required": user_object["password_change_required"],
                **helper_auth.get_access_rights(user_object, app.auth_roles),
            }
            if user_exists:
                return {
                    "status": "success",
                    "results": access_rights_with_password_change_required,
                }, 200
            else:
                return {"status": "error", "message": "Authentication failed."}, 401
        except Exception as err:
            return handle_error_message(err), 500


@api.route("/inference")
@api.doc(
    description=f"Produce a classification prediction for an image and update with feedback when given."
)
class Inference(Resource):
    @flask_login.login_required
    @api.expect(app.parsers["inference_parser"])
    def post(self):
        args = app.parsers["inference_parser"].parse_args()
        try:
            userID = flask_login.current_user.id
            uploaded_file = args["file"]
            top_k = args["top_k"]
            if top_k is None:
                top_k = 1
            if any(
                uploaded_file.filename.lower().endswith(prefix)
                for prefix in app.allowed_files
            ):
                image_bytes = uploaded_file.read()
                classifier = helper_model.load_classifier(app.aws_ils_model_bucket)
                model = {
                    "feature_extractor": app.image_feature_extractor,
                    "classifier": classifier,
                }
                predictions = helper_model.get_prediction(
                    app.image_processor, model, image_bytes, top_k
                )
                timestamp_ms = helper_misc.get_timestamp_ms()
                helper_model.save_prediction_records_to_bucket(
                    userID,
                    image_bytes,
                    predictions,
                    timestamp_ms,
                    app.aws_ils_records_bucket,
                )
                return {
                    "status": "success",
                    "results": {
                        "predictions": predictions,
                        "timestamp_ms": timestamp_ms,
                    },
                }, 200
            else:
                return {
                    "status": "error",
                    "message": f"Only {' '.join(app.allowed_files)} files are allowed.",
                }, 400
        except Exception as err:
            return handle_error_message(err), 500

    @flask_login.login_required
    @api.expect(app.parsers["feedback_parser"])
    def put(self):
        args = app.parsers["feedback_parser"].parse_args()
        try:
            userID = flask_login.current_user.id
            timestamp = args["timestamp"]
            prediction_record = helper_model.get_prediction_record_from_bucket(
                app.aws_ils_records_bucket, userID, timestamp
            )
            prediction_record["feedback_correct"] = args["feedback_correct"] == "true"
            prediction_record["feedback_remarks"] = helper_misc.parse_arg_str(
                args["feedback_remarks"]
            )

            json_record = json.dumps(prediction_record, indent=4)
            app.aws_ils_records_bucket.put_object(
                Key=f"{userID}/{timestamp}", Body=json_record
            )
            return {"status": "success", "results": True}, 200
        except Exception as err:
            return handle_error_message(err), 500


@api.route("/evaluation")
@api.doc(
    description=f"Predict zipped set of images and that has the folder name as ground truth."
)
class Evaluation(Resource):
    @flask_login.login_required
    @api.expect(app.parsers["background_get_parser"])
    def get(self):
        args = app.parsers["background_get_parser"].parse_args()
        try:
            userID = flask_login.current_user.id
            timestamp = args["timestamp"]
            eval_bytes = helper_model.get_batch_evaluation_record_from_bucket(
                app.s3_client, app.aws_ils_background_bucket, userID, timestamp
            )

            if eval_bytes is None:
                return {
                    "status": "error",
                    "message": f"Record {timestamp} not found.",
                }, 400
            else:
                return send_file(
                    eval_bytes,
                    mimetype="application/json",
                    as_attachment=True,
                    download_name=f"batch_evaluation_{timestamp}.json",
                )
        except Exception as err:
            return handle_error_message(err), 500

    @flask_login.login_required
    @api.expect(app.parsers["evaluation_parser"])
    def post(self):
        args = app.parsers["evaluation_parser"].parse_args()
        try:
            userID = flask_login.current_user.id
            uploaded_file = args["file"]
            if uploaded_file.filename.lower().endswith(".zip"):
                timestamp_ms = helper_misc.get_timestamp_ms()
                zip_bytes = uploaded_file.read()
                top_k = args["top_k"]
                if top_k is None:
                    top_k = 1

                thread = Thread(
                    target=helper_model.batch_evaluation,
                    args=(
                        app.image_processor,
                        app.image_feature_extractor,
                        zip_bytes,
                        top_k,
                        userID,
                        timestamp_ms,
                        app.aws_ils_model_bucket,
                        app.aws_ils_records_bucket,
                        app.aws_ils_background_bucket,
                    ),
                )
                thread.start()

                return {
                    "status": "success",
                    "results": {
                        "timestamp_ms": timestamp_ms,
                        "message": "Triggered batch evaluation job.",
                    },
                }, 200
            else:
                return {
                    "status": "error",
                    "message": f"Only .zip files are allowed.",
                }, 400
        except Exception as err:
            return handle_error_message(err), 500


@api.route("/all_records")
@api.doc(
    description=f"View records for all users. (e.g. past predictions, batch evaluations, training jobs, etc.)"
)
class AllRecords(Resource):
    @flask_login.login_required
    @api.expect(app.parsers["all_records_get_parser"])
    def get(self):
        args = app.parsers["all_records_get_parser"].parse_args()
        try:
            record_type = helper_misc.parse_arg_str(args["record_type"])
            username = helper_misc.parse_arg_str(args["username"])
            timestamp = helper_misc.parse_arg_str(args["timestamp"])

            if record_type == "batch_evaluation":
                record_bytes = helper_model.get_batch_evaluation_record_from_bucket(
                    app.s3_client, app.aws_ils_background_bucket, username, timestamp
                )
            elif record_type == "batch_training":
                record_bytes = helper_model.get_batch_training_record_from_bucket(
                    app.s3_client, app.aws_ils_background_bucket, username, timestamp
                )
            elif record_type == "validation_features_extraction":
                record_bytes = helper_model.get_validation_record_from_bucket(
                    app.s3_client, app.aws_ils_background_bucket, username, timestamp
                )
            else:
                return {
                    "status": "error",
                    "message": f"Record type {record_type} not accepted.",
                }, 400

            if record_bytes is None:
                return {
                    "status": "error",
                    "message": f"Record {record_type}/{username}/{timestamp} not found.",
                }, 400
            else:
                return send_file(
                    record_bytes,
                    mimetype="application/json",
                    as_attachment=True,
                    download_name=f"{record_type}_{timestamp}.json",
                )
        except Exception as err:
            return handle_error_message(err), 500

    @flask_login.login_required
    @api.expect(app.parsers["records_parser"])
    def post(self):
        args = app.parsers["records_parser"].parse_args()
        try:
            record_type = helper_misc.parse_arg_str(args["record_type"])
            offset = helper_misc.parse_arg_str(args["offset"])
            limit = helper_misc.parse_arg_str(args["limit"])
            if record_type == "inference":
                (
                    prediction_records,
                    num_pages,
                    current_page,
                ) = helper_model.get_prediction_records_from_bucket(
                    app.aws_ils_records_bucket, offset=offset, limit=limit
                )
                return {
                    "status": "success",
                    "results": prediction_records,
                    "num_pages": num_pages,
                    "current_page": current_page,
                }, 200
            elif record_type in [
                "batch_evaluation",
                "batch_training",
                "pca",
                "validation_features_extraction",
            ]:
                (
                    batch_records,
                    num_pages,
                    current_page,
                ) = helper_model.get_records_from_background_bucket(
                    app.aws_ils_background_bucket,
                    record_type,
                    offset=offset,
                    limit=limit,
                )
                return {
                    "status": "success",
                    "results": batch_records,
                    "num_pages": num_pages,
                    "current_page": current_page,
                }, 200
            else:
                return {
                    "status": "error",
                    "message": f"Record type {record_type} not accepted.",
                }, 400
        except Exception as err:
            return handle_error_message(err), 500


@api.route("/individual_records")
@api.doc(
    description=f"View records for a specified user. (e.g. past predictions, batch evaluations, training jobs, etc.)"
)
class IndividualRecords(Resource):
    @flask_login.login_required
    @api.expect(app.parsers["records_parser"])
    def post(self):
        args = app.parsers["records_parser"].parse_args()
        try:
            userID = flask_login.current_user.id
            record_type = helper_misc.parse_arg_str(args["record_type"])
            offset = helper_misc.parse_arg_str(args["offset"])
            limit = helper_misc.parse_arg_str(args["limit"])
            if record_type == "inference":
                (
                    prediction_records,
                    num_pages,
                    current_page,
                ) = helper_model.get_prediction_records_from_bucket(
                    app.aws_ils_records_bucket, user=userID, offset=offset, limit=limit
                )
                return {
                    "status": "success",
                    "results": prediction_records,
                    "num_pages": num_pages,
                    "current_page": current_page,
                }, 200
            elif record_type in [
                "batch_evaluation",
                "batch_training",
                "validation_features_extraction",
            ]:
                (
                    batch_records,
                    num_pages,
                    current_page,
                ) = helper_model.get_records_from_background_bucket(
                    app.aws_ils_background_bucket,
                    record_type,
                    user=userID,
                    offset=offset,
                    limit=limit,
                )
                return {
                    "status": "success",
                    "results": batch_records,
                    "num_pages": num_pages,
                    "current_page": current_page,
                }, 200
            else:
                return {
                    "status": "error",
                    "message": f"Record type {record_type} not accepted.",
                }, 400
        except Exception as err:
            return handle_error_message(err), 500


@api.route("/training")
@api.doc(
    description=f"Training functionalities to manage images, extract features of images, compute centroids, and fit classifier."
)
class Training(Resource):
    @flask_login.login_required
    @api.expect(app.parsers["background_get_parser"])
    def get(self):
        args = app.parsers["background_get_parser"].parse_args()
        try:
            userID = flask_login.current_user.id
            timestamp = args["timestamp"]
            train_bytes = helper_model.get_batch_training_record_from_bucket(
                app.s3_client, app.aws_ils_background_bucket, userID, timestamp
            )

            if train_bytes is None:
                return {
                    "status": "error",
                    "message": f"Record {timestamp} not found.",
                }, 400
            else:
                return send_file(
                    train_bytes,
                    mimetype="application/json",
                    as_attachment=True,
                    download_name=f"batch_training_{timestamp}.json",
                )
        except Exception as err:
            return handle_error_message(err), 500

    @flask_login.login_required
    @api.expect(app.parsers["training_post_parser"])
    def post(self):
        args = app.parsers["training_post_parser"].parse_args()
        if args["training_file"] is None and args["validation_file"] is None:
            return {
                "status": "error",
                "message": f"Either training_file or validation_file must be provided.",
            }, 400

        try:
            userID = flask_login.current_user.id
            if args["training_file"] is not None:
                if not args["training_file"].filename.lower().endswith(".zip"):
                    return {
                        "status": "error",
                        "message": f"Only .zip files are allowed.",
                    }, 400
            if args["validation_file"] is not None:
                if not args["validation_file"].filename.lower().endswith(".zip"):
                    return {
                        "status": "error",
                        "message": f"Only .zip files are allowed.",
                    }, 400

            if args["training_file"] is not None:
                training_file = args["training_file"]
                training_zip_bytes = training_file.read()

                meet_criteria, class_name_counts = helper_model.check_num_images_in_zip(
                    training_zip_bytes,
                    app.aws_ils_background_bucket,
                )

                if meet_criteria:
                    training_timestamp_ms = helper_misc.get_timestamp_ms()
                    thread = Thread(
                        target=helper_model.batch_training,
                        args=(
                            app.image_processor,
                            app.image_feature_extractor,
                            app.text_tokenizer,
                            app.text_feature_extractor,
                            training_zip_bytes,
                            userID,
                            training_timestamp_ms,
                            app.aws_ils_background_bucket,
                            app.aws_ils_model_bucket,
                        ),
                    )
                    thread.start()
                else:
                    # TODO 2.1.0
                    # err_msg = "Number of images in zip file does not meet criteria."
                    err_msg1 = "Number of images or number of folders"
                    err_msg2 = "in zip file does not meet criteria."
                    err_msg = f"{err_msg1} {err_msg2}"

                    return {
                        "status": "error",
                        "message": err_msg,
                        "class_name_counts": class_name_counts,
                    }, 400

            if args["validation_file"] is not None:
                val_file = args["validation_file"]
                val_zip_bytes = val_file.read()
                val_timestamp_ms = helper_misc.get_timestamp_ms()
                thread = Thread(
                    target=helper_model.validation_features_extraction,
                    args=(
                        app.image_processor,
                        app.image_feature_extractor,
                        app.text_tokenizer,
                        app.text_feature_extractor,
                        val_zip_bytes,
                        userID,
                        val_timestamp_ms,
                        app.aws_ils_background_bucket,
                    ),
                )
                thread.start()

            if (
                args["validation_file"] is not None
                and args["training_file"] is not None
            ):
                message = "Triggered batch training job"
                message += "and validation features extraction job."
                return {
                    "status": "success",
                    "results": {
                        "training_timestamp_ms": training_timestamp_ms,
                        "validation_timestamp_ms": val_timestamp_ms,
                        "message": message,
                    },
                }, 200
            elif args["training_file"] is not None:
                message = "Triggered batch training job."
                return {
                    "status": "success",
                    "results": {
                        "training_timestamp_ms": training_timestamp_ms,
                        "message": message,
                    },
                }, 200
            else:
                message = "Triggered validation features extraction job."
                return {
                    "status": "success",
                    "results": {
                        "validation_timestamp_ms": val_timestamp_ms,
                        "message": message,
                    },
                }, 200

        except Exception as err:
            return handle_error_message(err), 500

    @flask_login.login_required
    @api.expect(app.parsers["training_update_parser"])
    def put(self):
        args = app.parsers["training_update_parser"].parse_args()
        try:
            if (
                args["original_userID"] is not None
                and args["original_timestamp_ms"] is None
            ):
                return {
                    "status": "error",
                    "message": f"original: timestamp_ms must be provided if userID is provided.",
                }, 400

            if (
                args["original_userID"] is None
                and args["original_timestamp_ms"] is not None
            ):
                return {
                    "status": "error",
                    "message": f"original: userID must be provided if timestamp_ms is provided.",
                }, 400

            update_to_used = args["update_to_used"]
            update_to_unused = args["update_to_unused"]

            if update_to_used is None and update_to_unused is None:
                return {
                    "status": "error",
                    "message": f"Either update_to_used or update_to_unused must be provided.",
                }, 400

            if args["original_userID"] is None:
                userID = flask_login.current_user.id
                timestamp_ms = helper_misc.get_timestamp_ms()
                review_userID = None
                review_timestamp_ms = None
                update_existing_job = False
            else:
                userID = args["original_userID"]
                timestamp_ms = args["original_timestamp_ms"]
                review_userID = flask_login.current_user.id
                review_timestamp_ms = helper_misc.get_timestamp_ms()
                update_existing_job = True

            if update_to_used is None:
                update_to_used = []
            if update_to_unused is None:
                update_to_unused = []

            thread = Thread(
                target=helper_model.update_images_and_batch_training,
                args=(
                    update_to_used,
                    update_to_unused,
                    userID,
                    timestamp_ms,
                    app.aws_ils_background_bucket,
                    app.aws_ils_model_bucket,
                    update_existing_job,
                    review_userID,
                    review_timestamp_ms,
                    args["target_class_name"],
                ),
            )
            thread.start()

            return {
                "status": "success",
                "results": {
                    "timestamp_ms": timestamp_ms,
                    "message": "Triggered update images and batch training job.",
                },
            }, 200
        except Exception as err:
            return handle_error_message(err), 500


@api.route("/validation")
@api.doc(description="To get details of validation feature extraction jobs.")
class Validation(Resource):
    @flask_login.login_required
    @api.expect(app.parsers["background_get_parser"])
    def get(self):
        args = app.parsers["background_get_parser"].parse_args()
        try:
            userID = flask_login.current_user.id
            timestamp = args["timestamp"]
            val_bytes = (
                helper_model.get_validation_features_extraction_record_from_bucket(
                    app.s3_client, app.aws_ils_background_bucket, userID, timestamp
                )
            )

            if val_bytes is None:
                return {
                    "status": "error",
                    "message": f"Record {timestamp} not found.",
                }, 400
            else:
                return send_file(
                    val_bytes,
                    mimetype="application/json",
                    as_attachment=True,
                    download_name=f"validation_features_extraction_{timestamp}.json",
                )
        except Exception as err:
            return handle_error_message(err), 500


@api.route("/text_features")
@api.doc(description="To get list and details of text features.")
class TextFeatures(Resource):
    @flask_login.login_required
    @api.expect(app.parsers["class_name_parser"])
    def get(self):
        args = app.parsers["class_name_parser"].parse_args()
        try:
            class_name = args["class_name"]
            feature_bytes = helper_model.get_text_features_record_from_bucket(
                app.s3_client, app.aws_ils_background_bucket, class_name
            )

            if feature_bytes is None:
                return {
                    "status": "error",
                    "message": f"Record {class_name} not found.",
                }, 400
            else:
                return send_file(
                    feature_bytes,
                    mimetype="application/json",
                    as_attachment=True,
                    download_name=f"text_features_{class_name}.json",
                )
        except Exception as err:
            return handle_error_message(err), 500

    @flask_login.login_required
    @api.expect(app.parsers["text_features_parser"])
    def post(self):
        args = app.parsers["text_features_parser"].parse_args()
        try:
            offset = helper_misc.parse_arg_str(args["offset"])
            limit = helper_misc.parse_arg_str(args["limit"])
            (
                batch_records,
                num_pages,
                current_page,
            ) = helper_model.get_text_features_from_background_bucket(
                app.aws_ils_background_bucket,
                offset=offset,
                limit=limit,
            )
            return {
                "status": "success",
                "results": batch_records,
                "num_pages": num_pages,
                "current_page": current_page,
            }, 200
        except Exception as err:
            return handle_error_message(err), 500


@api.route("/visualisation")
@api.doc(
    description=f"Visualisation functionalities for centroids and training images."
)
class Visualisation(Resource):
    @flask_login.login_required
    @api.expect(app.parsers["visualisation_get_parser"])
    def get(self):
        args = app.parsers["visualisation_get_parser"].parse_args()
        try:
            username = helper_misc.parse_arg_str(args["username"])
            timestamp = args["timestamp"]
            return_type = helper_misc.parse_arg_str(args["return_type"])

            if return_type not in ["file", "json"]:
                return {
                    "status": "error",
                    "message": f"Return type {return_type} not accepted.",
                }, 400

            visualisation_bytes = helper_model.get_visualisation_record_from_bucket(
                app.s3_client, app.aws_ils_background_bucket, username, timestamp
            )

            if visualisation_bytes is None:
                return {
                    "status": "error",
                    "message": f"Record {timestamp} not found.",
                }, 400
            else:
                if return_type == "file":
                    return send_file(
                        visualisation_bytes,
                        mimetype="application/json",
                        as_attachment=True,
                        download_name=f"visualisation_{timestamp}.json",
                    )
                else:
                    return {
                        "status": "success",
                        "results": json.loads(visualisation_bytes.read().decode()),
                    }, 200
        except Exception as err:
            return handle_error_message(err), 500

    @flask_login.login_required
    @api.expect(app.parsers["visualisation_post_parser"])
    def post(self):
        args = app.parsers["visualisation_post_parser"].parse_args()
        try:
            username = flask_login.current_user.id
            centroids = args["centroids"] if args["centroids"] is not None else []
            used_training_images = (
                args["used_training_images"]
                if args["used_training_images"] is not None
                else []
            )
            unused_training_images = (
                args["unused_training_images"]
                if args["unused_training_images"] is not None
                else []
            )
            validation_images = (
                args["validation_images"]
                if args["validation_images"] is not None
                else []
            )
            class_names = args["class_names"] if args["class_names"] is not None else []

            timestamp_ms = helper_misc.get_timestamp_ms()
            thread = Thread(
                target=helper_model.plot_pca,
                args=(
                    timestamp_ms,
                    username,
                    app.aws_ils_background_bucket,
                    centroids,
                    used_training_images,
                    unused_training_images,
                    validation_images,
                    class_names,
                ),
            )
            thread.start()

            return {
                "status": "success",
                "results": {
                    "timestamp_ms": timestamp_ms,
                    "message": "Triggered visualisation job.",
                },
            }, 200
        except Exception as err:
            return handle_error_message(err), 500

    @flask_login.login_required
    @api.expect(app.parsers["visualisation_put_parser"])
    def put(self):
        args = app.parsers["visualisation_put_parser"].parse_args()
        try:
            pca_userID = args["pca_userID"]
            pca_timestamp_ms = args["pca_timestamp_ms"]
            update_status = helper_model.update_visualisation_record_about_manual(
                app.aws_ils_background_bucket, pca_userID, pca_timestamp_ms
            )
            pca_job_id = f"{pca_userID}/{pca_timestamp_ms}"
            if update_status:
                message = f"Updated visualisation record {pca_job_id}."
                return {
                    "status": "success",
                    "message": message,
                }, 200
            else:
                message = f"Visualisation record {pca_job_id} not updated."
                return {
                    "status": "error",
                    "message": message,
                }, 400
        except Exception as err:
            return handle_error_message(err), 500


@api.route("/image")
@api.doc(description=f"Getting training and validation images for visualisation.")
class Image(Resource):
    @api.expect(app.parsers["image_get_parser"])
    def get(self):
        args = app.parsers["image_get_parser"].parse_args()
        try:
            folder = helper_misc.parse_arg_str(args["folder"])
            image_class = helper_misc.parse_arg_str(args["image_class"])
            timestamp = helper_misc.parse_arg_str(args["timestamp"])

            img_io = helper_model.get_image(
                app.aws_ils_background_bucket, folder, image_class, timestamp
            )

            if img_io is None:
                return {
                    "status": "error",
                    "message": f"Record {timestamp} not found for class {image_class}.",
                }, 400
            else:
                return send_file(img_io, mimetype="image/png", as_attachment=False)
        except Exception as err:
            return handle_error_message(err), 500


@api.route("/user_admin")
@api.doc(description=f"User administration functionalities.")
class UserAdmin(Resource):
    @flask_login.login_required
    @api.expect(app.parsers["all_users_parser"])
    def get(self):
        args = app.parsers["all_users_parser"].parse_args()
        try:
            offset = helper_misc.parse_arg_str(args["offset"])
            limit = helper_misc.parse_arg_str(args["limit"])
            users, num_pages, current_page = helper_user_admin.get_all_users(
                app.aws_ils_auth_bucket, offset=offset, limit=limit
            )
            return {
                "status": "success",
                "results": users,
                "num_pages": num_pages,
                "current_page": current_page,
            }, 200
        except Exception as err:
            return handle_error_message(err), 500

    @flask_login.login_required
    @api.expect(app.parsers["new_user_parser"])
    def post(self):
        args = app.parsers["new_user_parser"].parse_args()
        try:
            username = helper_misc.parse_arg_str(args["username"])
            role = helper_misc.parse_arg_str(args["role"])

            # Check if username already exists in database.
            user_exists = helper_user_admin.get_user(app.aws_ils_auth_bucket, username)
            if user_exists:
                return {
                    "status": "error",
                    "message": f"Username {username} already exists.",
                }, 400

            # Add user to database
            password = helper_user_admin.generate_password()
            timestamp_ms = helper_misc.get_timestamp_ms()
            helper_user_admin.add_user(
                app.aws_ils_auth_bucket, username, role, password, timestamp_ms
            )

            # Send welcome email to user
            helper_user_admin.send_email(
                app.ses_client,
                app.source_email,
                username,
                app.welcome_template,
                username,
                password,
            )
            return {
                "status": "success",
                "message": f"Account created for {username} with notification sent.",
            }, 200
        except Exception as err:
            return handle_error_message(err), 500

    # For admin to update role of account or add acount to blacklist.
    @flask_login.login_required
    @api.expect(app.parsers["update_user_parser"])
    def put(self):
        args = app.parsers["update_user_parser"].parse_args()
        try:
            username = helper_misc.parse_arg_str(args["username"])
            role = helper_misc.parse_arg_str(args["role"])
            blacklist = helper_misc.parse_arg_str(args["blacklist"])
            active = helper_misc.parse_arg_str(args["active"])

            # Check for nonetype before converting string to boolean.
            if type(blacklist) == str:
                blacklist = blacklist.lower() == "true"
            if type(active) == str:
                active = active.lower() == "true"

            # Check if username already exists in database.
            user_exists, user_record = helper_user_admin.get_user(
                app.aws_ils_auth_bucket, username, return_record=True
            )
            if not user_exists:
                return {
                    "status": "error",
                    "message": f"Username {username} does not exist.",
                }, 400

            update_msg = ""
            if role is not None:
                user_record["role"] = role
                update_msg += f" Role: {role}"

            if blacklist is not None:
                user_record["blacklist"] = blacklist
                update_msg += f" Blacklist: {blacklist}"

            if active is not None:
                user_record["active"] = active
                update_msg += f" Active: {active}"

            # Update user in database
            helper_user_admin.update_user(
                app.aws_ils_auth_bucket, username, user_record
            )

            return {
                "status": "success",
                "message": f"User {username} updated.{update_msg}",
            }, 200
        except Exception as err:
            return handle_error_message(err), 500

    # For admin to soft delete account.
    @flask_login.login_required
    @api.expect(app.parsers["delete_user_parser"])
    def delete(self):
        args = app.parsers["delete_user_parser"].parse_args()
        try:
            username = helper_misc.parse_arg_str(args["username"])

            # Check if username already exists in database.
            user_exists, user_record = helper_user_admin.get_user(
                app.aws_ils_auth_bucket, username, return_record=True
            )
            if not user_exists:
                return {
                    "status": "error",
                    "message": f"Username {username} does not exist.",
                }, 400

            # Soft delete user in database
            helper_user_admin.delete_user(
                app.aws_ils_auth_bucket, username, user_record
            )

            return {
                "status": "success",
                "message": f"User {username} deleted.",
            }, 200
        except Exception as err:
            return handle_error_message(err), 500


@api.route("/password_change")
@api.doc(description=f"For handling password change.")
class PasswordChange(Resource):
    # For admin to reset password
    @flask_login.login_required
    @api.expect(app.parsers["pw_reset_parser"])
    def post(self):
        args = app.parsers["pw_reset_parser"].parse_args()
        try:
            username = helper_misc.parse_arg_str(args["username"])

            # Check if username already exists in database.
            user_exists, user_record = helper_user_admin.get_user(
                app.aws_ils_auth_bucket, username, return_record=True
            )
            if not user_exists:
                return {
                    "status": "error",
                    "message": f"Username {username} does not exist.",
                }, 400

            password = helper_user_admin.generate_password()
            hashed_password = helper_user_admin.hash_password(password)
            user_record["password"] = hashed_password
            user_record["password_change_required"] = True

            # Update user password in database
            helper_user_admin.update_user(
                app.aws_ils_auth_bucket, username, user_record
            )

            # Send reset password email to user
            helper_user_admin.send_email(
                app.ses_client,
                app.source_email,
                username,
                app.reset_pw_template,
                username,
                password,
            )

            return {
                "status": "success",
                "message": f"User {username} password reset.",
            }, 200
        except Exception as err:
            return handle_error_message(err), 500

    # For user to change password
    @flask_login.login_required
    @api.expect(app.parsers["pw_change_parser"])
    def put(self):
        args = app.parsers["pw_change_parser"].parse_args()
        try:
            userID = flask_login.current_user.id
            old_password = helper_misc.parse_arg_str(args["old_password"])
            hashed_password = helper_misc.parse_arg_str(args["new_password"])

            # Get user record from database.
            _, user_record = helper_user_admin.get_user(
                app.aws_ils_auth_bucket, userID, return_record=True
            )

            if user_record["password"] != old_password:
                return {
                    "status": "error",
                    "message": "Authentication failed.",
                }, 401

            # Update user password in database
            user_record["password"] = hashed_password
            user_record["password_change_required"] = False
            helper_user_admin.update_user(app.aws_ils_auth_bucket, userID, user_record)

            return {
                "status": "success",
                "message": f"Password updated sucessfully.",
            }, 200
        except Exception as err:
            return handle_error_message(err), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
