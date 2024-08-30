import base64
import json
import os
from io import BytesIO
from threading import Thread
from zipfile import ZipFile

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from botocore.exceptions import ClientError
from PIL import Image
from sagemaker.deserializers import JSONDeserializer
from sagemaker.pytorch import PyTorchPredictor
from sagemaker.serializers import JSONSerializer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from torchvision import models
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizerFast,
    CLIPVisionModelWithProjection,
)

from .helper_misc import (
    COS_SIM_PERC_THRESHOLD,
    DEFAULT_PAGE_SIZE,
    NEW_CLASS_PERC_TOLERANCE,
    NEW_EXISTING_PERC_TOLERANCE,
    check_offset_limit,
    get_timestamp_ms,
)


def load_models_to_local(bucket, load_image=True, load_text=True):
    print("Attempting downloading of feature extractor models from S3.")
    image_folders = ["clip_image_feature_extractor", "clip_image_processor"]
    text_folders = ["clip_text_feature_extractor", "clip_text_tokenizer"]
    if load_image and load_text:
        folders = [*image_folders, *text_folders]
    elif load_image:
        folders = image_folders
    elif load_text:
        folders = text_folders
    else:
        folders = []
        print("No need to download any models.")

    for folder in folders:
        if not os.path.exists(f"model/{folder}"):
            os.makedirs(f"model/{folder}", exist_ok=True)
            records = bucket.objects.filter(Prefix=f"model/{folder}/")
            for object_record in records:
                print(f"Downloading {object_record.key}.")
                response = object_record.get()
                with open(object_record.key, "wb") as f:
                    f.write(response["Body"].read())
                print(f"Downloaded {object_record.key}.")
        else:
            print(f"Folder 'model/{folder}' already exists.")


def load_image_processor():
    print("Loading image processor into memory.")
    image_processor = CLIPImageProcessor.from_pretrained("model/clip_image_processor")
    return image_processor


def load_image_feature_extractor():
    print("Loading image feature extractor into memory.")
    image_feature_extractor = CLIPVisionModelWithProjection.from_pretrained(
        "model/clip_image_feature_extractor"
    )
    return image_feature_extractor


def load_text_tokenizer():
    print("Loading text tokenizer into memory.")
    text_tokenizer = CLIPTokenizerFast.from_pretrained("model/clip_text_tokenizer")
    return text_tokenizer


def load_text_feature_extractor():
    print("Loading text feature extractor into memory.")
    text_feature_extractor = CLIPTextModelWithProjection.from_pretrained(
        "model/clip_text_feature_extractor"
    )
    return text_feature_extractor


def load_classifier(bucket):
    print("Loading classifier.")
    # Load classifier bytes from S3
    classifier_record = bucket.Object("model/knn_centroids_classifier")
    classifier_response = classifier_record.get()
    classifier_bytes = classifier_response["Body"].read()

    # Load classifier
    classifier = joblib.load(BytesIO(classifier_bytes))
    print("Classifier loaded.")
    return classifier


def resize_square(original_image, square_length=224):
    # Reference for maintaining aspect ratio
    # # Calculate the aspect ratio
    # aspect_ratio = original_image.width / original_image.height

    # # Determine the new width and height to fit within the specified max length
    # if original_image.width > original_image.height:
    #     new_width = max_length
    #     new_height = int(max_length / aspect_ratio)
    # else:
    #     new_width = int(max_length * aspect_ratio)
    #     new_height = max_length

    # Resize the image while ignoring aspect ratio
    resized_image = original_image.resize((square_length, square_length))
    return resized_image


def transform_image(image_processor, image_bytes: bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    if image_processor is None:
        resized_image = resize_square(image, square_length=224)
        with BytesIO() as f:
            resized_image.save(f, format="JPEG")
            image_bytes = f.getvalue()
        return base64.b64encode(image_bytes).decode("utf-8")
    else:
        inputs = image_processor(images=image, return_tensors="pt")
        return inputs.pixel_values


def get_img_features(img, feature_extractor):
    if type(feature_extractor) == str:
        predictor = PyTorchPredictor(
            endpoint_name=feature_extractor,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )
        return predictor.predict({"img_b64": img})
    else:
        feature_extractor.eval()
        with torch.no_grad():
            img_outputs = feature_extractor(img)
            return img_outputs.image_embeds[0].tolist()


def get_prediction(image_processor, model_dict: dict, image_bytes: bytes, top_k: int):
    class_names = model_dict["classifier"].classes_
    img = transform_image(image_processor, image_bytes=image_bytes)
    img_features = [get_img_features(img, model_dict["feature_extractor"])]

    pred_probas = model_dict["classifier"].predict_proba(img_features)[0]
    predicted_classes = {
        predicted_class: confidence
        for (predicted_class, confidence) in zip(class_names, pred_probas)
    }
    predicted_classes = {
        key: value
        for (key, value) in sorted(
            predicted_classes.items(), key=lambda x: x[1], reverse=True
        )
    }

    top_predictions = {
        key: value for (key, value) in list(predicted_classes.items())[:top_k]
    }
    return top_predictions


def save_prediction_records(
    userID: str,
    image_bytes: bytes,
    predictions: dict,
    timestamp_ms: int,
    bucket=None,
    feedback_correct: bool = None,
    feedback_remarks: str = None,
):
    file_path = "data/prediction_records.joblib"
    if os.path.exists(file_path):
        prediction_records = joblib.load(file_path)
    else:
        prediction_records = []

    record = {
        "userID": userID,
        "img64": base64.b64encode(image_bytes).decode("utf-8"),
        "predictions": predictions,
        "timestamp_ms": timestamp_ms,
        "feedback_correct": feedback_correct,
        "feedback_remarks": feedback_remarks,
    }
    prediction_records.append(record)

    if bucket is None:
        joblib.dump(prediction_records, file_path)
    else:
        json_record = json.dumps(record, indent=4)
        bucket.put_object(
            Key=f"{record['userID']}/{record['timestamp_ms']}", Body=json_record
        )


def save_prediction_records_to_bucket(
    userID: str,
    image_bytes: bytes,
    predictions: dict,
    timestamp_ms: int,
    bucket,
    feedback_correct: bool = None,
    feedback_remarks: str = None,
):
    save_prediction_records(
        userID,
        image_bytes,
        predictions,
        timestamp_ms,
        bucket,
        feedback_correct,
        feedback_remarks,
    )


def batch_evaluation(
    image_processor,
    image_feature_extractor,
    zip_bytes: bytes,
    top_k: int,
    userID: str,
    timestamp_ms: int,
    model_bucket,
    prediction_records_bucket,
    background_bucket,
):
    # Simple logging
    print(f"Batch evaluation started for {userID}/{timestamp_ms}.")

    # Initialise overall prediction record to be saved to S3
    json_record = json.dumps({"status": "in progress"}, indent=4)
    background_bucket.put_object(
        Key=f"batch_evaluation/{userID}/{timestamp_ms}", Body=json_record
    )
    overall_prediction = {
        "status": "completed",
        "top_k": top_k,
        "correct": {},
        "wrong": {},
        "skipped": {},
    }

    # Load classifier
    try:
        classifier = load_classifier(model_bucket)
        model_dict = {
            "feature_extractor": image_feature_extractor,
            "classifier": classifier,
        }
    except Exception as err:
        print(err)
        model_dict = None
        overall_prediction["status"] = "error_encountered"
        overall_prediction["skipped"]["classifier"] = "Classifier could not be loaded."

    # Commence batch evaluation
    if model_dict is not None:
        counter = 1
        with ZipFile(BytesIO(zip_bytes)) as zip_ref:
            for filename in zip_ref.namelist():
                if any(
                    filename.lower().endswith(prefix)
                    for prefix in [".png", ".jpg", ".jpeg"]
                ):
                    try:
                        with zip_ref.open(filename) as f:
                            image_bytes = f.read()
                    except Exception as err:
                        print(err)
                        image_bytes = None
                        overall_prediction["status"] = "error_encountered"
                        overall_prediction["skipped"][
                            filename
                        ] = "File could not be read."

                    if image_bytes is not None:
                        try:
                            predictions = get_prediction(
                                image_processor, model_dict, image_bytes, top_k
                            )
                            if any(
                                filename.lower().startswith(suffix)
                                for suffix in list(predictions.keys())[:top_k]
                            ):
                                overall_prediction["correct"][filename] = predictions
                                feedback_correct = True
                                feedback_remarks = None
                            else:
                                overall_prediction["wrong"][filename] = predictions
                                feedback_correct = False
                                feedback_remarks = filename.split("/")[0]

                            save_prediction_records_to_bucket(
                                userID,
                                image_bytes,
                                predictions,
                                f"{timestamp_ms}_{counter}",
                                prediction_records_bucket,
                                feedback_correct,
                                feedback_remarks,
                            )
                            counter += 1
                        except Exception as err:
                            print(err)
                            overall_prediction["status"] = "error_encountered"
                            overall_prediction["skipped"][
                                filename
                            ] = "Prediction failed."
                else:
                    overall_prediction["status"] = "error_encountered"
                    overall_prediction["skipped"][
                        filename
                    ] = "File extension not supported."

    # Save overall_prediction to S3
    json_record = json.dumps(overall_prediction, indent=4)
    background_bucket.put_object(
        Key=f"batch_evaluation/{userID}/{timestamp_ms}", Body=json_record
    )

    # Simple logging
    print(f"Batch evaluation completed for {userID}/{timestamp_ms}.")


def get_batch_evaluation_record_from_bucket(client, bucket, user, timestamp):
    try:
        object_key = f"batch_evaluation/{user}/{timestamp}"
        file_object = BytesIO()
        client.download_fileobj(bucket.name, object_key, file_object)
        file_object.seek(0)
        return file_object
    except ClientError:
        return None


def get_prediction_records_from_bucket(bucket, user=None, offset=None, limit=None):
    offset, limit = check_offset_limit(offset, limit)
    prediction_records = []
    if user is None:
        # records = bucket.objects.all().page_size(limit)
        records = bucket.objects.all()
    else:
        # records = bucket.objects.filter(Prefix=f"{user}/").page_size(limit)
        records = bucket.objects.filter(Prefix=f"{user}/")

    # page_idx = offset // limit
    # pages = list(records.pages())
    # len_pages = len(pages)
    # for object_record in pages[page_idx]:

    records = sorted(
        records,
        key=lambda x: (
            x.key.split("/")[-1].split("_")[0],
            x.key.split("/")[-1].split("_")[-1],
        ),
        reverse=True,
    )
    page_idx = offset // limit
    num_records = len(records)
    if num_records % limit == 0:
        len_pages = len(records) // limit
    else:
        len_pages = len(records) // limit + 1

    for object_record in records[offset : offset + limit]:
        response = object_record.get()
        json_record = response["Body"].read().decode()
        record = json.loads(json_record)
        prediction_records.append(record)

    page_num = page_idx + 1
    return prediction_records, len_pages, page_num


def get_prediction_record_from_bucket(bucket, user, timestamp):
    object_key = f"{user}/{timestamp}"
    object_record = bucket.Object(object_key)
    response = object_record.get()
    json_record = response["Body"].read().decode()
    prediction_record = json.loads(json_record)
    return prediction_record


def get_text_features_from_background_bucket(bucket, offset=None, limit=None):
    offset, limit = check_offset_limit(offset, limit)
    object_records = bucket.objects.filter(Prefix="text_features/")

    object_records = sorted(object_records, key=lambda x: x.key)
    records = []
    page_idx = offset // limit
    num_records = len(object_records)
    if num_records % limit == 0:
        len_pages = len(object_records) // limit
    else:
        len_pages = len(object_records) // limit + 1

    for object_record in object_records[offset : offset + limit]:
        record = {}
        record["class_name"] = object_record.key.split("/")[-1]
        records.append(record)
    page_num = page_idx + 1
    return records, len_pages, page_num


def get_records_from_background_bucket(
    bucket, record_type, user=None, offset=None, limit=None
):
    offset, limit = check_offset_limit(offset, limit)
    if user is None:
        # object_records = bucket.objects.filter(Prefix=f"{record_type}/").page_size(
        #     limit
        # )
        object_records = bucket.objects.filter(Prefix=f"{record_type}/")
    else:
        # object_records = bucket.objects.filter(
        #     Prefix=f"{record_type}/{user}/"
        # ).page_size(limit)
        object_records = bucket.objects.filter(Prefix=f"{record_type}/{user}/")

    # records = []
    # page_idx = offset // limit
    # pages = list(object_records.pages())
    # len_pages = len(pages)
    # for object_record in pages[page_idx]:

    records = []
    object_records = sorted(
        object_records, key=lambda x: x.key.split("/")[-1], reverse=True
    )
    page_idx = offset // limit
    num_records = len(object_records)
    if num_records % limit == 0:
        len_pages = len(object_records) // limit
    else:
        len_pages = len(object_records) // limit + 1

    for object_record in object_records[offset : offset + limit]:
        print(object_record)
        response = object_record.get()
        json_record = response["Body"].read().decode()
        raw_record = json.loads(json_record)
        object_record_key_split = object_record.key.split("/")

        record = {}
        record["username"] = object_record_key_split[1]
        record["timestamp"] = int(object_record_key_split[2])
        record["status"] = raw_record["status"]

        if record_type == "batch_evaluation":
            if record["status"] != "in progress":
                record["top_k"] = raw_record["top_k"]
                record["num_correct"] = len(raw_record["correct"])
                record["num_wrong"] = len(raw_record["wrong"])
                record["num_skipped"] = len(raw_record["skipped"])
            else:
                record["top_k"] = None
                record["num_correct"] = None
                record["num_wrong"] = None
                record["num_skipped"] = None
        elif record_type == "batch_training":
            if record["status"] != "in progress":
                class_names = set()
                if "features_success" in raw_record:
                    record["num_features_extracted"] = len(
                        raw_record["features_success"]
                    )
                    class_names = set(
                        [x.split("/")[0] for x in raw_record["features_success"].keys()]
                    )
                if "features_skipped" in raw_record:
                    record["num_features_skipped"] = len(raw_record["features_skipped"])
                    class_names = class_names | set(
                        [x.split("/")[0] for x in raw_record["features_skipped"].keys()]
                    )
                if "failed_used_list" in raw_record:
                    record["num_used_failed"] = len(raw_record["failed_used_list"])
                    # class_names = class_names | set(
                    #     [x.split("/")[1] for x in raw_record["failed_used_list"]]
                    # )
                if "failed_unused_list" in raw_record:
                    record["num_unused_failed"] = len(raw_record["failed_unused_list"])
                    # class_names = class_names | set(
                    #     [x.split("/")[1] for x in raw_record["failed_unused_list"]]
                    # )
                if "retrained_classes" in raw_record:
                    record["retrained_classes"] = raw_record["retrained_classes"]
                    # class_names = class_names | set(raw_record["retrained_classes"])
                record["classes"] = sorted(list(class_names))
                record["centroids_computed"] = raw_record["centroids_computed"]
                record["classifier_fitted"] = raw_record["classifier_fitted"]

                if "manual_review_supervisor" in raw_record:
                    record["manual_review_supervisor"] = raw_record[
                        "manual_review_supervisor"
                    ]
                if "manual_review_date" in raw_record:
                    record["manual_review_date"] = raw_record["manual_review_date"]
            else:
                record["num_features_extracted"] = None
                record["num_features_skipped"] = None
                record["centroids_computed"] = None
                record["classifier_fitted"] = None
                record["classes"] = []
        elif record_type == "pca":
            if "centroid_dict" in raw_record:
                record["centroids"] = sorted(list(raw_record["centroid_dict"].keys()))
            else:
                record["centroids"] = []
            if "used_training_image_dict" in raw_record:
                record["used_training_images"] = sorted(
                    list(raw_record["used_training_image_dict"].keys())
                )
            else:
                record["used_training_images"] = []
            if "unused_training_image_dict" in raw_record:
                record["unused_training_images"] = sorted(
                    list(raw_record["unused_training_image_dict"].keys())
                )
            else:
                record["unused_training_images"] = []
            if "validation_image_dict" in raw_record:
                record["validation_images"] = sorted(
                    list(raw_record["validation_image_dict"].keys())
                )
            else:
                record["validation_images"] = []
            if "class_name_dict" in raw_record:
                record["class_names"] = sorted(
                    list(raw_record["class_name_dict"].keys())
                )
            else:
                record["class_names"] = []
            if "new_class_image_dict" in raw_record:
                record["new_class_images"] = sorted(
                    list(raw_record["new_class_image_dict"].keys())
                )
            else:
                record["new_class_images"] = []
            if "failed_validation_image_dict" in raw_record:
                record["failed_validation_images"] = sorted(
                    list(raw_record["failed_validation_image_dict"].keys())
                )
            else:
                record["failed_validation_images"] = []
            if "passed_new_image_dict" in raw_record:
                record["passed_new_images"] = sorted(
                    list(raw_record["passed_new_image_dict"].keys())
                )
            else:
                record["passed_new_images"] = []
            if "failed_new_image_dict" in raw_record:
                record["failed_new_images"] = sorted(
                    list(raw_record["failed_new_image_dict"].keys())
                )
            else:
                record["failed_new_images"] = []
            if "display_message" in raw_record:
                record["display_message"] = raw_record["display_message"]
            if "batch_training_userID" in raw_record:
                record["batch_training_userID"] = raw_record["batch_training_userID"]
        elif record_type == "validation_features_extraction":
            if record["status"] != "in progress":
                record["num_features_extracted"] = len(raw_record["features_success"])
                record["num_features_skipped"] = len(raw_record["features_skipped"])
                record["classes"] = []
                class_names = set(
                    [x.split("/")[0] for x in raw_record["features_success"].keys()]
                ) | set(
                    [x.split("/")[0] for x in raw_record["features_skipped"].keys()]
                )
                record["classes"] = sorted(list(class_names))
            else:
                record["num_features_extracted"] = None
                record["num_features_skipped"] = None
                record["classes"] = []
        else:
            raise ValueError("Invalid record_type.")
        records.append(record)
    page_num = page_idx + 1
    return records, len_pages, page_num


def check_num_images_in_zip(zip_bytes: bytes, background_bucket):
    meet_criteria = True
    with ZipFile(BytesIO(zip_bytes)) as zip_ref:
        class_name_occurrences = [f.split("/")[0] for f in zip_ref.namelist()]
        class_names_text = list(set(class_name_occurrences))
    class_name_counts = {}

    # TODO 2.1.0: Allow multiple files in v2.1.0
    if len(class_names_text) > 1:
        meet_criteria = False
        class_name_counts = {"number of folders": len(class_names_text)}
        return meet_criteria, class_name_counts

    for class_name in class_names_text:
        records = background_bucket.objects.filter(
            Prefix=f"training_images/{class_name}"
        )
        num_images = len(list(records))
        if num_images == 0:
            count = class_name_occurrences.count(class_name)
            class_name_counts[class_name] = count
            # Check if there are at least 100 occurrences of class_name in zip_bytes
            if count < 100:
                meet_criteria = False

    return meet_criteria, class_name_counts


def extract_and_upload_text_features(
    bucket,
    class_name: str,
    text_tokenizer,
    text_feature_extractor,
    userID: str,
):
    # Check if class_name already exists in bucket
    records = bucket.objects.filter(Prefix=f"text_features/{class_name}")
    if len(list(records)) > 0:
        print(f"Text features for {class_name} already exists in bucket.")
        return
    if text_tokenizer is None or type(text_feature_extractor) == str:
        text_predictor = PyTorchPredictor(
            endpoint_name=text_feature_extractor,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )
        text_features = text_predictor.predict({"class_name": class_name})
    else:
        text_inputs = text_tokenizer(class_name, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_outputs = text_feature_extractor(
                text_inputs.input_ids, text_inputs.attention_mask
            )
            text_features = text_outputs.text_embeds[0].tolist()

    json_record = json.dumps(text_features, indent=4)
    bucket.put_object(Key=f"text_features/{class_name}", Body=json_record)


def extract_and_upload_img_features(
    bucket,
    class_name: str,
    image_processor,
    image_feature_extractor,
    image_bytes: bytes,
    userID: str,
    original_filename: str,
    folder_name: str,
    robustness_check: bool = False,
):
    img = transform_image(image_processor, image_bytes=image_bytes)
    img_features = get_img_features(img, image_feature_extractor)

    feature_record = {
        "userID": userID,
        "img64": base64.b64encode(image_bytes).decode("utf-8"),
        "img_features": img_features,
        "original_filename": original_filename,
    }

    json_record = json.dumps(feature_record, indent=4)
    timestamp_ms = get_timestamp_ms()

    # Put in training/validation/pending folder
    bucket.put_object(
        Key=f"{folder_name}/{class_name}/{timestamp_ms}", Body=json_record
    )

    if robustness_check:
        return f"{class_name}/{timestamp_ms}"


def do_robustness_check(
    batch_training_timestamp_ms: int,
    batch_training_userID: str,
    timestamp_ids: list,
    background_bucket,
):
    class_names_text = list(set([f.split("/")[0] for f in timestamp_ids]))
    class_names_cos_sim = {k: [] for k in class_names_text}
    class_name_keys = {k: [] for k in class_names_text}  # used by existing class check
    class_name_text_features = {}
    class_name_robustness_check = {k: True for k in class_names_text}

    for timestamp_id in timestamp_ids:
        object_record = background_bucket.Object(f"pending_images/{timestamp_id}")
        response = object_record.get()
        json_record = response["Body"].read().decode()
        record = json.loads(json_record)
        class_name = timestamp_id.split("/")[0]
        pending_img_features = np.array(record["img_features"])

        if class_name not in class_name_text_features:
            object_record = background_bucket.Object(f"text_features/{class_name}")
            response = object_record.get()
            json_record = response["Body"].read().decode()
            record = json.loads(json_record)
            class_name_text_features[class_name] = record
        text_features = np.array(class_name_text_features[class_name])

        cos_sim = np.dot(pending_img_features, text_features) / (
            np.linalg.norm(pending_img_features) * np.linalg.norm(text_features)
        )
        class_names_cos_sim[class_name].append(cos_sim)
        class_name_keys[class_name].append(timestamp_id)

    for class_name in class_names_cos_sim:
        # Get text features of class_name
        text_features = class_name_text_features[class_name]

        # Check if class_name exists in training_images
        records = background_bucket.objects.filter(
            Prefix=f"training_images/{class_name}"
        )
        used_for_centroids_keys = []
        for object_record in records:
            response = object_record.get()
            json_record = response["Body"].read().decode()
            record = json.loads(json_record)
            if record["used_for_centroid"]:
                used_for_centroids_keys.append(object_record.key)

        if len(list(used_for_centroids_keys)) == 0:
            ###########################################
            # Handle robustness check for new classes #
            ###########################################
            print(f"Handling robustness check for new class: {class_name}")

            # Get sorted cos_sim of new images
            sorted_cos_sim = sorted(class_names_cos_sim[class_name])

            # Get 5th percentile of sorted_cos_sim
            threshold = sorted_cos_sim[
                int(len(sorted_cos_sim) * COS_SIM_PERC_THRESHOLD)
            ]

            # Get all validation images and their cos_sim with class_name
            val_cos_sim = []
            validation_img_records = background_bucket.objects.filter(
                Prefix=f"validation_images"
            )
            failed_validation_object_keys = []
            for record in validation_img_records:
                if record.key.split("/")[1] != class_name:
                    record_key = record.key
                    response = record.get()
                    json_record = response["Body"].read().decode()
                    record = json.loads(json_record)
                    val_img_features = np.array(record["img_features"])
                    cos_sim = np.dot(val_img_features, text_features) / (
                        np.linalg.norm(val_img_features) * np.linalg.norm(text_features)
                    )
                    val_cos_sim.append(cos_sim)
                    if cos_sim > threshold:
                        failed_validation_object_keys.append(record_key)

            # Count number of validation images with cos_sim > threshold
            num_val_images = len(val_cos_sim)
            num_val_images_above_threshold = len(failed_validation_object_keys)
            perc_val_images_above_threshold = (
                num_val_images_above_threshold / num_val_images
            )
            print(f"{class_name}: {perc_val_images_above_threshold}")

            # Check validity of class_name
            if perc_val_images_above_threshold > NEW_CLASS_PERC_TOLERANCE:
                class_name_robustness_check[class_name] = False

                # Plot PCA of new images, selected validation images, and text features
                pca_msg1 = f"{perc_val_images_above_threshold*100:.2f}% "
                pca_msg2 = "of validation images have cos_sim >"
                pca_msg3 = f"{threshold:.2f} against class name,"
                pca_msg4 = f"'{class_name}' above new class percentile threshold of"
                pca_msg5 = f"{NEW_CLASS_PERC_TOLERANCE*100:.2f}%."
                pca_msg = f"{pca_msg1} {pca_msg2} {pca_msg3} {pca_msg4} {pca_msg5}"

                thread = Thread(
                    target=plot_pca,
                    args=(
                        batch_training_timestamp_ms,
                        f"system-new-{class_name}",
                        background_bucket,
                    ),
                    kwargs={
                        "class_names": [class_name],
                        "new_class_object_keys": [
                            f"pending_images/{x}" for x in timestamp_ids
                        ],
                        "failed_validation_object_keys": failed_validation_object_keys,
                        "display_message": pca_msg,
                        "batch_training_userID": batch_training_userID,
                    },
                )
                thread.start()
        else:
            ################################################
            # Handle robustness check for existing classes #
            ################################################
            print(f"Handling robustness check for existing class: {class_name}")

            # Get existing training images and their cos_sim with class_name
            train_cos_sim = []
            for record_key in used_for_centroids_keys:
                record = background_bucket.Object(record_key)
                response = record.get()
                json_record = response["Body"].read().decode()
                record = json.loads(json_record)
                train_img_features = np.array(record["img_features"])
                cos_sim = np.dot(train_img_features, text_features) / (
                    np.linalg.norm(train_img_features) * np.linalg.norm(text_features)
                )
                train_cos_sim.append(cos_sim)

            # Get sorted cos_sim of existing images
            sorted_cos_sim = sorted(train_cos_sim)

            # Get 5th percentile of sorted_cos_sim
            threshold = sorted_cos_sim[
                int(len(sorted_cos_sim) * COS_SIM_PERC_THRESHOLD)
            ]

            # Count number of pending images with cos_sim < threshold
            num_pending_images = len(class_names_cos_sim[class_name])
            num_pending_images_below_threshold = len(
                [x for x in class_names_cos_sim[class_name] if x < threshold]
            )
            perc_pending_images_below_threshold = (
                num_pending_images_below_threshold / num_pending_images
            )
            print(f"{class_name}: {perc_pending_images_below_threshold}")

            # Check validity of class_name
            if perc_pending_images_below_threshold > NEW_EXISTING_PERC_TOLERANCE:
                class_name_robustness_check[class_name] = False
                failed_timestamp_ids = [
                    timestamp_id
                    for (timestamp_id, x) in zip(
                        class_name_keys[class_name], class_names_cos_sim[class_name]
                    )
                    if x < threshold
                ]
                passed_timestamp_ids = [
                    timestamp_id
                    for timestamp_id in timestamp_ids
                    if timestamp_id not in failed_timestamp_ids
                ]

                # Plot PCA of new images that pass, new images that fail,
                # existing training images, and text features
                pca_msg1 = f"{perc_pending_images_below_threshold*100:.2f}%"
                pca_msg2 = f"of pending images for {class_name} have cos_sim"
                pca_msg3 = f"< {threshold:.2f} against existing training images,"
                pca_msg4 = "above new existing percentile threshold of"
                pca_msg5 = f"{NEW_EXISTING_PERC_TOLERANCE*100:.2f}%."
                pca_msg = f"{pca_msg1} {pca_msg2} {pca_msg3} {pca_msg4} {pca_msg5}"

                thread = Thread(
                    target=plot_pca,
                    args=(
                        batch_training_timestamp_ms,
                        f"system-existing-{class_name}",
                        background_bucket,
                    ),
                    kwargs={
                        "used_training_images": [class_name],
                        "class_names": [class_name],
                        "passed_new_object_keys": [
                            f"pending_images/{x}" for x in passed_timestamp_ids
                        ],
                        "failed_new_object_keys": [
                            f"pending_images/{x}" for x in failed_timestamp_ids
                        ],
                        "display_message": pca_msg,
                        "batch_training_userID": batch_training_userID,
                    },
                )
                thread.start()

    return class_name_robustness_check


def move_pending_to_training(
    background_bucket,
    timestamp_id: str,
    used_for_centroid: bool,
    target_class_name: str = None,
):
    object_record = background_bucket.Object(f"pending_images/{timestamp_id}")
    response = object_record.get()
    json_record = response["Body"].read().decode()
    record = json.loads(json_record)
    record["used_for_centroid"] = used_for_centroid
    json_record = json.dumps(record, indent=4)
    if target_class_name is None:
        background_bucket.put_object(
            Key=f"training_images/{timestamp_id}", Body=json_record
        )
    else:
        [class_name, timestamp_ms] = timestamp_id.split("/")
        background_bucket.put_object(
            Key=f"training_images/{target_class_name}/{timestamp_ms}", Body=json_record
        )
    object_record.delete()


def update_training(background_bucket, image_path: str, used_for_centroid: bool):
    object_record = background_bucket.Object(image_path)
    response = object_record.get()
    json_record = response["Body"].read().decode()
    record = json.loads(json_record)
    record["used_for_centroid"] = used_for_centroid
    json_record = json.dumps(record, indent=4)
    background_bucket.put_object(Key=image_path, Body=json_record)


def compute_and_upload_centroids(bucket, class_name: str):
    records = bucket.objects.filter(Prefix=f"training_images/{class_name}")
    all_img_features_list = []
    for object_record in records:
        response = object_record.get()
        json_record = response["Body"].read().decode()
        record = json.loads(json_record)
        if record["used_for_centroid"]:
            all_img_features_list.append(record["img_features"])

    if len(all_img_features_list) > 0:
        all_img_features = np.stack(all_img_features_list)
        kmeans = KMeans(
            n_clusters=30,
            init="k-means++",
            n_init=1,
            max_iter=300,
            random_state=42,
        )
        kmeans.fit(all_img_features)

        centroids = kmeans.cluster_centers_
        centroids_record = centroids.tolist()
        json_record = json.dumps(centroids_record, indent=4)
        bucket.put_object(Key=f"centroids/{class_name}", Body=json_record)
        return True
    else:
        print(f"No usuable training images for {class_name}.")
        return False


def fit_and_upload_classifier(centroids_bucket, model_bucket):
    records = centroids_bucket.objects.filter(Prefix=f"centroids/")
    all_centroids_dict = {}
    for object_record in records:
        response = object_record.get()
        json_record = response["Body"].read().decode()
        record = json.loads(json_record)
        all_centroids_dict[object_record.key] = record

    X = []
    y = []

    for key in all_centroids_dict:
        centroid = all_centroids_dict[key]
        class_name = key.split("/")[1]
        X.append(centroid)
        y.append([class_name] * len(centroid))

    X = np.concatenate(X)
    y = np.concatenate(y)

    classifier = KNeighborsClassifier(n_neighbors=30)
    classifier.fit(X, y)

    # Upload model to S3 as a joblib file
    temp_file = "temp/temp_knn.joblib"
    joblib.dump(classifier, temp_file)
    with open(temp_file, "rb") as data:
        model_bucket.put_object(Key="model/knn_centroids_classifier", Body=data)

    # Delete the local file
    os.remove(temp_file)


def batch_training(
    image_processor,
    image_feature_extractor,
    text_tokenizer,
    text_feature_extractor,
    zip_bytes: bytes,
    userID: str,
    timestamp_ms: int,
    background_bucket,
    model_bucket,
):
    # Simple logging
    print(f"Batch training started for {userID}/{timestamp_ms}.")

    # Initialise overall training record to be saved to S3
    json_record = json.dumps({"status": "in progress"}, indent=4)
    background_bucket.put_object(
        Key=f"batch_training/{userID}/{timestamp_ms}", Body=json_record
    )

    # Initialise overall training record
    overall_training = {
        "status": "completed",
        "features_success": {},
        "features_skipped": {},
        "robustness_check_passed": True,
        "centroids_computed": True,
        "classifier_fitted": True,
    }

    # Extract and upload features (images and class name text)
    print(f"Batch Training {userID}/{timestamp_ms}: Extracting and uploading features.")
    class_names = set()
    with ZipFile(BytesIO(zip_bytes)) as zip_ref:
        class_names_text = list(set([f.split("/")[0] for f in zip_ref.namelist()]))

        # Extract and upload features (class name text)
        for class_name in class_names_text:
            extract_and_upload_text_features(
                background_bucket,
                class_name,
                text_tokenizer,
                text_feature_extractor,
                userID,
            )

        # Extract and upload features (images)
        timestamp_ids = []
        for filename in zip_ref.namelist():
            if any(
                filename.lower().endswith(prefix)
                for prefix in [".png", ".jpg", ".jpeg"]
            ):
                try:
                    with zip_ref.open(filename) as f:
                        image_bytes = f.read()
                except Exception as err:
                    print(err)
                    image_bytes = None
                    overall_training["status"] = "error_encountered"
                    overall_training["features_skipped"][
                        filename
                    ] = "File could not be read."

                try:
                    class_name = filename.split("/")[0]
                    class_names.add(class_name)
                except Exception as err:
                    print(err)
                    class_name = None
                    message = "Class (i.e. folder name) could not be read."
                    overall_training["status"] = "error_encountered"
                    overall_training["features_skipped"][filename] = message

                if image_bytes is not None:
                    try:
                        timestamp_id = extract_and_upload_img_features(
                            background_bucket,
                            class_name,
                            image_processor,
                            image_feature_extractor,
                            image_bytes,
                            userID,
                            filename,
                            folder_name="pending_images",
                            robustness_check=True,
                        )
                        timestamp_ids.append(timestamp_id)
                        overall_training["features_success"][
                            filename
                        ] = "Feature extracted."
                    except Exception as err:
                        print(err)
                        overall_training["status"] = "error_encountered"
                        overall_training["features_skipped"][
                            filename
                        ] = "Feature extraction failed."
            else:
                overall_training["status"] = "error_encountered"
                overall_training["features_skipped"][
                    filename
                ] = "File extension not supported."

    # Robustness check: move images to training images if robustness check passed
    class_name_robustness_check = do_robustness_check(
        timestamp_ms, userID, timestamp_ids, background_bucket
    )
    for class_name in class_name_robustness_check:
        if class_name_robustness_check[class_name]:
            print(f"Moving images for {class_name} to training_images.")
            filtered_timestamp_ids = [
                timestamp_id
                for timestamp_id in timestamp_ids
                if timestamp_id.split("/")[0] == class_name
            ]
            for timestamp_id in filtered_timestamp_ids:
                move_pending_to_training(background_bucket, timestamp_id, True)
        else:
            overall_training["status"] = "pending manual review"
            overall_training["robustness_check_passed"] = False

    # Upload centroids
    cc_msg1 = "Computing and uploading centroids for classes"
    cc_msg2 = "that passed robustness checks."
    print(f"Batch Training {userID}/{timestamp_ms}: {cc_msg1} {cc_msg2}")
    try:
        for class_name in class_name_robustness_check:
            if class_name_robustness_check[class_name]:
                print(f"Computing and uploading centroids for {class_name}.")
                compute_and_upload_centroids(background_bucket, class_name)

        if not any(class_name_robustness_check.values()):
            overall_training["centroids_computed"] = False
    except Exception as err:
        print(err)
        overall_training["status"] = "error_encountered"
        overall_training["centroids_computed"] = False

    # Fit and upload classifier
    try:
        if any(class_name_robustness_check.values()):
            print(
                f"Batch Training {userID}/{timestamp_ms}: Fitting and uploading classifier."
            )
            fit_and_upload_classifier(background_bucket, model_bucket)
        else:
            overall_training["classifier_fitted"] = False
    except Exception as err:
        print(err)
        overall_training["status"] = "error_encountered"
        overall_training["classifier_fitted"] = False

    # Save overall_training to S3
    json_record = json.dumps(overall_training, indent=4)
    background_bucket.put_object(
        Key=f"batch_training/{userID}/{timestamp_ms}", Body=json_record
    )

    # Simple logging
    print(f"Batch training completed for {userID}/{timestamp_ms}.")


def validation_features_extraction(
    image_processor,
    image_feature_extractor,
    text_tokenizer,
    text_feature_extractor,
    zip_bytes: bytes,
    userID: str,
    timestamp_ms: int,
    background_bucket,
):
    # Simple logging
    print(f"Validation features extraction started for {userID}/{timestamp_ms}.")

    # Initialise overall extraction record to be saved to S3
    json_record = json.dumps({"status": "in progress"}, indent=4)
    background_bucket.put_object(
        Key=f"validation_features_extraction/{userID}/{timestamp_ms}", Body=json_record
    )

    # Initialise overall extraction record
    overall_extraction = {
        "status": "completed",
        "features_success": {},
        "features_skipped": {},
    }

    # Extract and upload features (images and class name text)
    print(f"{userID}/{timestamp_ms}: Extracting and uploading features.")
    class_names = set()
    with ZipFile(BytesIO(zip_bytes)) as zip_ref:
        class_names_text = list(set([f.split("/")[0] for f in zip_ref.namelist()]))

        # Extract and upload features (class name text)
        for class_name in class_names_text:
            extract_and_upload_text_features(
                background_bucket,
                class_name,
                text_tokenizer,
                text_feature_extractor,
                userID,
            )

        # Extract and upload features (images)
        for filename in zip_ref.namelist():
            if any(
                filename.lower().endswith(prefix)
                for prefix in [".png", ".jpg", ".jpeg"]
            ):
                try:
                    with zip_ref.open(filename) as f:
                        image_bytes = f.read()
                except Exception as err:
                    print(err)
                    image_bytes = None
                    overall_extraction["status"] = "error_encountered"
                    overall_extraction["features_skipped"][
                        filename
                    ] = "File could not be read."

                try:
                    class_name = filename.split("/")[0]
                    class_names.add(class_name)
                except Exception as err:
                    print(err)
                    class_name = None
                    message = "Class (i.e. folder name) could not be read."
                    overall_extraction["status"] = "error_encountered"
                    overall_extraction["features_skipped"][filename] = message

                if image_bytes is not None:
                    try:
                        extract_and_upload_img_features(
                            background_bucket,
                            class_name,
                            image_processor,
                            image_feature_extractor,
                            image_bytes,
                            userID,
                            filename,
                            folder_name="validation_images",
                        )
                        overall_extraction["features_success"][
                            filename
                        ] = "Feature extracted."
                    except Exception as err:
                        print(err)
                        overall_extraction["status"] = "error_encountered"
                        overall_extraction["features_skipped"][
                            filename
                        ] = "Feature extraction failed."
            else:
                overall_extraction["status"] = "error_encountered"
                overall_extraction["features_skipped"][
                    filename
                ] = "File extension not supported."

    # Save overall_extraction to S3
    json_record = json.dumps(overall_extraction, indent=4)
    background_bucket.put_object(
        Key=f"validation_features_extraction/{userID}/{timestamp_ms}", Body=json_record
    )

    # Simple logging
    print(f"features extraction completed for {userID}/{timestamp_ms}.")


def update_images_and_batch_training(
    update_to_used: list,
    update_to_unused: list,
    userID: str,
    timestamp_ms: int,
    background_bucket,
    model_bucket,
    update_existing_job: bool,
    review_userID: str = None,
    review_timestamp_ms: int = None,
    target_class_name: str = None,
):
    # Initialise overall training record to be saved to S3
    if update_existing_job:
        print(f"Post manual review update started for {userID}/{timestamp_ms}.")
        # Check review userID and review timestamp
        if review_userID is None and review_timestamp_ms is None:
            raise ValueError(
                "review_userID and review_timestamp_ms are needed to update existing job."
            )
        elif review_userID is None:
            raise ValueError("review_userID is needed to update existing job.")
        elif review_timestamp_ms is None:
            raise ValueError("review_timestamp_ms is needed to update existing job.")

        # Get existing job details
        object_record = background_bucket.Object(
            f"batch_training/{userID}/{timestamp_ms}"
        )
        response = object_record.get()
        json_record = response["Body"].read().decode()
        overall_training = json.loads(json_record)

        # Update info for display
        overall_training["status"] = "post manual review update"
        overall_training["manual_review_supervisor"] = review_userID
        overall_training["manual_review_date"] = get_timestamp_ms()
        json_record = json.dumps(overall_training, indent=4)
        background_bucket.put_object(
            Key=f"batch_training/{userID}/{timestamp_ms}", Body=json_record
        )

        # Initialise overall training record for updates
        overall_training["status"] = "completed"
        overall_training["centroids_computed"] = True
        overall_training["classifier_fitted"] = True
    else:
        print(f"Adhoc batch updating and training started for {userID}/{timestamp_ms}.")
        json_record = json.dumps({"status": "in progress"}, indent=4)
        background_bucket.put_object(
            Key=f"batch_training/{userID}/{timestamp_ms}", Body=json_record
        )

        # Initialise overall training record for updates
        overall_training = {
            "status": "completed",
            "centroids_computed": True,
            "classifier_fitted": True,
        }

    # Initialise classes to be updated
    classes_to_update = set()

    # Update images to used list
    failed_used_list = []
    for image_path in update_to_used:
        try:
            [image_folder, class_name, image_timestamp_ms] = image_path.split("/")
            if image_folder == "training_images":
                update_training(background_bucket, image_path, True)
            else:
                timestamp_id = f"{class_name}/{image_timestamp_ms}"
                move_pending_to_training(
                    background_bucket, timestamp_id, True, target_class_name
                )
            if target_class_name is None:
                classes_to_update.add(class_name)
            else:
                classes_to_update.add(target_class_name)
        except Exception as err:
            print(err)
            overall_training["status"] = "error_encountered"
            failed_used_list.append(image_path)
    if len(failed_used_list) > 0:
        overall_training["failed_used_list"] = failed_used_list

    # Update images to unused list
    failed_unused_list = []
    for image_path in update_to_unused:
        try:
            [image_folder, class_name, image_timestamp_ms] = image_path.split("/")
            if image_folder == "training_images":
                update_training(background_bucket, image_path, False)
            else:
                timestamp_id = f"{class_name}/{image_timestamp_ms}"
                move_pending_to_training(
                    background_bucket, timestamp_id, False, target_class_name
                )
            if target_class_name is None:
                classes_to_update.add(class_name)
            else:
                classes_to_update.add(target_class_name)
        except Exception as err:
            print(err)
            overall_training["status"] = "error_encountered"
            failed_unused_list.append(image_path)
    if len(failed_unused_list) > 0:
        overall_training["failed_unused_list"] = failed_unused_list

    # Compute centroids for above updated classes
    print(f"Batch Training {userID}/{timestamp_ms}: Computing and uploading centroids.")
    overall_training["retrained_classes"] = list(classes_to_update)
    any_centroids_compute_status = False
    for class_name in classes_to_update:
        try:
            if compute_and_upload_centroids(background_bucket, class_name):
                any_centroids_compute_status = True
        except Exception as err:
            print(err)
            overall_training["status"] = "error_encountered"
            overall_training["centroids_computed"] = False
    if overall_training["status"] != "error_encountered":
        overall_training["centroids_computed"] = any_centroids_compute_status

    # Fit and upload classifier
    if any_centroids_compute_status:
        print(
            f"Batch Training {userID}/{timestamp_ms}: Fitting and uploading classifier."
        )
        try:
            fit_and_upload_classifier(background_bucket, model_bucket)
        except Exception as err:
            print(err)
            overall_training["status"] = "error_encountered"
            overall_training["classifier_fitted"] = False
    if overall_training["status"] != "error_encountered":
        overall_training["classifier_fitted"] = any_centroids_compute_status

    # Save overall_training to S3
    json_record = json.dumps(overall_training, indent=4)
    background_bucket.put_object(
        Key=f"batch_training/{userID}/{timestamp_ms}", Body=json_record
    )

    # Simple logging
    print(f"Batch updating and training completed for {userID}/{timestamp_ms}.")


def get_batch_training_record_from_bucket(client, bucket, user, timestamp):
    try:
        object_key = f"batch_training/{user}/{timestamp}"
        file_object = BytesIO()
        client.download_fileobj(bucket.name, object_key, file_object)
        file_object.seek(0)
        return file_object
    except ClientError:
        return None


def get_validation_features_extraction_record_from_bucket(
    client, bucket, user, timestamp
):
    try:
        object_key = f"validation_features_extraction/{user}/{timestamp}"
        file_object = BytesIO()
        client.download_fileobj(bucket.name, object_key, file_object)
        file_object.seek(0)
        return file_object
    except ClientError:
        return None


def get_text_features_record_from_bucket(client, bucket, class_name):
    try:
        object_key = f"text_features/{class_name}"
        file_object = BytesIO()
        client.download_fileobj(bucket.name, object_key, file_object)
        file_object.seek(0)
        return file_object
    except ClientError:
        return None


def plot_pca(
    timestamp_ms: int,
    userID: str,
    bucket,
    centroids: list = [],
    used_training_images: list = [],
    unused_training_images: list = [],
    validation_images: list = [],
    class_names: list = [],
    new_class_object_keys: list = [],  # new class robustness check
    failed_validation_object_keys: list = [],  # new class robustness check
    passed_new_object_keys: list = [],  # existing class robustness check
    failed_new_object_keys: list = [],  # existing class robustness check
    display_message: str = "",
    batch_training_userID: str = None,
):
    # Simple logging
    print(f"Visualisation started for {userID}/{timestamp_ms}.")

    # Initialise overall training record to be saved to S3
    pca_job = {
        "status": "in progress",
        "centroid_dict": {k: {} for k in centroids},
        "used_training_image_dict": {k: {} for k in used_training_images},
        "unused_training_image_dict": {k: {} for k in unused_training_images},
        "validation_image_dict": {k: {} for k in validation_images},
        "class_name_dict": {k: {} for k in class_names},
        "new_class_image_dict": {},
        "failed_validation_image_dict": {},
        "passed_new_image_dict": {},
        "failed_new_image_dict": {},
        "display_message": display_message,
        "batch_training_userID": batch_training_userID,
    }
    json_record = json.dumps(pca_job, indent=4)
    bucket.put_object(Key=f"pca/{userID}/{timestamp_ms}", Body=json_record)

    try:
        # All Tensors and Labels
        all_tensors = []
        all_labels = []

        # Get tensors from centroids bucket
        centroid_dict = {}
        for centroid in centroids:
            object_record = bucket.Object(f"centroids/{centroid}")
            try:
                response = object_record.get()
                json_record = response["Body"].read().decode()
                record = json.loads(json_record)
                all_tensors.extend(record)
                labels = [(centroid, "centroid")] * len(record)
                all_labels.extend(labels)
                centroid_dict[centroid] = {}
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    print(f"The object with key 'centroids/{centroid}' does not exist.")
                else:
                    # Handle other errors
                    print(f"Error: {e}")
        all_tensors = np.array(all_tensors)  # dim will be 1 if list is empty

        # Get tensors from training images bucket
        used_training_image_dict = {}
        unused_training_image_dict = {}
        training_images = set(used_training_images) | set(unused_training_images)
        for training_image in training_images:
            training_image_tensors = []
            # used_training_image_labels = []
            # unused_training_image_labels = []

            used_meta = []
            unused_meta = []
            object_records = bucket.objects.filter(
                Prefix=f"training_images/{training_image}"
            )
            for object_record in object_records:
                response = object_record.get()
                json_record = response["Body"].read().decode()
                record = json.loads(json_record)
                training_image_tensors.append(record["img_features"])
                del record["img_features"]
                del record["img64"]
                record["timestamp_raw_id"] = object_record.key.split("/")[-1]

                if record["used_for_centroid"]:
                    if training_image in used_training_images:
                        # used_training_image_labels.append(training_image)
                        used_meta.append(record)
                        all_labels.append((training_image, "used"))
                else:
                    if training_image in unused_training_images:
                        # unused_training_image_labels.append(training_image)
                        unused_meta.append(record)
                        all_labels.append((training_image, "unused"))

            if len(used_meta) > 0:
                used_training_image_dict[training_image] = {}
                used_training_image_dict[training_image]["meta"] = used_meta

            if len(unused_meta) > 0:
                unused_training_image_dict[training_image] = {}
                unused_training_image_dict[training_image]["meta"] = unused_meta

            if len(list(object_records)) > 0:
                training_image_tensors = np.array(training_image_tensors)
                if len(all_tensors.shape) == 1:
                    all_tensors = training_image_tensors
                else:
                    all_tensors = np.concatenate([all_tensors, training_image_tensors])

        # Get tensors from validation images bucket
        validation_image_dict = {}
        for validation_image in validation_images:
            validation_image_tensors = []
            # validation_image_labels = []
            validation_meta = []

            object_records = bucket.objects.filter(
                Prefix=f"validation_images/{validation_image}"
            )
            for object_record in object_records:
                response = object_record.get()
                json_record = response["Body"].read().decode()
                record = json.loads(json_record)
                validation_image_tensors.append(record["img_features"])

                del record["img_features"]
                del record["img64"]
                record["timestamp_raw_id"] = object_record.key.split("/")[-1]
                validation_meta.append(record)

                # validation_image_labels.append(validation_image)
                all_labels.append((validation_image, "validation"))

            if len(list(object_records)) > 0:
                validation_image_tensors = np.array(validation_image_tensors)
                if len(all_tensors.shape) == 1:
                    all_tensors = validation_image_tensors
                else:
                    all_tensors = np.concatenate(
                        [all_tensors, validation_image_tensors]
                    )

                validation_image_dict[validation_image] = {}
                validation_image_dict[validation_image]["meta"] = validation_meta

        # Get tensors from class name text features bucket
        class_name_dict = {}
        for class_name in class_names:
            object_record = bucket.Object(f"text_features/{class_name}")
            try:
                response = object_record.get()
                json_record = response["Body"].read().decode()
                record = [json.loads(json_record)]
                if len(all_tensors.shape) == 1:
                    all_tensors = np.array(record)
                else:
                    all_tensors = np.concatenate([all_tensors, np.array(record)])
                all_labels.append((class_name, "class_name"))
                class_name_dict[class_name] = {}
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    print(
                        f"The object with key 'text_features/{class_name}' does not exist."
                    )
                else:
                    # Handle other errors
                    print(f"Error: {e}")

        # Get tensors from object_keys robustness check
        category_to_object_keys = {
            "new_class": new_class_object_keys,  # e.g. chickadee, new class
            "failed_validation": failed_validation_object_keys,  # <class_name>, failed validation
            "passed_existing": passed_new_object_keys,  # e.g. chickadee, passed existing
            "failed_existing": failed_new_object_keys,  # e.g. chickadee, failed existing
        }

        new_class_dict = {}
        failed_validation_dict = {}
        passed_existing_dict = {}
        failed_existing_dict = {}

        for category in category_to_object_keys:
            image_tensors = []
            image_meta = []

            for object_key in category_to_object_keys[category]:
                object_record = bucket.Object(object_key)
                response = object_record.get()
                json_record = response["Body"].read().decode()
                record = json.loads(json_record)
                image_tensors.append(record["img_features"])

                [class_name, timestamp_raw_id] = object_key.split("/")[1:]
                del record["img_features"]
                del record["img64"]
                record["timestamp_raw_id"] = timestamp_raw_id

                if category == "failed_validation":
                    dict_key = "failed validation"
                    failed_validation_dict[dict_key] = failed_validation_dict.get(
                        dict_key, {}
                    )
                    failed_validation_dict[dict_key]["meta"] = failed_validation_dict[
                        dict_key
                    ].get("meta", [])
                    failed_validation_dict[dict_key]["meta"].append(record)
                    all_labels.append((dict_key, category))
                else:
                    if category == "new_class":
                        dict_to_update = new_class_dict
                        # default false. supervisor will select which ones to use
                        record["used_for_centroid"] = False
                    elif category == "passed_existing":
                        dict_to_update = passed_existing_dict
                        # default true. supervisor will select which ones to drop (i.e. unused)
                        record["used_for_centroid"] = True
                    elif category == "failed_existing":
                        dict_to_update = failed_existing_dict
                        # default false. supervisor will select which ones to use
                        record["used_for_centroid"] = False

                    dict_to_update[class_name] = dict_to_update.get(class_name, {})
                    dict_to_update[class_name]["meta"] = dict_to_update[class_name].get(
                        "meta", []
                    )
                    record["pending"] = True
                    dict_to_update[class_name]["meta"].append(record)
                    all_labels.append((class_name, category))

            if len(category_to_object_keys[category]) > 0:
                image_tensors = np.array(image_tensors)
                if len(all_tensors.shape) == 1:
                    all_tensors = image_tensors
                else:
                    all_tensors = np.concatenate([all_tensors, image_tensors])

        # Compute PCA from above tensors
        pca = PCA(n_components=2, random_state=42)
        pca_points = pca.fit_transform(all_tensors)
        for dict_key, category in set(all_labels):
            idx = [i for i, x in enumerate(all_labels) if x == (dict_key, category)]
            if category == "centroid":
                dict_to_update = centroid_dict
            elif category == "used":
                dict_to_update = used_training_image_dict
            elif category == "unused":
                dict_to_update = unused_training_image_dict
            elif category == "validation":
                dict_to_update = validation_image_dict
            elif category == "class_name":
                dict_to_update = class_name_dict
            elif category == "new_class":
                dict_to_update = new_class_dict
            elif category == "failed_validation":
                dict_to_update = failed_validation_dict
            elif category == "passed_existing":
                dict_to_update = passed_existing_dict
            elif category == "failed_existing":
                dict_to_update = failed_existing_dict
            else:
                raise ValueError("Invalid category.")

            dict_to_update[dict_key]["pca_x"] = pca_points[idx, 0].tolist()
            dict_to_update[dict_key]["pca_y"] = pca_points[idx, 1].tolist()

        # Upload PCA job to S3 as a json file
        pca_job = {
            "status": "completed",
            "centroid_dict": centroid_dict,
            "used_training_image_dict": used_training_image_dict,
            "unused_training_image_dict": unused_training_image_dict,
            "validation_image_dict": validation_image_dict,
            "class_name_dict": class_name_dict,
            "new_class_image_dict": new_class_dict,
            "failed_validation_image_dict": failed_validation_dict,
            "passed_new_image_dict": passed_existing_dict,
            "failed_new_image_dict": failed_existing_dict,
            "display_message": display_message,
            "batch_training_userID": batch_training_userID,
        }
        json_record = json.dumps(pca_job, indent=4)
        bucket.put_object(Key=f"pca/{userID}/{timestamp_ms}", Body=json_record)
    except Exception as err:
        print(err)
        pca_job["status"] = f"error_encountered: {err}"
        json_record = json.dumps(pca_job, indent=4)
        bucket.put_object(Key=f"pca/{userID}/{timestamp_ms}", Body=json_record)


def get_visualisation_record_from_bucket(client, bucket, user, timestamp):
    try:
        object_key = f"pca/{user}/{timestamp}"
        file_object = BytesIO()
        client.download_fileobj(bucket.name, object_key, file_object)
        file_object.seek(0)
        return file_object
    except ClientError:
        return None


def update_visualisation_record_about_manual(bucket, user, timestamp):
    try:
        object_key = f"pca/{user}/{timestamp}"
        object_record = bucket.Object(object_key)
        response = object_record.get()
        json_record = response["Body"].read().decode()
        overall_training = json.loads(json_record)
        overall_training["status"] = "completed with manual review triggered"
        json_record = json.dumps(overall_training, indent=4)
        bucket.put_object(Key=f"pca/{user}/{timestamp}", Body=json_record)
        return True
    except Exception as e:
        print(e)
        return False


def get_image(bucket, folder, class_name, timestamp_raw_id):
    object_record = bucket.Object(f"{folder}/{class_name}/{timestamp_raw_id}")
    response = object_record.get()
    json_record = response["Body"].read().decode()
    record = json.loads(json_record)
    img_b64 = record["img64"]
    if img_b64 is None:
        return None
    img_bytes = base64.b64decode(img_b64)
    img_io = BytesIO(img_bytes)
    return img_io
