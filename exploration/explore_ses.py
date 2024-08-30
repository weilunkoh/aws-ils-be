import json
import sys

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

ses_client = boto3.client("ses")
source_email = "<insert source email>"
dest_email = "<insert destination email>"

welcome_template_name = "welcome_email_change_pw"
reset_pw_template_name = "reset_pw_email"


def verify_identity():
    try:
        response = ses_client.get_identity_verification_attributes(
            Identities=[source_email]
        )
        print(response)
        status = response["VerificationAttributes"].get(
            source_email, {"VerificationStatus": "NotFound"}
        )["VerificationStatus"]
        print(f"Got status of {source_email}.")

        verified = status == "Success"

        if verified:
            print(f"{source_email} is verified.")
        else:
            print(f"{source_email} is not verified.")

        return True

    except ClientError:
        print(f"Couldn't get status for {source_email}.")
        raise


def create_welcome_template():
    subject_part = "Welcome to SMU ILS on AWS!"
    text_part = "\n".join(
        [
            # "Navigate to ILS at either of the following URLs:",
            # " - https://smuils1.jklmgroup.com",
            # " - https://smuils2.jklmgroup.com",
            "Navigate to ILS: https://smuils1.jklmgroup.com.",
            "",
            "After that, use the following credentials to login:",
            " - Username: {{username}}",
            " - Password: {{password}}",
            "",
            " ".join(
                [
                    "Upon logging in,",
                    "follow the on screen instructions to",
                    "change your password for subsequent logins.",
                ]
            ),
        ]
    )
    html_part = "".join(
        [
            # "<p>Navigate to ILS at either of the following URLs:</p>",
            # "<ul><li>https://smuils1.jklmgroup.com</li>",
            # "<li>https://smuils2.jklmgroup.com</li></ul>",
            "<p>Navigate to ILS: https://smuils1.jklmgroup.com.</p>",
            "<p>After that, use the following credentials to login:</p>",
            "<ul><li>Username: {{username}}</li>",
            "<li>Password: {{password}}</li></ul>",
            " ".join(
                [
                    "<p>Upon logging in,",
                    "follow the on screen instructions to",
                    "change your password for subsequent logins.</p>",
                ]
            ),
        ]
    )

    email_template = {
        "TemplateName": welcome_template_name,
        "SubjectPart": subject_part,
        "TextPart": text_part,
        "HtmlPart": html_part,
    }
    try:
        ses_client.create_template(Template=email_template)
        print(f"Created template {welcome_template_name}")
    except ClientError:
        print(f"Couldn't create template {welcome_template_name}")
        raise


def create_reset_pw_template():
    subject_part = "SMU ILS Reset Password Request"
    text_part = "\n".join(
        [
            "Your request to reset your password has been processed.",
            # "Navigate to ILS at either of the following URLs:",
            # " - https://smuils1.jklmgroup.com",
            # " - https://smuils2.jklmgroup.com",
            "Navigate to ILS: https://smuils1.jklmgroup.com.",
            "",
            "After that, use the new temporary password below to login:",
            " - Username: {{username}}",
            " - Password: {{password}}",
            "",
            " ".join(
                [
                    "Upon logging in,",
                    "you will be required to",
                    "change your password for subsequent logins.",
                ]
            ),
        ]
    )
    html_part = "".join(
        [
            "<p>Your request to reset your password has been processed.</p>",
            # "<p>Navigate to ILS at either of the following URLs:</p>",
            # "<ul><li>https://smuils1.jklmgroup.com</li>",
            # "<li>https://smuils2.jklmgroup.com</li></ul>",
            "<p>Navigate to ILS: https://smuils1.jklmgroup.com.</p>",
            "<p>After that, use the new temporary password below to login:</p>",
            "<ul><li>Username: {{username}}</li>",
            "<li>Password: {{password}}</li></ul>",
            " ".join(
                [
                    "<p>Upon logging in,",
                    "you will be required to",
                    "change your password for subsequent logins.</p>",
                ]
            ),
        ]
    )

    email_template = {
        "TemplateName": reset_pw_template_name,
        "SubjectPart": subject_part,
        "TextPart": text_part,
        "HtmlPart": html_part,
    }
    try:
        ses_client.create_template(Template=email_template)
        print(f"Created template {reset_pw_template_name}")
    except ClientError:
        print(f"Couldn't create template {reset_pw_template_name}")
        raise


def send_email(template_name):
    destination = {"ToAddresses": [dest_email]}
    username = source_email
    password = "<insert password>"
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


def delete_welcome_template():
    try:
        ses_client.delete_template(TemplateName=welcome_template_name)
        print(f"Deleted template {welcome_template_name}")
    except ClientError:
        print(f"Couldn't delete template {welcome_template_name}")
        raise


def delete_reset_pw_template():
    try:
        ses_client.delete_template(TemplateName=reset_pw_template_name)
        print(f"Deleted template {reset_pw_template_name}")
    except ClientError:
        print(f"Couldn't delete template {reset_pw_template_name}")
        raise


if __name__ == "__main__":
    if int(sys.argv[1]) == 0:
        verify_identity()
    elif int(sys.argv[1]) == 1:
        create_welcome_template()
    elif int(sys.argv[1]) == 2:
        send_email("welcome_email_change_pw")
    elif int(sys.argv[1]) == 3:
        delete_welcome_template()
    elif int(sys.argv[1]) == 4:
        create_reset_pw_template()
    elif int(sys.argv[1]) == 5:
        send_email("reset_pw_email")
    elif int(sys.argv[1]) == 6:
        delete_reset_pw_template()
