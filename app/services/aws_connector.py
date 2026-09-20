import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError


CONNECTION_FILE = Path(__file__).resolve().parents[2] / "aws_connection.json"


def _save_state(state: dict):
    CONNECTION_FILE.write_text(
        __import__("json").dumps(state, indent=2),
        encoding="utf-8",
    )


def get_connection_state():
    if not CONNECTION_FILE.exists():
        return {"connected": False}

    try:
        return __import__("json").loads(
            CONNECTION_FILE.read_text(encoding="utf-8")
        )
    except Exception:
        return {"connected": False}


def _session(
    access_key_id: str,
    secret_access_key: str,
    region: str,
):
    return boto3.Session(
        aws_access_key_id=access_key_id.strip(),
        aws_secret_access_key=secret_access_key.strip(),
        region_name=region,
    )


def test_connection(
    account_id: str,
    access_key_id: str,
    secret_access_key: str,
    region: str,
):
    """
    Test AWS credentials and CloudSense read-only capabilities.

    Required for overall AWS connection:
        - Identity
        - Billing / Cost Explorer
        - EC2

    Optional:
        - CloudWatch

    CloudWatch failure does NOT make the overall
    AWS connection fail.
    """

    # ---------------------------------------------------------
    # 1. Validate input
    # ---------------------------------------------------------
    if not access_key_id or not access_key_id.strip():
        return {
            "success": False,
            "message": "AWS Access Key ID and Secret Access Key are required.",
        }

    if not secret_access_key or not secret_access_key.strip():
        return {
            "success": False,
            "message": "AWS Access Key ID and Secret Access Key are required.",
        }

    if not region or not region.strip():
        return {
            "success": False,
            "message": "AWS Region is required.",
        }

    region = region.strip()
    account_id = account_id.strip() if account_id else ""

    try:
        # ---------------------------------------------------------
        # 2. Create AWS session
        # ---------------------------------------------------------
        session = _session(
            access_key_id,
            secret_access_key,
            region,
        )

        # ---------------------------------------------------------
        # 3. Verify AWS identity
        # ---------------------------------------------------------
        sts = session.client(
            "sts",
            region_name=region,
        )

        identity = sts.get_caller_identity()

        actual_account = identity.get("Account")

        # ---------------------------------------------------------
        # 4. Validate AWS Account ID
        # ---------------------------------------------------------
        if account_id and actual_account != account_id:
            return {
                "success": False,
                "message": (
                    f"AWS Account ID mismatch. "
                    f"Credentials belong to account {actual_account}, "
                    f"not {account_id}."
                ),
                "account_id": actual_account,
            }

        # ---------------------------------------------------------
        # 5. Initialize checks
        # ---------------------------------------------------------
        checks = {
            "identity": True,
            "billing": False,
            "cloudwatch": False,
            "ec2": False,
        }

        errors = {}

        # ---------------------------------------------------------
        # 6. Billing / Cost Explorer check
        #
        # Cost Explorer uses us-east-1 endpoint.
        # ---------------------------------------------------------
        try:
            ce = session.client(
                "ce",
                region_name="us-east-1",
            )

            today = datetime.now(timezone.utc)

            start_date = (
                today - timedelta(days=2)
            ).strftime("%Y-%m-%d")

            end_date = today.strftime("%Y-%m-%d")

            ce.get_cost_and_usage(
                TimePeriod={
                    "Start": start_date,
                    "End": end_date,
                },
                Granularity="DAILY",
                Metrics=[
                    "UnblendedCost"
                ],
            )

            checks["billing"] = True

        except Exception as exc:
            errors["billing"] = str(exc)

        # ---------------------------------------------------------
        # 7. CloudWatch check
        #
        # CloudWatch is OPTIONAL.
        #
        # IMPORTANT:
        # list_metrics() does NOT support MaxResults.
        # ---------------------------------------------------------
        try:
            cloudwatch = session.client(
                "cloudwatch",
                region_name=region,
            )

            cloudwatch.list_metrics(
                RecentlyActive="PT3H"
            )

            checks["cloudwatch"] = True

        except Exception as exc:
            checks["cloudwatch"] = False
            errors["cloudwatch"] = str(exc)

        # ---------------------------------------------------------
        # 8. EC2 resource check
        # ---------------------------------------------------------
        try:
            ec2 = session.client(
                "ec2",
                region_name=region,
            )

            ec2.describe_instances(
                MaxResults=5
            )

            checks["ec2"] = True

        except Exception as exc:
            checks["ec2"] = False
            errors["ec2"] = str(exc)

        # ---------------------------------------------------------
        # 9. Determine overall connection status
        #
        # Identity + Billing + EC2 are required.
        # CloudWatch is optional.
        # ---------------------------------------------------------
        connected = (
            checks.get("identity", False)
            and checks.get("billing", False)
            and checks.get("ec2", False)
        )

        # ---------------------------------------------------------
        # 10. Save connection state
        # ---------------------------------------------------------
        state = {
            "connected": connected,
            "account_id": actual_account,
            "region": region,
            "last_tested_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "checks": checks,
        }

        _save_state(state)

        # ---------------------------------------------------------
        # 11. Create user-friendly message
        # ---------------------------------------------------------
        if connected:
            if checks["cloudwatch"]:
                message = "AWS connected successfully."
            else:
                message = (
                    "AWS connected successfully. "
                    "CloudWatch is unavailable, so "
                    "metric-based analysis may be limited."
                )
        else:
            failed_required_checks = []

            if not checks["identity"]:
                failed_required_checks.append("Identity")

            if not checks["billing"]:
                failed_required_checks.append("Billing")

            if not checks["ec2"]:
                failed_required_checks.append("EC2")

            if failed_required_checks:
                message = (
                    "AWS connection failed. "
                    "Required checks failed: "
                    + ", ".join(failed_required_checks)
                    + "."
                )
            else:
                message = "AWS connection failed."

        # ---------------------------------------------------------
        # 12. Return result
        # ---------------------------------------------------------
        return {
            "success": connected,
            "message": message,
            **state,
            "errors": errors,
        }

    # ---------------------------------------------------------
    # AWS-specific errors
    # ---------------------------------------------------------
    except (
        ClientError,
        BotoCoreError,
        NoCredentialsError,
    ) as exc:

        return {
            "success": False,
            "message": f"AWS connection failed: {exc}",
        }

    # ---------------------------------------------------------
    # Unexpected errors
    # ---------------------------------------------------------
    except Exception as exc:

        return {
            "success": False,
            "message": f"AWS connection failed: {exc}",
        }