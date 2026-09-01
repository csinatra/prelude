"""Minimal Lambda Cloud API helper — terminate the box when a batch is done.

Only the terminate operation is needed on-box: the batch driver drains the
run queue, then (with --terminate-on-done) destroys the instance so it never
idles after the last run. Lambda on-demand instances have no "stopped"
state — terminating is the only thing that stops billing (a guest-OS
shutdown does not).

Auth via LAMBDA_API_KEY (kept off the dev machine; set on the box only).
The instance id comes from the launch response / dashboard — Lambda has no
metadata service to self-discover it, so it is passed in explicitly.
"""

import json
import os
import urllib.error
import urllib.request

# cloud.lambda.ai is the primary server; cloud.lambdalabs.com still resolves but
# is marked deprecated in Lambda's OpenAPI spec.
TERMINATE_URL = "https://cloud.lambda.ai/api/v1/instance-operations/terminate"


def terminate_instance(*, instance_id: str, api_key: str | None = None) -> dict:
    """POST a terminate for one instance id. Returns the parsed API response."""
    key = api_key or os.environ["LAMBDA_API_KEY"]
    body = json.dumps({"instance_ids": [instance_id]}).encode()
    request = urllib.request.Request(url=TERMINATE_URL, data=body, method="POST")
    request.add_header("Authorization", f"Bearer {key}")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request) as response:  # noqa: S310 (fixed https URL)
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        # urlopen raises before anyone reads the body, so the most consequential
        # call in the harness was failing with the least information: a bare
        # "HTTP Error 403: Forbidden" at the end of a drain, with no way to tell
        # an inactive account from a wrong key or a non-terminable instance.
        # Lambda returns a JSON error carrying code/message/suggestion plus a
        # request_id that support asks for, so surface all of it.
        detail = error.read().decode(errors="replace")
        try:
            reported = json.loads(detail).get("error", {})
            detail = " | ".join(
                str(reported[field])
                for field in ("code", "message", "suggestion", "request_id")
                if reported.get(field)
            )
        except (json.JSONDecodeError, AttributeError):
            pass  # not JSON; the raw body is still better than nothing
        raise RuntimeError(
            f"terminate failed for {instance_id}: HTTP {error.code} — {detail}"
        ) from error
