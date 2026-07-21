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
import urllib.request

TERMINATE_URL = "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate"


def terminate_instance(*, instance_id: str, api_key: str | None = None) -> dict:
    """POST a terminate for one instance id. Returns the parsed API response."""
    key = api_key or os.environ["LAMBDA_API_KEY"]
    body = json.dumps({"instance_ids": [instance_id]}).encode()
    request = urllib.request.Request(url=TERMINATE_URL, data=body, method="POST")
    request.add_header("Authorization", f"Bearer {key}")
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request) as response:  # noqa: S310 (fixed https URL)
        return json.loads(response.read())
