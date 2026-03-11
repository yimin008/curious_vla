# Copyright 2026 Curious-VLA Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import os
import json
from typing import Dict, List, Any

def _get_mime_type(image_path: str) -> str:
    """Gets the MIME type based on the file extension."""
    extension = image_path.lower().split('.')[-1]
    if extension in ["jpg", "jpeg"]:
        return "image/jpeg"
    elif extension == "png":
        return "image/png"
    elif extension == "webp":
        return "image/webp"
    else:
        return "application/octet-stream"

def convert_to_openai_format(data: Dict[str, Any], dataset_dir: str = None) -> Dict[str, Any]:
    output_messages = []

    # 1. Encode all images listed in the 'images' field to Base64 data URIs
    image_data_uris = []
    if "images" in data and data["images"]:
        for image_path in data["images"]:
            if isinstance(image_path, str) and image_path.startswith("data:"):
                image_data_uris.append(image_path)
                continue
            full_path = os.path.join(dataset_dir, image_path) if dataset_dir else image_path
            if not os.path.exists(full_path):
                print(f"Warning: Image file not found at {full_path}. Skipping.")
                continue

            mime_type = _get_mime_type(full_path)
            with open(full_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                image_data_uris.append(f"data:{mime_type};base64,{base64_image}")

    # 2. Process the conversational messages
    # All images belong to the first user message containing an <image> tag
    images_injected = False
    for message in data.get("messages", []):
        role = message.get("role")
        content = message.get("content", "")

        if role == "user" and "<image>" in content and not images_injected:
            new_content = []
            for uri in image_data_uris:
                new_content.append({
                    "type": "image_url",
                    "image_url": {"url": uri}
                })

            output_messages.append({"role": "user", "content": new_content})
            text_content = content.replace("<image>", "").strip()
            new_content.append({"type": "text", "text": text_content})

            images_injected = True
        else:
            output_messages.append({"role": role, "content": content})

    return output_messages
