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

import torch.nn as nn
from abc import ABCMeta

class BaseModule(nn.Module, metaclass=ABCMeta):
    """
    A minimal, dependency-free BaseModule.

    Used solely to satisfy the class inheritance requirement of CuriousVLAClient.
    In the Navsim Agent context, no complex weight initialization or logging is needed.
    """

    def __init__(self, init_cfg=None):
        super(BaseModule, self).__init__()
        self._is_init = False

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):
        """No-op init_weights for API compatibility."""
        self._is_init = True
