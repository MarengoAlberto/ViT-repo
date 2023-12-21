# Copyright The Lightning AI team.
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

import logging
import os

from typing_extensions import override
from lightning.fabric.utilities.rank_zero import rank_zero_only

from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment

log = logging.getLogger(__name__)


class KubeflowEnvironment(ClusterEnvironment):
    """Environment for distributed training using the `PyTorchJob`_ operator from `Kubeflow`_.

    This environment, unlike others, does not get auto-detected and needs to be passed to the Fabric/Trainer
    constructor manually.

    .. _PyTorchJob: https://www.kubeflow.org/docs/components/training/pytorch/
    .. _Kubeflow: https://www.kubeflow.org

    """
    def __init__(self) -> None:
        super().__init__()
        self._main_port: int = -1
        self._global_rank: int = 0
        self._world_size: int = 1

    @property
    @override
    def creates_processes_externally(self) -> bool:
        """Returns whether the cluster creates the processes or not.

        If at least :code:`LOCAL_RANK` is available as environment variable, Lightning assumes the user acts as the
        process launcher/job scheduler and Lightning will not launch new processes.

        """
        # return "LOCAL_RANK" in os.environ
        return True

    @property
    @override
    def main_address(self) -> str:
        return os.environ["MASTER_ADDR"]

    @property
    @override
    def main_port(self) -> int:
        return int(os.environ["MASTER_PORT"])

    @staticmethod
    @override
    def detect() -> bool:
        return True

    @override
    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])

    @override
    def set_world_size(self, size: int) -> None:
        self._world_size = size

    @override
    def global_rank(self) -> int:
        return int(os.environ["RANK"])

    @override
    def set_global_rank(self, rank: int) -> None:
        self._global_rank = rank
        rank_zero_only.rank = rank

    @override
    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    @override
    def node_rank(self) -> int:
        group_rank = os.environ.get("GROUP_RANK", 0)
        return int(os.environ.get("NODE_RANK", group_rank))

    @override
    def teardown(self) -> None:
        if "WORLD_SIZE" in os.environ:
            del os.environ["WORLD_SIZE"]
