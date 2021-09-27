# -*- coding: utf-8 -*-
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Download config test."""

import unittest
from unittest.mock import MagicMock, patch

from lpot.ux.components.model_zoo.download_config import download_config


@patch("sys.argv", ["neural_compressor_bench.py", "-p5000"])
class TestDownloadConfig(unittest.TestCase):
    """DownloadConfig tests."""

    @patch("lpot.ux.components.model_zoo.download_config.Downloader")
    def test_download_config(self, downloader_mock: MagicMock) -> None:
        """Test download_config."""
        data = {
            "id": "some request id",
        }

        download_config(data)

        downloader_mock.assert_called_once_with(data)
        downloader_mock.return_value.download_config.assert_called_once()


if __name__ == "__main__":
    unittest.main()
