# utility is a function that is used frequently in the code
import base64
import json
import os
from pathlib import Path
from typing import Any, List, Union

import joblib
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from cnnClassifier import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully!")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty.")
    except Exception as e:
        raise e


# Do not put -> None when a function returns None as it breaks this apparently or I just don't know how to assign types using this lib.
@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """create a list of directories

    Args:
        path_to_directories (list): list of path of directories
        verbose (bool): log what directory is created
    """

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")
