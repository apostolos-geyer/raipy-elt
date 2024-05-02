from pathlib import Path
from functools import cache

import yaml

from typing import TypedDict, Required, Literal


CONFIGS_FILE = Path(__file__).with_suffix(".yaml")

RAW = "raw"
BRONZE = "bronze"
SILVER = "silver"
GOLD = "gold"

RAW_DIRNAME = f'0-{RAW}'
BRONZE_DIRNAME = f'1-{BRONZE}'
SILVER_DIRNAME = f'2-{SILVER}'
GOLD_DIRNAME = f'3-{GOLD}'

DIRNAMES = {
    RAW: RAW_DIRNAME,
    BRONZE: BRONZE_DIRNAME,
    SILVER: SILVER_DIRNAME,
    GOLD: GOLD_DIRNAME,
}


TABLES = {
    RAW: ("questions", "scoring"),
    BRONZE: ("questions", "scoring"),
    SILVER: (),
    GOLD: (),
}


class RawConfig(TypedDict):
    """
    Defines the expected keys in a configuration dict for
    reading in raw data
    """

    glob: Required[str]
    columns: Required[list[str | dict[str:str]]]
    dates: Required[list[dict[str:str]]]


class BronzeConfig(TypedDict):
    """
    Defines the expected keys in a configuration dict for
    reading in bronze data
    """

    table: Required[str]
    columns: Required[list[str | dict[str:str]]]
    dates: Required[list[dict[str:str]]]

@cache
def read_configs() -> dict:
    with CONFIGS_FILE.open('r') as conf:
        conf_dict = yaml.safe_load(conf)
    
    return conf_dict


def read_config(
    layer: Literal["raw", "bronze", "silver", "gold"], table: str
) -> RawConfig | BronzeConfig:
    if layer not in (RAW):
        raise NotImplementedError("Configuration above raw layer not implemented")

    config: dict = read_configs()
    table_layer_conf = config.get(layer, {}).get(table)

    if table_layer_conf is None:
        raise AttributeError(
            f"Configuration not found for table {table} in layer {layer}"
        )

    return table_layer_conf