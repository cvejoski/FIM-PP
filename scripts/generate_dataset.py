from copy import deepcopy
from pathlib import Path

import click

from fim.leftovers_from_old_library import create_class_instance, load_config, save_in_yaml


@click.command()
@click.option("-c", "--config", "cfg_path", required=True, type=click.Path(exists=True), help="path to config file")
def generate_data(cfg_path: Path) -> None:
    cfg: dict = load_config(cfg_path)
    cfg_copy: dict = deepcopy(cfg)

    # prepare dataset path
    dataset_path: Path = Path("data/synthetic_data/" + cfg["process_type"] + "/" + cfg["dataset_name"])
    # Drop dataset_path from cfg
    cfg.pop("dataset_name")
    cfg.pop("process_type")

    # save original config yaml
    save_in_yaml(cfg_copy, dataset_path, "config")

    data_generator = create_class_instance(cfg["data_generator"])
    data = data_generator.assemble()

    # save data
    data_saver = create_class_instance(cfg["data_saver"])
    data_saver(data)


if __name__ == "__main__":
    generate_data()
