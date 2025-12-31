"""
setup.py

Contains functions related to setting up training runs
"""

import os
import random
import numpy as np
import torch
import hashlib
import json
from argparse import ArgumentParser, Namespace
from typing import Optional, Union
from provada.utils.config import flatten_config
from provada.utils.log import get_logger
from provada.utils.config import load_config, update_config_from_args
from omegaconf import OmegaConf
from yaml import dump

logger = get_logger(__name__)

DEFAULT_WANDB_PROJECT = "provada"


def init_wandb(
    no_wandb: bool = False,
    run_name: str = None,
    config: Union[dict, OmegaConf] = None,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
):
    """
    Initializes a wandb run
    """
    # Check if wandb is disabled via environment variable
    if os.environ.get("WANDB_MODE") == "disabled":
        if not no_wandb:
            logger.warning(
                "WANDB_MODE environment variable is set to 'disabled'. "
                "Overriding no_wandb parameter to True."
            )
        no_wandb = True

    # If dict, convert to OmegaConf
    if isinstance(config, dict):
        config = OmegaConf.create(config)

    flattened_config = None
    if config is not None:
        flattened_config = flatten_config(config)

        # Remove the run_name from the config
        flattened_config.pop("run_name", None)

    if run_name is None:
        run_name = config.get("run_name", None)
        if run_name is None:
            raise ValueError(
                "run_name is not set. Please set the run_name in the config as config.run_name or pass it as an argument."
            )

    if no_wandb:
        wandb = None

    else:
        import wandb  # pylint: disable=import-outside-toplevel

        wandb.init(
            config=flattened_config,
            project=wandb_project,
            name=run_name,
            tags=get_tags_from_config(config),
        )

        # Save unflattened config to wandb as well (easier to reuse later)
        yaml_file_path = os.path.join(wandb.run.dir, "unflat_config.yaml")

        with open(yaml_file_path, "w") as f:
            dump(
                OmegaConf.to_container(config, resolve=True),
                f,
                default_flow_style=False,
                indent=2,
                sort_keys=False,
            )

        # Save the yaml file to wandb
        wandb.save(yaml_file_path)

    return wandb


def get_tags_from_config(config: dict):
    """
    Get the wandb tags from the config
    """
    tags = []
    if config.get("tags", None) is not None:
        tags = config.get("tags")

    # config.dataset.dataset_name
    if hasattr(config, "dataset"):
        if hasattr(config.dataset, "dataset_name"):
            tags.append(config.dataset.dataset_name)

    # config.model.model_name
    if hasattr(config, "model"):
        if hasattr(config.model, "model_name"):
            tags.append(config.model.model_name)

    # config.trainer.trainer_name
    if hasattr(config, "trainer"):
        if hasattr(config.trainer, "trainer_name"):
            tags.append(config.trainer.trainer_name)

    return tags


def seed_everything(seed):
    """
    Seed everything
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config_from_args(parser: ArgumentParser):
    """
    Get the arguments from the parser
    """
    # Parse args into known and unknown args
    known_args, unknown_args = parser.parse_known_args()

    # Load the config
    if hasattr(known_args, "config"):
        config = load_config(known_args.config)
    else:
        config = OmegaConf.create({})

    # Update the config with the known args (do not require keys to be in config since a few things may just be added like verbose)
    update_config_from_args(config, known_args, require_keys_to_be_in_config=False)

    if len(unknown_args) > 0:
        # If we have any unknown args, they must update the config (can't just be added)
        update_config_from_args(
            config,
            parse_flexible_unknown_args(unknown_args),
            require_keys_to_be_in_config=True,
        )

    return config


def parse_flexible_unknown_args(unknown_args_list):
    """
    Parses a list of strings which may include formats like
    --key value, --key=value, or --flag. Handles wandb agent style args.

    Args:
        unknown_args_list: A list of strings from parse_known_args().

    Returns:
        argparse.Namespace: A namespace containing the parsed arguments.
                            Keys are processed (e.g., --my-key -> my_key).
    """
    parsed_dict = {}
    i = 0
    while i < len(unknown_args_list):
        arg = unknown_args_list[i]

        # Ensure argument starts with '--'
        if not arg.startswith("--"):
            raise ValueError(
                f"Unknown argument without '--' prefix: {arg}",
                "All arguments must start with '--'",
            )

        key_str = arg
        value = True  # Default for flags

        # Check for --key=value format
        if "=" in arg:
            parts = arg.split("=", 1)
            key_str = parts[0]
            value_str = parts[1]
            key = key_str.lstrip("-").replace("-", "_")  # Process key

            # Attempt type inference on the value part
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    if value_str.lower() == "true":
                        value = True
                    elif value_str.lower() == "false":
                        value = False
                    else:
                        value = value_str  # Keep as string

            parsed_dict[key] = value
            i += 1  # Consumed one list item (--key=value)

        # Handle --key value format or --flag format
        else:
            key = key_str.lstrip("-").replace("-", "_")  # Process key

            # Check if next item exists and is *not* another key (i.e., it's a value)
            if i + 1 < len(unknown_args_list) and not unknown_args_list[i + 1].startswith(
                "--"
            ):
                value_str = unknown_args_list[i + 1]
                # Attempt type inference
                try:
                    value = int(value_str)
                except ValueError:
                    try:
                        value = float(value_str)
                    except ValueError:
                        if value_str.lower() == "true":
                            value = True
                        elif value_str.lower() == "false":
                            value = False
                        else:
                            value = value_str

                parsed_dict[key] = value
                i += 2  # Consumed key and value
            else:
                # It's a boolean flag (e.g., --no_checkpoint_saving), value remains True
                parsed_dict[key] = value
                i += 1  # Consumed key only

    return Namespace(**parsed_dict)


def get_run_name_from_config(
    config,
    script_name: Optional[str] = None,
):
    """
    Produces a run name from the config by generating a SHA-256 hash of the config
    """
    config.script = script_name

    if script_name is None:
        script_name = ""
    elif not script_name.endswith("_"):
        script_name = script_name + "_"

    run_name = script_name

    if hasattr(config, "config_name"):
        run_name = config.config_name + "_"

    def default_serializer(obj):
        if isinstance(obj, set):
            return sorted(list(obj))

    # Convert the config to a dictionary
    dict_version = OmegaConf.to_container(config, resolve=True)

    # Drop keys that don't matter for the config hash
    dict_version.pop("verbose", None)
    dict_version.pop("device", None)
    dict_version.pop("checkpoint_dir", None)
    dict_version.pop("no_cuda", None)
    dict_version.pop("no_wandb", None)
    dict_version.pop("debug", None)
    dict_version.pop("no_checkpoint_saving", None)
    dict_version.pop("num_workers", None)

    # Produce a string of the config
    config_str = json.dumps(
        dict_version,
        sort_keys=True,
        separators=(",", ":"),
        default=default_serializer,
    )
    if config_str is None:
        raise ValueError("Config conversion failed to be serialized")

    # Produce a hash of the config
    full_hash = hashlib.sha256(config_str.encode()).hexdigest()
    name_index = int(full_hash, 16) % len(RUN_NAMES)

    # Produce the run name
    return f"{run_name}{RUN_NAMES[name_index]}_{full_hash[:5]}"


RUN_NAMES = [
    "aang",
    "katara",
    "sokka",
    "toph",
    "iroh",
    "zuko",
    "azula",
    "ozai",
    "suki",
    "appa",
    "momo",
    "roku",
    "kyoshi",
    "luke",
    "leia",
    "hansolo",
    "yoda",
    "obiwan",
    "quigon",
    "padme",
    "palpatine",
    "sidious",
    "macewindu",
    "chewbacca",
    "r2d2",
    "c3po",
    "anakin",
    "darthvader",
    "bobafett",
    "jabba",
    "jangofett",
    "ashoka",
    "grevious",
    "dooku",
    "maul",
    "frodo",
    "sam",
    "gandalf",
    "aragorn",
    "legolas",
    "gimli",
    "saruman",
    "sauron",
    "boromir",
    "faramir",
    "eowyn",
    "gollum",
    "bilbo",
    "theoden",
    "galadriel",
    "witchking",
    "treebeard",
    "pippin",
    "merry",
    "smaug",
    "smeagol",
    "balrog",
    "pikachu",
    "charizard",
    "bulbasaur",
    "squirtle",
    "mewtwo",
    "mew",
    "gyarados",
    "eevee",
    "vaporeon",
    "jolteon",
    "flareon",
    "espeon",
    "umbreon",
    "leafeon",
    "glaceon",
    "lugia",
    "enti",
    "suicune",
    "raikou",
    "zapdos",
    "articuno",
    "moltres",
    "rayquaza",
    "kyogre",
    "groudon",
    "latias",
    "latios",
    "deoxys",
    "diagla",
    "palkia",
    "giratina",
    "arceus",
    "darkrai",
    "cresselia",
    "gengar",
    "alakazam",
    "machamp",
    "golem",
    "onix",
    "snorlax",
    "lapras",
    "dragonite",
    "harry",
    "hermione",
    "ron",
    "dumbledore",
    "snape",
    "voldemort",
    "hagrid",
    "malfoy",
    "mcgonagall",
    "dobby",
    "hedwig",
    "newton",
    "einstein",
    "curie",
    "darwin",
    "galilei",
    "tesla",
    "feynman",
    "hawking",
    "turing",
    "bohr",
    "heisenberg",
    "faraday",
    "sagan",
    "goodall",
    "franklin",
    "wheeler",
    "jumper",
    "charpentier",
    "doudna",
    "arnold",
    "smith",
    "watson",
    "crick",
    "pauling",
    "mendel",
    "fourier",
    "cooley",
    "tukey",
    "euler",
    "plato",
    "aristotle",
    "confucius",
    "descartes",
    "kant",
    "hume",
    "nietzsche",
    "socrates",
    "locke",
    "hobbes",
]
