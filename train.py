import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from baselines.ippo.run_sb3_ppo import train
import pyrallis
from baselines.ippo.config import ExperimentConfig
from pygpudrive.env.config import SceneConfig

# Helper function to handle enums and skip unwanted fields
def dataclass_to_serializable(dataclass_instance):
    if not is_dataclass(dataclass_instance):
        raise ValueError("Provided instance is not a dataclass")
    
    # Convert dataclass to a dictionary
    data_dict = asdict(dataclass_instance)

    # Iterate through fields and handle enums
    for field in fields(dataclass_instance):
        value = getattr(dataclass_instance, field.name)
        if isinstance(value, Enum):
            # Replace enum with its value
            data_dict[field.name] = value.value
    
    return data_dict

exp_config = pyrallis.parse(config_class=ExperimentConfig)

# save experiment config
exp_config_dict = dataclass_to_serializable(exp_config)

# Write the JSON file
with open("saved_policies/exp_config.json", "w") as json_file:
    json.dump(exp_config_dict, json_file, indent=4)

scene_config = SceneConfig(
    path=exp_config.data_dir,
    num_scenes=exp_config.num_worlds,
    discipline=exp_config.selection_discipline,
    k_unique_scenes=exp_config.k_unique_scenes,
)

# save scene config
scene_config_dict = dataclass_to_serializable(scene_config)

with open("saved_policies/scene_config.json", "w") as json_file:
    json.dump(scene_config_dict, json_file, indent=4)

NUM_SEEDS = 3
exp_config.seed += 0

for i in range(NUM_SEEDS):

    print(f"training with seed {exp_config.seed}...")
    train(exp_config, scene_config)
    print(f"done training seed {exp_config.seed}")

    exp_config.seed += 1