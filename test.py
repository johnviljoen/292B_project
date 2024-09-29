import os
import gpudrive

from pygpudrive.env.config import SceneConfig
from pygpudrive.env.scene_selector import select_scenes

scene_config = SceneConfig(
    path="/mnt/nocturne_mini/formatted_json_v2_no_tl_train/formatted_json_v2_no_tl_train", 
    num_scenes=1
)

sim = gpudrive.SimManager(
    exec_mode=gpudrive.madrona.ExecMode.CUDA, # Specify the execution mode
    gpu_id=0,
    scenes=select_scenes(scene_config),
    params=gpudrive.Parameters(), # Environment parameters
)

