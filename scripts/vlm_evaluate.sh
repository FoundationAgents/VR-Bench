#!/bin/bash

CONFIG_FILE=config/vlm/$your_config$.yaml

python -m evaluation.vlm_eval.run_vlm_eval "$CONFIG_FILE"

