#!/bin/bash

chmod +x ./run_train_shadow.sh

chmod +x ./run_generate_obs.sh

chmod +x ./run_generate_attack_results.sh

./run_train_shadow.sh

./run_generate_obs.sh

./run_generate_attack_results.sh
