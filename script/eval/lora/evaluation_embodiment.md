# Evaluation for embodiment tasks
We provide a simple communication mechanism like [ROS](https://www.ros.org/), for you to run robots, or virtual robots in one machine or virtual environment (venv), 
send the images and language instructions to SpatialBot by ssh and sftp, run SpatialBot in another machine or venv, and send robot actions back to the robot.

## Calvin
1. Refer to [Calvin GitHub](https://github.com/mees/calvin) to prepare data and venv.
2. In Calvin venv, run ```calvin/calvin_models/calvin_agent/evaluation/evaluate_policy_local_spatialbot.py``` 
or ```calvin/calvin_models/calvin_agent/evaluation/evaluate_policy_remote_spatialbot.py``` (We will provide the scripts soon).
3. In SpatialBot venv, run SpatialBot to generate robot actions by
```shell
sh script/eval/lora/calvin_infer.sh
```


## Open X-Embodiment (RTX)
1. We use datasets downloaded from [unofficial RTX HuggingFace repo](https://huggingface.co/datasets/jxu124/OpenX-Embodiment).
2. Prepare images and data json with our script. (Available soon)
3. Evaluate model by
```shell
sh script/eval/lora/remote_rtx_infer.sh
```
4. If you would like to see open-loop results by evaluating model on a validation episode, run
```shell
sh script/eval/lora/local_rtx_infer.sh
```
Results will be written to ```control.png```