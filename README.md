# installation steps

install docker-ce

install nvidia-container-toolkit

sudo nvim /etc/nvidia-container-runtime/config.toml in here change no-cgroups to false

# vscode setup

```bash
cd ~/Documents # move Documents
mkdir gpudrive_dev
docker run -v ~/Documents/gpudrive_dev:/home -v ~/Datasets:/mnt --gpus all -it --name gpudrive_container ghcr.io/emerge-lab/gpudrive:latest
```
This should create a workspace called gpudrive_dev (feel free change name) in your Documents, and then run the docker image and mount that workspace in /home directory of the container (-v ~/Documents/gpudrive_dev:~) and mount my datasets in /home as well -v ~/Datasets:/home. As well as pass gpus through (--gpus all) and make it interactive (-it), name the container gpudrive_container, and specify the image we are using is ghcr.io/emerge-lab/gpudrive:latest

This annoyingly doesnt assign the same UID and GID in the container as the host ubuntu system in my case, meaning that I need to chown the files created within the container to be writable outside of the container. To change this we need to change the dockerfile from the authors - I will ask them on monday.

- then you should quit the interactive terminal that the docker run produced
- then you should run the docker container from the docker extension in a window in vscode (or run docker run gpudrive_container)
- then you should attach to the running container from vscode
- then you should install the python development extensions in the connected instance to the container
- then you should run poetry install in /gpudrive in the container

# git setup

- need ssh installed for ssh pushes and so on: apt-get install openssh-client (after apt get update)

