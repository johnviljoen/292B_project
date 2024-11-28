# installation

<details>
  <summary>Docker installation (most robust)</summary>

## Supported Platforms

- Linux, Ubuntu 20.04 or newer

## General Dependencies

- CMake 3.24 or newer
- Python 3.9 or newer
- CUDA 12.2 or newer

## (Optional) for Madrona Viewer 

sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev mesa-common-dev libc++1

## Docker dependencies
- [install docker-ce](https://docs.docker.com/engine/install/ubuntu/)
- [install nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (A problem I ran into required: sudo vim /etc/nvidia-container-runtime/config.toml in here change no-cgroups to false)

# installation steps

## Create a workspace

This will be mounted in the docker container so we can run it from within the container later.

```bash
mkdir ~/Documents/gpudrive_dev
```

## Get the Waymo datasets

- Create a Datasets directory in your /home/${USER}

```bash
mkdir ~/Datasets
```

- Download the [mini dataset](https://www.dropbox.com/sh/8mxue9rdoizen3h/AADGRrHYBb86pZvDnHplDGvXa?dl=0)
- (Optional but recommended) Download the [large dataset](https://www.dropbox.com/sh/wv75pjd8phxizj3/AABfNPWfjQdoTWvdVxsAjUL_a?dl=0)

- unzip the files into ~/Datasets/ such that ~/Datasets follows the structure:

```bash
.
├── nocturne
│   ├── formatted_json_v2_no_tl_train
│   └── formatted_json_v2_no_tl_valid
└── nocturne_mini
    ├── formatted_json_v2_no_tl_train
    └── formatted_json_v2_no_tl_valid
```

## Pull the image and run the container

```bash
docker pull ghcr.io/emerge-lab/gpudrive:latest
```

## Create the container and run it

```bash
docker run -v ~/Documents/gpudrive_dev:/home -v ~/Datasets:/mnt --gpus all -it --name gpudrive_container ghcr.io/emerge-lab/gpudrive:latest
```

## Build GPUDrive

From within the running container you have just created

```bash
poetry install
```

Now you have a working installation of GPUDrive, but it is not yet setup to be debuggable and editable from VSCode.

## Install dependencies for IPPO baseline

```bash
cd /home
pip install -e .
```

## Final Steps

- You will need to sort some stuff out which isnt installed, ffpeg, imageio, imageio-ffmpeg, moviepy, wandb\[media\] and so on
- downgrad moviepy to 1.0.3 (definitely less than 2.0)

## (optional but recommended) local vscode setup

NOTE: that if you want to do remote code running and creation via ssh you will need to setup the system to also work locally as is shown here.

### Required extensions

- install docker extension
- install dev containers extension

### Running the container

- click the docker logo in the left hand side
- under containers, right click ghcr.io/emerge-lab/gpudrive:latest
- if it is not running, select start
- once/if it is running, right click on it again
- select attach visual studio code

This will generate a new VSCode window that is attached in a similar manner to SSH but inside the container.

You should notice that the gpudrive_dev directory we created earlier is mounted within the /home directory of the container itself, and that the gpudrive conda environment is automatically activate, and that we have already built gpudrive, so we are now ready to go!

- (optional) install the python development packages within the container connected vscode, lets you use conda environments nicely inside the container

## (optional but recommended) remote vscode

This is a rather simple extension if we have already setup the local vscode instance with the docker container inside it. 

- install the necessary extensions for remote ssh
- install the dev-containers extension here as well (unsure if necessary, but I have it)
- ssh connect to the remote server with the container
- ensure that the docker and dev-container extensions are also installed on the server
- start the container as previously shown
- attach the container as previously shown
- you should have a remote VSCode instance within the container within the server ready to go!
- (optional) install the python development packages within the remote container vscode, lets you use conda environments nicely inside the container

</details>

<details>
  <summary>Annoyances</summary>

## UID/GID misalignment with host

This dockerfile annoyingly doesn't assign the same UID and GID in the container as the host ubuntu system in my case, meaning that I need to chown the files created within the container to be writable outside of the container. To change this we need to change the dockerfile from the authors - I will ask them on monday.

## Github setup

to push to github via ssh we need to install the ssh client in the container, this could be done in the dockerfile:

```bash
apt-get install openssh-client (after apt get update)
```

# alternative interactions with container

You can also get into the container via terminal to check things quickly from the machine hosting the container.

```bash
docker run gpudrive_container
docker exec -it something something I cant remember
```

</details>


# Run IPPO

```bash
python baselines/ippo/run_sb3_ppo.py
```

