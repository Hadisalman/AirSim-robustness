# Robustness Challenge


## Quickstart
The following should get you all set to run the RobustnessNeighborhood environment, in which various robustness researc tasks can be done! This is all done using the AirSim simulators.

#### Setup

We provide [Linux] binaries only. The following downloads the binary for the RobustnessNeighborhood, and installs the python api used to play with the environment.
- Download and extract the binary (works with both vulkan and opengl):
    ```
    wget -c https://github.com/Hadisalman/AirSim/releases/download/v0.1-alpha/RobustnessNeighborhood_opengl.zip && \
    unzip RobustnessNeighborhood_opengl.zip
    ```

- Create a conda environment and install our Python API and its dependecies dependencies:
    ```
    conda create -n airsim python=3.7 && \
    conda activate airsim && \
    git clone https://github.com/Hadisalman/AirSim.git && \
    pip install Airsim/PythonClient
    conda install -n airsim pytorch torchvision cudatoolkit=10.1 -c pytorch
    pip install matplotlib "pillow<7" IPython opencv-python
    ``` 
- Download and move the `settings.json` file to `~/Documents/AirSim/settings.json`.
    ```
    mkdir -p ~/Documents/AirSim &&
    cp AirSim/PythonClient/robustness/settings.json ~/Documents/AirSim
    ```

#### Running
- Open a Linux terminal and enter the following command:
    ```
    cd RobustnessNeighborhood/
    ./AirSimNH.sh -windowed
    ```
- Running headless (with rendering of images enabled):
    ```
    DISPLAY= ./AirSimNH.sh
    ```
- To disable rendering completely (no `simGetImages`), you can use:
    ```
    ./AirSimNH.sh -nullrhi
    ```

- To increase speed of `simGetImages` / increase speed of Unreal Engine's game thread;
    - Add the `"ViewMode": "NoDisplay"` to your settings.json file. This disables rendering in the main viewport camera.Then run the binary with the following options.  
        ```
        ./AirSimNH.sh -windowed -NoVSync -BENCHMARK
        ```
    
    Check how fast the game is running using the Unreal console commands `Stat FPS`, `Stat UnitGraph`, `r.VSync`, `t.maxFPS`.
- **Note: add `-opengl` to any of the above commands to run with opengl instead of vulkan.**

## Docker
- Prerequisites:
	- Install [docker-ce](https://docs.docker.com/install/linux/docker-ce/ubuntu/). 
	- Complete the desired [post-installation steps for linux](https://docs.docker.com/install/linux/linux-postinstall/) after installing docker.    
	At the minimum, the page tells you how torun docker without root, and other useful setup options. 
	- Install [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)). 

- Dockerfile:
	We provide a sample [dockerfile](docker/Dockerfile) you can modify.   

- Building the docker image:    
	You can use [build_docker_image.py](docker/build_docker_image.py) to build the dockerfile above (or your own custom one)    
	**Usage** (with default arguments)
	```shell
	cd docker/;
	python3 build_docker_image.py \
		--dockerfile Dockerfile \
		--base_image nvidia/vulkan:1.1.121-cuda-10.1-alpha \
		-- target_image airsim_robustness:nvidia/vulkan:1.1.121-cuda-10.1-alpha
	```
- Running the docker image:
	See [docker/run_docker_image.sh](docker/run_docker_image.sh) to run the docker image:   
	**Usage**
	- for running a custom image in windowed mode, pass in you image name and tag:    
	    `$ ./run_docker_image.sh DOCKER_IMAGE_NAME:TAG`
	- for running a custom image in headless mode, pass in you image name and tag, followed by "headless":    
	     `$ ./run_docker_image.sh DOCKER_IMAGE_NAME:TAG headless`

## Demo
When you get **the binary running** (either using docker or by manually setting things up), you can try a demo that demonstrates stuff you can do in the `Robustness Neighborhood` environment.
```
cd AirSim/PythonClient/robustness;
conda activate airsim;
python robustness_utils_demo.py results/ped_recognition_new/224x224/model_best.pth.tar --demo-id 0;
```
`demo-id` an take several values :
- 0 -> image callback thread
- 1 -> test all threads
- 2 -> search for 3D advesarial configuration
- 3 -> read adv_config.json and run ped recognition
- 4 -> pixel pgd attack
