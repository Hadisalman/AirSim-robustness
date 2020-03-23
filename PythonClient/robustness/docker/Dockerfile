# ARG BASE_IMAGE=pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
ARG BASE_IMAGE=nvidia/vulkan:1.1.121-cuda-10.1-alpha
FROM $BASE_IMAGE

RUN apt-get update
RUN apt-get install \
	git \ 
	libglu1-mesa-dev \
	pulseaudio \
	python3 \
	python3-pip \
	sudo \
	sudo \
	wget \ 
	x11-xserver-utils \
	xdg-user-dirs \
	unzip \
	-y --no-install-recommends

RUN pip3 install setuptools wheel
RUN pip3 install airsimneurips

RUN adduser --force-badname --disabled-password --gecos '' --shell /bin/bash airsim_user && \ 
	echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \ 
	adduser airsim_user sudo && \ 
	adduser airsim_user audio && \ 
	adduser airsim_user video

ENV PATH /opt/conda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sudo /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    sudo /opt/conda/bin/conda clean -tipsy && \
    sudo ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
	echo "source activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-10.1/lib64:/usr/local/cuda-10.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH

USER airsim_user
ENV USER airsim_user
WORKDIR /home/airsim_user
RUN sudo chown -R airsim_user /home/airsim_user

RUN	conda create -n airsim python=3.7 && \
	echo "source activate airsim" >> ~/.bashrc && \
	conda install -n airsim pytorch torchvision cudatoolkit=10.1 -c pytorch

RUN wget -c https://github.com/Hadisalman/AirSim/releases/download/v0.1-alpha/RobustnessNeighborhood.zip && \
	mkdir -p /home/$USER/Documents/AirSim && \
	unzip RobustnessNeighborhood.zip && \
	git clone https://github.com/Hadisalman/AirSim.git
	# mv RobustnessNeighborhood/ ../ && \
	# cd ../

RUN cd AirSim && git checkout 4.24 && cd ../ && \
	pip install AirSim/PythonClient

RUN pip install matplotlib "pillow<7" IPython opencv-python

RUN cp AirSim/PythonClient/robustness/settings.json ~/Documents/AirSim/

RUN wget -c https://github.com/Hadisalman/AirSim/releases/download/v0.1-alpha/RobustnessNeighborhood2.zip && \
	unzip RobustnessNeighborhood2.zip
