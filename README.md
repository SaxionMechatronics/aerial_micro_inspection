# ai_scanner 

Inspecting surfaces for tiny targets is challenging, especially with an aerial robot that is constantly moving, vibrating, and changing viewpoint. This package provides a modular dual-camera solution that combines a wide-FOV navigation camera with a zoomed camera on a gimbal so drones can automatically perform micro-detection workflows in flight. To adapt the system to a new application, you only need a YOLO segmentation model for surface segmentation and a YOLO detection model for tiny-subject detection; the rest of the pipeline remains reusable.

<iframe width="900" height="506" src="https://www.youtube.com/embed/biAapcSxhtY" title="ai_scanner demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

# Installation 

## Docker (Ubuntu 22.04 + ROS 2 Humble + Gazebo Harmonic)

The most simple installation is through docker using the docker file at `ai_scanner/Dockerfile`.

### 1. Build image

Build from the workspace `src` directory:

```bash
cd ~ && mkdir scanner_ws && cd ~/scanner_ws/src
docker build -f ai_scanner/Dockerfile -t ai_scanner:humble .
```

### 2. Run container with Gazebo GUI (Linux/X11)

Make sure NVIDIA Container Toolkit is installed on host, then run with GPU access:

```bash
xhost +local:docker

docker run --rm -it \
	--gpus all \
	--net=host \
	-e DISPLAY=$DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	-e XAUTHORITY=$XAUTHORITY \
	--device=/dev/dri:/dev/dri \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v $XAUTHORITY:$XAUTHORITY:ro \
	-v ~/scanner_ws/src:/ws/src:rw \
	ai_scanner:humble
```



### 3. Build workspace manually inside container

```bash
cd /ws
colcon build --symlink-install
source install/setup.bash
```

### 4. Simulation

Launch full PX4 simulation:

```bash
ros2 launch ai_scanner tree_car_px4.launch.py
```

To launch the dual-camera inspection pipeline, in a separate terminal, run:

```bash
ros2 launch ai_scanner scanner.launch.py mission_config_file:=/ws/src/ai_scanner/config/simulation/mission.yaml det_mode:='ai'
```

To visualize:

```bash
rviz2 -d /ws/src/ai_scanner/config/rviz_config.rviz
```

## Models

## Configs

# Real test

## Calibrarion

## Test with ZED2 camera

