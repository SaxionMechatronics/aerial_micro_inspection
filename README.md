# ai_scanner 

# Installation 

## Docker (Ubuntu 22.04 + ROS 2 Humble + Gazebo Fortress)

The most simple installation is through docker using the docker file at `ai_scanner/Dockerfile`.

### 1. Build image

Build from the workspace `src` directory:

```bash
cd ~ && mkdir scanner_ws && cd ~/scanner_ws/src
docker build -f ai_scanner/Dockerfile -t ai_scanner:humble .
```

If your machine becomes unstable while compiling PX4 during image build, limit build parallelism:

```bash
docker build -f ai_scanner/Dockerfile -t ai_scanner:humble \
	--build-arg XRCE_BUILD_JOBS=1 \
	--build-arg PX4_BUILD_JOBS=1 \
	.
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

If your host/container cannot access GPU acceleration, run with software rendering:

```bash
docker run --rm -it \
	--gpus all \
	--net=host \
	-e DISPLAY=$DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	-e XAUTHORITY=$XAUTHORITY \
	-e LIBGL_ALWAYS_SOFTWARE=1 \
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

Make sure to make necessary modifications on the PX4 to add the custom airframe of our drone:
```bash
cp /ws/src/ai_scanner/simulation/gazebo/px4_airframe/2026_gz_x500_inspection /opt/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/
python3 /ws/src/ai_scanner/simulation/gazebo/px4_airframe/modify_px4.py
```

# Simulation

The following simulation is tested on Ubuntu 22.04, ROS Humble, and Ignition Gazebo Fortress.
## Dependencies
Make sure PX4-Autopilot development environment is installed by following the [instructions](https://docs.px4.io/main/en/dev_setup/dev_env_linux_ubuntu).

The Docker image now downloads PX4 source and builds PX4 SITL during `docker build`:

- PX4 source path in container: `/opt/PX4-Autopilot`
- PX4 build step in image: `make px4_sitl_default`
- `px4_msgs` is built from source in `/px4_msgs_ws` and sourced automatically in the container entrypoint.
- By default, `px4_msgs` tries to match `PX4_GIT_TAG`; you can override with `--build-arg PX4_MSGS_GIT_REF=<ref>`.
- Micro XRCE-DDS Agent is built from source (eProsima `Micro-XRCE-DDS-Agent`) and installed system-wide.


## Building the package

The simulation world and local models are in:

- `simulation/gazebo/worlds/tree_car_world.sdf`
- `simulation/gazebo/worlds/tree_model/`
- `simulation/gazebo/worlds/car_model/`

Launch simulation from inside the container:

```bash
ros2 launch ai_scanner tree_car_sim.launch.py
```

The container entrypoint and `tree_car_sim.launch.py` set Ignition/Gazebo Fortress resource environment variables automatically.

## PX4 Drone Simulation (Fortress + ROS2 bridge)

This package also provides an integrated launch that starts:

- PX4 SITL using `PX4_GZ_STANDALONE=1` and `PX4_GZ_WORLD=tree_car_world`
- A follow-up Gazebo startup through `scripts/simulation-gazebo`

Build and source first:

```bash
cd /ws
colcon build --symlink-install
source install/setup.bash
```

Launch full PX4 simulation:

```bash
ros2 launch ai_scanner tree_car_px4.launch.py
```

Useful launch arguments:

```bash
ros2 launch ai_scanner tree_car_px4.launch.py \
	px4_dir:=/opt/PX4-Autopilot \
	px4_make_target:=gz_x500_gimbal \
	px4_gz_model_pose:="2,-1,0.5,0,0,1.57" \
	world:=/ws/src/ai_scanner/simulation/gazebo/worlds/tree_car_world.sdf \
	extra_resource_path:=/ws/src/ai_scanner/simulation/gazebo/worlds/
```

`px4_make_target` selects the PX4 SITL target to build/run.

To launch the dual-camera inspection pipeline:

```bash
ros2 launch ai_scanner scanner.launch.py mission_config_file:=/ws/src/ai_scanner/config/simulation/mission.yaml det_mode:='ai'
```

## Models

## Configs

# Real test

## Calibrarion

## Test with ZED2 camera

