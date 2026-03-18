# Installation Guide (Orin Nano, Ubuntu 22.04, JetPack 6, L4T 36.3)

This guide covers native (non-Docker) setup and usage for real testing with:
- ZED2 as navigation camera
- USB inspection camera (Sony ILX-LR)
- PX4 interface via `px4_msgs`

## 1. Install ROS 2 Humble

Follow the official ROS 2 Humble Ubuntu installation guide:
- https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

## 2. Install ZED SDK and ZED ROS 2 wrapper

Follow the official ZED pages:
- ZED SDK downloads: https://www.stereolabs.com/developers/release/
- ZED ROS 2 wrapper: https://github.com/stereolabs/zed-ros2-wrapper

For installing `ZED_Wrapper`, we suggest either building it in a common workspace (such as `scanner_ws`) or in case of a dedicated workspace, make sure to add sourcing it to `.bashrc`.

## 3. Build `px4_msgs` in a dedicated workspace

Choose the `px4_msgs` branch/tag that matches your PX4 firmware.
For PX4 `1.15.x`, use `release/1.15`.

```bash
source /opt/ros/humble/setup.bash
cd ~/scanner_ws/src
git clone --depth 1 --branch release/1.15 https://github.com/PX4/px4_msgs.git

cd ~/scanner_ws
colcon build --symlink-install
```

## 4. Clone and compile the package

```bash
cd ~/scanner_ws/src
git clone https://github.com/SaxionMechatronics/aerial_micro_inspection.git

cd ~/scanner_ws
colcon build --symlink-install
```

Optional: add sourcing to `~/.bashrc` for new terminals:

```bash
echo "source ~/scanner_ws/install/setup.bash" >> ~/.bashrc
```

## 5. Install Python dependencies

```bash
cd ~/scanner_ws/src/aerial_micro_inspection
pip3 install --no-cache-dir -r requirements.jetson-real.txt
```

NVIDIA Jetson PyTorch reference:
- https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html

> [!WARNING]
> 1. Make sure your `torch` installation is GPU compatible
> ```bash
> python3 -c "torch.cuda.is_available()"
> ```
> 2. Make sure your numpy version is not above 2.0:
> ```bash
> pip3 install numpy==1.26.4
> ```

## 6. Place model checkpoints and set model paths

Some example checkpoints can be downloaded from:
https://drive.google.com/file/d/1DVNta4nifv4AS4VwWwGPogL-U68__rrm/view?usp=sharing

Place checkpoints in:

```bash
~/scanner_ws/src/aerial_micro_inspection/weights/
```

Update model paths in:

```bash
~/scanner_ws/src/aerial_micro_inspection/config/real_test/mission.yaml
```

Required fields:
- `surface_segmentation.path`
- `micro_detection.path`


## 7. Enable TensorRT for real testing

In `config/real_test/mission.yaml`:

```yaml
runtime:
  use_trt: true
```

> [!Note]
> First-time TensorRT optimization can take several minutes.


## 8. Run and visualize

```bash
ros2 launch aerial_micro_inspection scanner_zed.launch.py \
  mission_config_file:=~/scanner_ws/src/aerial_micro_inspection/config/real_test/mission.yaml \
  det_mode:=ai
```

```bash
rviz2 -d ~/scanner_ws/src/aerial_micro_inspection/config/rviz_config.rviz
```

## Troubleshooting

If you see `AttributeError: module 'numpy' has no attribute 'float'`, apply this patch to `transforms3d`:

```bash
sudo sed -i 's/np\.float/float/g' /usr/lib/python3/dist-packages/transforms3d/quaternions.py
```
