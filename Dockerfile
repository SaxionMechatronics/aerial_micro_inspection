FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    wget \
    gnupg2 \
    lsb-release \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    ros-humble-ros-gz \
    ros-humble-ros-gz-bridge \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-message-filters \
    ros-humble-tf-transformations \
    ros-humble-vision-opencv \
    ros-humble-pcl-ros \
    ros-humble-ament-cmake \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-pip \
    python3-opencv \
    python3-numpy \
    python3-scipy \
    python3-yaml \
    python3-jinja2 \
    python3-empy \
    python3-toml \
    python3-jsonschema \
    python3-packaging \
    build-essential \
    cmake \
    ninja-build \
    ccache \
    libasio-dev \
    libtinyxml2-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init 2>/dev/null || true

WORKDIR /ws

ARG PX4_GIT_TAG=v1.16.0
ARG MICRO_XRCE_DDS_AGENT_TAG=v2.4.3
ARG XRCE_BUILD_JOBS=2
ARG PX4_BUILD_JOBS=8

RUN git clone --depth 1 --branch ${MICRO_XRCE_DDS_AGENT_TAG} https://github.com/eProsima/Micro-XRCE-DDS-Agent.git /tmp/Micro-XRCE-DDS-Agent && \
    cmake -S /tmp/Micro-XRCE-DDS-Agent -B /tmp/Micro-XRCE-DDS-Agent/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/Micro-XRCE-DDS-Agent/build --parallel ${XRCE_BUILD_JOBS} && \
    cmake --install /tmp/Micro-XRCE-DDS-Agent/build && \
    rm -rf /tmp/Micro-XRCE-DDS-Agent

RUN git clone --depth 1 --branch ${PX4_GIT_TAG} https://github.com/PX4/PX4-Autopilot.git /opt/PX4-Autopilot && \
    cd /opt/PX4-Autopilot && \
    git submodule update --init --recursive && \
    DONT_RUN=1 bash ./Tools/setup/ubuntu.sh --no-nuttx && \
    make -j${PX4_BUILD_JOBS} px4_sitl_default

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir --retries 10 --timeout 1000 \
            --index-url ${TORCH_INDEX_URL} \
            torch==2.5.1 \
            torchvision==0.20.1 && \
        # TensorRT Python wheels are served via NVIDIA's index.
        pip3 install --no-cache-dir --retries 10 --timeout 1000 \
            --extra-index-url https://pypi.nvidia.com \
            "tensorrt>7.0.0,!=10.1.0" && \
        pip3 install --no-cache-dir --retries 10 --timeout 1000 \
            ultralytics==8.4.14 \
            pymavlink==2.4.49 \
            "onnx>=1.12.0,<2.0.0" \
            "onnxslim>=0.1.71" \
            onnxruntime-gpu \
            "transforms3d>=0.4.1" && \
        # Keep numpy at the tested version for tf_transformations/transforms3d compatibility.
        pip3 install --no-cache-dir --retries 10 --timeout 1000 --upgrade "numpy==1.26.4" && \
        # Keep OpenCV Python below 4.12 for compatibility with current stack.
        pip3 install --no-cache-dir --retries 10 --timeout 1000 --upgrade "opencv-python<4.12" && \
        python3 -c "import cv2; from packaging.version import Version; assert Version(cv2.__version__) < Version('4.12.0'), cv2.__version__"

ARG PX4_MSGS_GIT_REF=
RUN set -e; \
        PX4_VERSION_STRIPPED="${PX4_GIT_TAG#v}"; \
        PX4_VERSION_MINOR="${PX4_VERSION_STRIPPED%.*}"; \
        PX4_MSGS_REF="${PX4_MSGS_GIT_REF}"; \
        if [[ -z "${PX4_MSGS_REF}" ]]; then \
            PX4_MSGS_REF="${PX4_GIT_TAG}"; \
        fi; \
        mkdir -p /px4_msgs_ws/src; \
        git clone https://github.com/PX4/px4_msgs.git /px4_msgs_ws/src/px4_msgs; \
        cd /px4_msgs_ws/src/px4_msgs; \
        if git rev-parse --verify --quiet "refs/tags/${PX4_MSGS_REF}" >/dev/null; then \
            git checkout "${PX4_MSGS_REF}"; \
        elif git rev-parse --verify --quiet "refs/remotes/origin/release/${PX4_VERSION_MINOR}" >/dev/null; then \
            git checkout "release/${PX4_VERSION_MINOR}"; \
        else \
            echo "No matching px4_msgs ref found for PX4 ${PX4_GIT_TAG}."; \
            echo "Set --build-arg PX4_MSGS_GIT_REF=<ref> explicitly."; \
            exit 1; \
        fi; \
        source /opt/ros/humble/setup.bash; \
        cd /px4_msgs_ws; \
        colcon build --symlink-install

# For ros_gz bridge to work, we must make sure that the ros-gz-bridge compatible with Gazebo Harmonic is installed.
RUN apt-get update && \
        apt-get remove -y ros-humble-ros-gz-sim || true && \
        apt-get install -y --no-install-recommends ros-humble-ros-gzharmonic; \
        rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=humble
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV PATH=/root/.local/bin:${PATH}

COPY ai_scanner/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]