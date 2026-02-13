# docker/images/ubuntu22.04-cuda12.4.1/dn9.19.0.56-ros2_humble-cv_4.11.0.mk

# Define steps using format folder:profile
STEPS = 01_cudnn:9.19.0.56 \
        02_ros2:humble-ros-base \
        03_opencv:4.11.0
