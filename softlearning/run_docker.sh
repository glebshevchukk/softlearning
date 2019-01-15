#!/bin/bash
xhost +local:
sudo docker run -it --net=host \
  --device=/dev/video0 \
  --user=0 \
  -e DISPLAY=$DISPLAY \
  -e QT_GRAPHICSSYSTEM=native \
  -e CONTAINER_NAME=ros-kinetic-dev \
  -e USER=$USER \
  --workdir=/home/$USER \
  -v "/tmp/.X11-unix:/tmp/.X11-unix" \
  -v "/etc/group:/etc/group:ro" \
  -v "/etc/passwd:/etc/passwd:ro" \
  -v "/etc/shadow:/etc/shadow:ro" \
  -v "/etc/sudoers.d:/etc/sudoers.d:ro" \
  -v "/scr/glebs/:/scr/glebs/" \
  -v "/usr/lib/:/usr/lib/" \
  -v "/scr/glebs/deps/anaconda3:/scr/glebs/deps/anaconda3/" \
  -v "/usr/:/usr/" \
  --name=ros-kinetic-dev \
  kinetic:dev
