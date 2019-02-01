FROM osrf/ros:kinetic-desktop-full
ENV LANG en_US.UTF-8
RUN apt-get update && apt-get install -y \
            tmux \
            zsh \
            curl \
            wget \
            vim \
            emacs24 \
            sudo \
            unzip \
            python-catkin-tools \
            && rm -rf /var/likb/apt/lists/*
CMD ["zsh"]
