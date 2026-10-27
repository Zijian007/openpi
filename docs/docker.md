### Docker Setup

This checkout includes the policy-server Docker configuration. Example environments
are maintained in the [upstream OpenPI repository](https://github.com/Physical-Intelligence/openpi)
and are not included here. Run their Docker configurations from a separate upstream checkout.

- Basic Docker installation instructions are [here](https://docs.docker.com/engine/install/).
- Docker must be installed in [rootless mode](https://docs.docker.com/engine/security/rootless/).
- To use your GPU you must also install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- The version of docker installed with `snap` is incompatible with the NVIDIA container toolkit, preventing it from accessing `libnvidia-ml.so` ([issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/154)). The snap version can be uninstalled with `sudo snap remove docker`.
- Docker Desktop is also incompatible with the NVIDIA runtime ([issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/229)). Docker Desktop can be uninstalled with `sudo apt remove docker-desktop`.


If starting from scratch and your host machine is Ubuntu 22.04, you can use accomplish all of the above with the convenience scripts `scripts/docker/install_docker_ubuntu22.sh` and `scripts/docker/install_nvidia_container_toolkit.sh`.

Build the Docker image and start the container with the following command:
```bash
docker compose -f scripts/docker/compose.yml up --build
```

To build and run the Docker image for a specific example, use the following command:
```bash
docker compose -f /path/to/upstream-openpi/examples/<example_name>/compose.yml up --build
```
where `<example_name>` is the name of the example you want to run.

During the first run of any example, Docker will build the images. Go grab a coffee while this happens. Subsequent runs will be faster since the images are cached.
