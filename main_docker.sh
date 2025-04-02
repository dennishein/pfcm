docker run -u dhein -it --gpus 'device=0' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -p 8896:8896 -v "$(pwd)":/home test
