# KoGPT2-Server

<br>

## Train (GPU Spot Instance)

```python
git clone https://github.com/imsoncod/KoGPT2-Server.git

python train.py
```

<br>

## Run (CPU Instance)

```python
git clone https://github.com/imsoncod/KoGPT2-Server.git

docker pull imsoncod/docker-kogpt2

docker run -p 5000:5000 imsoncod/docker-kogpt2
```
