# Install
conda env
```
conda create -n verl python==3.10
conda activate verl
```

verl  
Here you may run the shell script or install them yourself
```
# Install dependencies
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# Install verl
pip install --no-deps -e .
```

sqlite3
```
sudo apt install sqlite3 libsqlite3-dev
sqlite3 --version
```

other
```
pip install chardet
```


