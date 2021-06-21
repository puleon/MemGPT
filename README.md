First, install requirements:

```bash
pip install -r requirements.txt
```

To train the MemGPT-2 model run:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; python run_mem_clm.py --do_train --model_name_or_path gpt2 --output_dir ./memgpt2_wikipedia --per_device_train_batch_size=2 --block_size 512 --dataset_name wikipedia --dataset_config_name 20200501.en --save_steps 1000 --preprocessing_num_workers 32
```

To evaluate the trained MemGPT-2 model on the Penn Treebank (PTB) corpus run:

```bash
wget https://files.deeppavlov.ai/memgpt2/memgpt2_wikipedia.tar.gz
```

```bash
tar -xvzf memgpt2_wikipedia.tar.gz
```

```bash
python eval_mem_gpt2.py --model_name_or_path ./memgpt2_wikipedia/checkpoint-9000
```
