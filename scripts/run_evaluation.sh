ZhengMing_path='/root/ZhengMing'
export PYTHONPATH="$CLLLM_path/src:$CLLLM_path/src/literature-evaluation:$CLLLM_path/src/metrics/BARTScore"
echo $PYTHONPATH
export CUDA_VISIBLE_DEVICES="0"

cd ..
python src/eval.py \
    --model hf-causal-vllm \
    --tasks ZM_aclue \
    --model_args use_accelerate=True,pretrained=Baichuan2-7B-Base,tokenizer=Baichuan2-7B-Base,max_gen_toks=1024,use_fast=False,dtype=float16,trust_remote_code=True \
    --no_cache \
    --batch_size 5 \
    --write_out