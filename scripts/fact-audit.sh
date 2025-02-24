CUDA_VISIBLE_DEVICES=0

tasks=("complex_claim" "fake_news" "social_rumor")

for task in ${tasks[@]};
do
    echo ${task}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} HF_HUB_CACHE=/next_share/hf_cache/hub python fact-audit.py \
    --category ${task} 
done