chmod -R 777 /home/research/LLM_halluginate

conda create -n hallucination_env python=3.10.12 -y
conda activate hallucination_env


conda env update --file environment.yml --name hallucination_env


wget https://huggingface.co/datasets/fava-uw/fava-data/resolve/main/annotations.json

chmod +x run.sh
huggingface-cli login

export HF_HOME="/home/japmyy/.cache/huggingface"
changed the dtype . 
./run.sh
Then I ran the check_scores_fava_annotated.ipynb
