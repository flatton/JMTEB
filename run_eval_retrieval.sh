#!/bin/sh

for model in "BAAI/bge-m3" "bclavie/JaColBERTv2" "answerdotai/JaColBERTv2.5"; do
    poetry run python -m jmteb \
        --evaluators "src/jmteb/configs/tasks/jmteb_retrieval.jsonnet" \
        --embedder SentenceBertEmbedder \
        --embedder.model_name_or_path $model \
        --embedder.batch_size 1 \
        --embedder.device "mps" \
        --save_dir "output/"$model
done

for model in "intfloat/multilingual-e5-large" "intfloat/multilingual-e5-base" "pkshatech/GLuCoSE-base-ja-v2" "pkshatech/RoSEtta-base-ja"; do
    poetry run python -m jmteb \
        --evaluators "src/jmteb/configs/jmteb_retrieval_prompt_en.jsonnet" \
        --embedder SentenceBertEmbedder \
        --embedder.model_name_or_path $model \
        --embedder.batch_size 1 \
        --embedder.device "mps" \
        --save_dir "output/"$model
done

for model in "cl-nagoya/ruri-large" "cl-nagoya/ruri-base" "cl-nagoya/ruri-small"; do
    poetry run python -m jmteb \
        --evaluators "src/jmteb/configs/jmteb_retrieval_prompt_en.jsonnet" \
        --embedder SentenceBertEmbedder \
        --embedder.model_name_or_path $model \
        --embedder.batch_size 1 \
        --embedder.device "ops" \
        --save_dir "output/"$model
done
