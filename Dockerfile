FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        git \
        wget

RUN git clone -b feat-docker https://github.com/aisingapore/IndoMMLU.git indommlu
WORKDIR /workspace/llm_benchmarks
RUN wget \
    https://raw.githubusercontent.com/mesolitica/malaysian-dataset/eb740445ad13857278d47cc17636b515aa339fbd/llm-benchmark/BM-pt3/BM-A-pt3 \
    https://raw.githubusercontent.com/mesolitica/malaysian-dataset/eb740445ad13857278d47cc17636b515aa339fbd/llm-benchmark/BM-pt3/BM-B-pt3 \
    https://raw.githubusercontent.com/mesolitica/malaysian-dataset/eb740445ad13857278d47cc17636b515aa339fbd/llm-benchmark/tatabahasabm.tripod.com/quiz-tatabahasa.jsonl
RUN pip install -r requirements.txt

ENTRYPOINT ["/workspace/llm_benchmarks/entrypoint.sh"]
