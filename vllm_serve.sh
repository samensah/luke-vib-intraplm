#!/bin/bash
# Serve model with vLLM
export CUDA_VISIBLE_DEVICES=0

python -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-4-mini-instruct \
    --port 8000 \
    --host 0.0.0.0


Large language models have enabled natural language interfaces for enterprise data access, allowing users to query client information conversationally. While LLMs excel at understanding query intent, resolving client mentions to database entities remains a bottleneck. Users refer to clients through abbreviations (JPMC for JPMorgan Chase), acronyms, shorthand conventions (Mgmt. for Management), and identifiers with no lexical overlap to canonical names—and mentions often contain typos. Embedding the full client database in the prompt context is infeasible at scale, necessitating a retrieval-based resolution step. In domains such as financial services, resolving to the wrong client risks erroneous analysis, regulatory exposure, and loss of user trust, demanding high precision at rank one.
We present a production system combining a fuzzy matching tool with an LLM-based ReAct agent for client resolution. Given an ambiguous mention, the agent queries the fuzzy matcher to retrieve scored candidates, then iteratively refines the mention—correcting typos, expanding abbreviations, or reformulating—and re-queries until a confidence threshold is met. This self-correcting loop addresses compounding difficulty when multiple challenges co-occur, where single-pass approaches fail. We evaluate on real enterprise data, demonstrating improvements in hits@1 over baseline methods. Our work offers practical insights for integrating retrieval tools into LLM-powered systems where context limitations preclude in-prompt enumeration and first-result accuracy is critical
