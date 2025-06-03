
local_dir=data/corpus/2wikimultihopqa
system_template="You are a helpful assistant that can solve the given question step by step. For each step, start by explaining your thought process. If additional information is needed, provide a specific query enclosed in <search> and </search>. The system will return the top search results within <observation> and </observation>. You can perform multiple searches as needed.
When you know the final answer, use <original_evidence> and </original_evidence> to provide all potentially relevant original information from the observations. Ensure the information is complete and preserves the original wording without modification. If no searches were conducted or observations were made, omit the evidence section. Finally, provide the final answer within <answer> and </answer> tags."

python src/pre_data/2wikimultihopqa.py \
    --local_dir $local_dir \
    --system_template "$system_template"