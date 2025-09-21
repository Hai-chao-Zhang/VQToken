import pandas as pd
import numpy as np

# vqtoken_res="/mnt/data/TokenDynamics/vq_token/HaichaoZhang/LLaVa-Video/LLaVA-NeXT/outputs/vqtoken_qwen2_0.5b_ov/T20250801_G4d6759c1/bak_20250801174510_LongVideoBench_8frame_subs/vqtoken_qwen2_0.5b_ov_LongVideoBench_8frame_subs.xlsx"
# llavaov_res = "/mnt/data/TokenDynamics/vq_token/HaichaoZhang/LLaVa-Video/LLaVA-NeXT/outputs/llava_onevision_qwen2_0.5b_ov/T20250801_G4d6759c1/llava_onevision_qwen2_0.5b_ov_LongVideoBench_8frame_subs.xlsx"

vqtoken_res="/mnt/data/TokenDynamics/vq_token/HaichaoZhang/LLaVa-Video/LLaVA-NeXT/outputs/vqtoken_qwen2_0.5b_ov/T20250801_G4d6759c1/bak_20250801162221_LongVideoBench_8frame_subs/vqtoken_qwen2_0.5b_ov_LongVideoBench_8frame_subs.xlsx"

# 1. 读入 VLMEvalKit 的 xlsx
df = pd.read_excel(vqtoken_res)



# 2. 把预测 A./B./… 映射到索引
labels = ['A','B','C','D','E']
label2idx = {lab: i for i, lab in enumerate(labels)}
df['pred_idx'] = df['prediction'].str.rstrip('.').map(label2idx)

# 3. 标记是否正确
df['correct'] = df['pred_idx'] == df['correct_choice']

# 4. 定义 duration 划分的边界和标签
#    DURATIONS=[15,60,600,3600] 意味着区间：(0,15]、(15,60]、(60,600]、(600,3600]、(3600, +∞)
bins   = [0, 15, 60, 600, 3600, np.inf]
labels = ['<=15s', '15–60s', '60–600s', '600–3600s', '>3600s']

df['dur_bin'] = pd.cut(df['duration'], bins=bins, labels=labels, right=True)

# 5. 构建 “Duration × Task” 的准确率矩阵
pivot = df.pivot_table(
    index='dur_bin',
    columns='question_category',
    values='correct',
    aggfunc='mean'
).reindex(index=labels)  # 保证顺序

# 6. 输出矩阵（Markdown 格式）
print("\nAccuracy matrix (rows=duration bins, cols=question_category):")
print(pivot.to_markdown(floatfmt=".4f"))

# 7. 额外：只按 duration 统计 overall accuracy
print("\nOverall accuracy by duration bin:")
print(df.groupby('dur_bin')['correct']
        .mean()
        .reindex(labels)
        .rename('accuracy')
        .to_markdown(floatfmt=".4f"))

# 8. 额外：只按 question_category 统计 overall accuracy
task_cats = [
    "S2E","S2O","S2A","E2O","O2E","T2E",
    "T2O","T2A","E3E","O3O","SSS","SOS",
    "SAA","T3E","T3O","TOS","TAA"
]
print("\nOverall accuracy by question_category:")
print(df.groupby('question_category')['correct']
        .mean()
        .reindex(task_cats)
        .rename('accuracy')
        .to_markdown(floatfmt=".4f"))
