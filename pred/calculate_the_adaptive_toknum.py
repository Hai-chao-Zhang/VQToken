import re

# 文件路径
path = '/mnt/data/TokenDynamics/vq_token/HaichaoZhang/LLaVa-Video/LLaVA-NeXT/trlogs_new/evallog_adp_abl.txt'

# 读取文件内容
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# 提取所有 "📈 Total Visual Tokens After Compression:" 后的数字
values = [int(v) for v in re.findall(r'📈 Total Visual Tokens After Compression:\s*(\d+)', text)]

# 计算平均值
average = sum(values) / len(values) if values else float('nan')

# 输出结果
print(f"提取到的值: {values}")
print(f"平均视觉压缩后 Token 数量: {average:.2f}")
