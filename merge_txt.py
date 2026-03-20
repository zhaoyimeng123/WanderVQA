import glob

# 获取所有 txt 文件路径（确保按顺序读取）
txt_files = sorted(glob.glob("/data/user/zhaoyimeng/ModularBVQA/ckpts_xgc_LSVQ_kvq/output_dim*.txt"))
output_file = "ckpts_xgc_LSVQ_kvq/output.txt"

# 用字典存储合并数据
data = {}

for file in txt_files:
    with open(file, "r") as infile:
        for line in infile:
            columns = line.strip().split(",")  # 按逗号分割
            key, value = columns[0], columns[1]  # 第一列是文件名，第二列是数值
            if key not in data:
                data[key] = []
            data[key].append(value)  # 追加数值

# 写入合并后的数据
with open(output_file, "w") as outfile:
    for key, values in data.items():
        outfile.write(f"{key},{','.join(values)}\n")

print(f"合并完成，保存为 {output_file}")
