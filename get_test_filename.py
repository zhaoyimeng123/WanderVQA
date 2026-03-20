import os

# 设置要读取的目录路径
directory = '/data/dataset/XGC-dataset/test'

# 获取目录下所有文件和子目录的文件名
file_names = []
for root, dirs, files in os.walk(directory):
    for file in files:
        file_names.append(file)

# 对文件名进行排序
file_names.sort()

# 将文件名保存到txt文件中
with open('/data/dataset/XGC-dataset/test.txt', 'w') as f:
    for file_name in file_names:
        f.write(file_name + '\n')

print("文件名已保存到 file_names.txt")
