import csv
import random

# 读取槽位信息文件
def read_slot_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        slots = [line.strip() for line in file.readlines()]
    return slots

# 读取句子模板文件
def read_template_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        templates = [row['sentence'] for row in reader]
    return templates

# 合并槽位信息成一个集合
def merge_slots(slot_files):
    merged_slots = set()
    for file_name in slot_files:
        slots = read_slot_file(file_name)
        merged_slots.update(slots)
    
    # 随机抽样10万条数据
    if len(merged_slots) > 100000:
        merged_slots = set(random.sample(list(merged_slots), 100000))
    
    return merged_slots

# 填充模板文本的槽位
def fill_template(template, slot):
    filled_template = template.replace('{{company}}', slot)
    return filled_template

# 将填充后的文本转换为序列标注的格式
def convert_to_sequence_labeling(text, slot):
    labeled_text = []
    start_index = 0
    
    while True:
        index = text.find(slot, start_index)
        if index == -1:
            break
        
        prefix = text[start_index:index]
        if prefix:
            labeled_text.extend([char + "\tO" for char in prefix])
        
        labeled_text.append(slot[0] + "\tB-ORG")
        labeled_text.extend([char + "\tI-ORG" for char in slot[1:]])
        
        start_index = index + len(slot)
    
    remaining_text = text[start_index:]
    if remaining_text:
        labeled_text.extend([char + "\tO" for char in remaining_text])
    
    return labeled_text

# 主函数
def main():
    # 合并槽位信息为一个集合
    merged_slots = list(merge_slots(['./slot/company_name.txt', './slot/company_short_name.txt', './slot/org_name.txt']))
    
    # 读取句子模板文件
    templates = read_template_file('template.csv')
    
    # 设置采样大小和分割率
    sample_size = 500000
    split_rate = 0.8
    
    # 转换为序列标注的格式
    labeled_data = []
    
    for i in range(sample_size):
        template = random.choice(templates)
        if "{{company}}" in template: 
            slot = random.choice(merged_slots)
            sentence = fill_template(template, slot)
            labeled_text = convert_to_sequence_labeling(sentence, slot)
            labeled_data.append(labeled_text)

    # 随机打乱数据
    random.shuffle(labeled_data)
    
    # 计算切割点
    train_size = int(len(labeled_data) * split_rate)
    dev_size = (len(labeled_data) - train_size) // 2
    
    # 分割数据集
    train_data = labeled_data[:train_size]
    dev_data = labeled_data[train_size:train_size + dev_size]
    test_data = labeled_data[train_size + dev_size:]
    
    # 写入文件
    def write_to_file(data, file_name):
        with open(file_name, 'w', encoding='utf-8') as file:
            for item in data:
                file.write('\n'.join(item) + '\n\n')
    
    write_to_file(train_data, './dataset/train.char.bio.tsv')
    write_to_file(dev_data, './dataset/dev.char.bio.tsv')
    write_to_file(test_data, './dataset/test.char.bio.tsv')

if __name__ == "__main__":
    main()
