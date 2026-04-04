#!/usr/bin/env python3
"""
将03_portrait_examples.md文件按分隔符分块，构建成chunks，分离metadata和content
"""
import json
import re

def chunk_portrait_examples(input_file, output_file):
    """
    将portrait examples文件按---分隔符分块，分离metadata和content
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按---分隔符分块
    chunks = content.split('---')
    
    # 过滤空块并处理每个块
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        
        lines = chunk.split('\n')
        
        # 提取标题
        title = None
        tags = []
        likes = 0
        content_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 提取标题
            if line.startswith('##'):
                # 提取标题文本，去掉编号
                title_match = re.search(r'## \[\d+\] (.*)', line)
                if title_match:
                    title = title_match.group(1)
                else:
                    title = line.strip('# ')
            # 提取维度标签
            elif line.startswith('维度标签：'):
                # 提取所有标签
                tags_str = line.replace('维度标签：', '')
                # 匹配所有[维度:值1,值2]格式的标签
                tag_matches = re.findall(r'\[(.*?)\]', tags_str)
                for tag_match in tag_matches:
                    if ':' in tag_match:
                        dimension, values = tag_match.split(':', 1)
                        # 将多个值拆分
                        value_list = [v.strip() for v in values.split(',') if v.strip()]
                        # 对第一个值添加维度前缀，后续值直接添加
                        for j, value in enumerate(value_list):
                            if j == 0:
                                tags.append(f"{dimension}:{value}")
                            else:
                                tags.append(value)
                    else:
                        # 没有维度的标签直接添加
                        if tag_match.strip():
                            tags.append(tag_match.strip())
            # 提取收藏数
            elif line.startswith('收藏数：'):
                likes_str = line.replace('收藏数：', '')
                try:
                    likes = int(likes_str.strip())
                except:
                    likes = 0
            # 其他行作为content
            else:
                content_lines.append(line)
        
        # 构建content
        content = '\n'.join(content_lines).strip()
        
        # 跳过没有内容的块
        if not content:
            continue
        
        processed_chunks.append({
            "id": f"chunk_{i+1}",
            "metadata": {
                "title": title,
                "tags": tags,
                "likes": likes
            },
            "content": content
        })
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"成功生成{len(processed_chunks)}个chunks，保存至{output_file}")

if __name__ == "__main__":
    input_file = "/workspace/rag/Z-Image_2steps/03_portrait_examples.md"
    output_file = "/workspace/rag/Z-Image_2steps/portrait_examples_chunks_v4.json"
    chunk_portrait_examples(input_file, output_file)
