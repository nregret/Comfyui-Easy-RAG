#!/usr/bin/env python3
"""
将03_portrait_examples.md文件按分隔符分块，构建成chunks
"""
import json

def chunk_portrait_examples(input_file, output_file):
    """
    将portrait examples文件按---分隔符分块
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
        
        # 提取标题（如果有）
        lines = chunk.split('\n')
        title = None
        for line in lines:
            if line.startswith('##'):
                title = line.strip()
                break
        
        processed_chunks.append({
            "id": f"chunk_{i+1}",
            "title": title,
            "content": chunk
        })
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"成功生成{len(processed_chunks)}个chunks，保存至{output_file}")

if __name__ == "__main__":
    input_file = "/workspace/rag/Z-Image_2steps/03_portrait_examples.md"
    output_file = "/workspace/rag/Z-Image_2steps/portrait_examples_chunks.json"
    chunk_portrait_examples(input_file, output_file)
