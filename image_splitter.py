import cv2
import numpy as np
import os

def split_image(image_path, output_dir, overlap_ratio=0.6):
    """
    将图片分解成多个重叠的子图片
    
    Args:
        image_path: 输入图片路径
        output_dir: 输出目录
        overlap_ratio: 重叠比例（0-1之间）
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    height, width = img.shape[:2]
    
    # 计算子图片的大小
    # 为了保持60%的重叠，我们需要确保每个子图片的大小合适
    # 这里我们选择图片宽度的40%作为子图片的宽度
    sub_width = int(width * 0.4)
    sub_height = int(height * 0.4)
    
    # 计算步长（重叠部分）
    step_x = int(sub_width * (1 - overlap_ratio))
    step_y = int(sub_height * (1 - overlap_ratio))
    
    # 生成子图片
    count = 0
    for y in range(0, height - sub_height + 1, step_y):
        for x in range(0, width - sub_width + 1, step_x):
            # 提取子图片
            sub_img = img[y:y + sub_height, x:x + sub_width]
            
            # 保存子图片
            output_path = os.path.join(output_dir, f'sub_image_{count}.jpg')
            cv2.imwrite(output_path, sub_img)
            count += 1
    
    return count

def main():
    # 直接设置输入输出路径
    image_path = "image.png"
    output_dir = "images"
    
    try:
        # 处理图片
        count = split_image(image_path, output_dir)
        print(f"成功将图片分解成 {count} 个子图片")
        print(f"子图片保存在: {output_dir}")
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main() 