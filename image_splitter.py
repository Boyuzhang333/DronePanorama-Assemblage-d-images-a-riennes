import cv2
import numpy as np
import os

def find_next_test_dir(base_path="images"):
    """查找下一个可用的测试目录编号"""
    i = 1
    while True:
        test_dir = os.path.join(base_path, f"imageTest{i}")
        if not os.path.exists(test_dir):
            return test_dir
        i += 1

def split_image(image_path, overlap_ratio=0.6):
    """
    将图片分解成多个重叠的子图片，避免重复
    
    Args:
        image_path: 输入图片路径
        overlap_ratio: 重叠比例（0-1之间）
    """
    # 自动创建下一个可用的测试目录
    output_dir = find_next_test_dir()
    print(f"将使用输出目录: {output_dir}")
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    height, width = img.shape[:2]
    print(f"原始图片尺寸: {width}x{height}")
    
    # 计算子图片的大小
    sub_width = int(width * 0.4)  # 子图片宽度为原图的40%
    sub_height = int(height * 0.4)  # 子图片高度为原图的40%
    
    # 计算步长（非重叠部分）
    step_x = int(sub_width * (1 - overlap_ratio))
    step_y = int(sub_height * (1 - overlap_ratio))
    
    print(f"子图片尺寸: {sub_width}x{sub_height}")
    print(f"步长: x={step_x}, y={step_y}")
    print(f"重叠比例: {overlap_ratio*100}%")
    
    # 生成子图片
    count = 0
    y = 0
    while y < height:
        x = 0
        while x < width:
            # 如果这是最后一列，调整x的位置以确保完整覆盖
            if x + sub_width > width:
                x = width - sub_width
            
            # 如果这是最后一行，调整y的位置以确保完整覆盖
            if y + sub_height > height:
                y = height - sub_height
            
            # 提取子图片
            sub_img = img[y:y + sub_height, x:x + sub_width]
            
            # 保存子图片
            output_path = os.path.join(output_dir, f'sub_image_{count}.jpg')
            cv2.imwrite(output_path, sub_img)
            print(f"已保存: {output_path} (位置: [{x}:{x+sub_width}, {y}:{y+sub_height}])")
            count += 1
            
            # 如果已经处理到最后一列，跳出内循环
            if x == width - sub_width:
                break
            
            x += step_x
        
        # 如果已经处理到最后一行，跳出外循环
        if y == height - sub_height:
            break
        
        y += step_y
    
    print(f"\n总共生成了 {count} 个子图片")
    return output_dir, count

def main():
    # 获取用户输入
    image_path = input("请输入要分割的图片路径: ")
    overlap_ratio = float(input("请输入重叠比例 (0-1之间，默认0.6): ") or "0.6")
    
    try:
        output_dir, count = split_image(image_path, overlap_ratio)
        print(f"\n成功将图片分解成 {count} 个子图片")
        print(f"子图片保存在: {output_dir}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main() 