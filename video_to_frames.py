import cv2
import os
from datetime import datetime

def find_next_imagetest_folder():
    """
    查找下一个可用的imageTest文件夹
    例如：如果存在imageTest10，则返回imageTest11
    """
    images_dir = os.path.join(os.getcwd(), 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        return os.path.join('images', 'imageTest1')
    
    test_folders = [d for d in os.listdir(images_dir) if d.startswith('imageTest') and d[9:].isdigit()]
    if not test_folders:
        return os.path.join('images', 'imageTest1')
    
    max_num = max(int(folder[9:]) for folder in test_folders)
    return os.path.join('images', f'imageTest{max_num + 1}')

def extract_frames_from_video(video_path, output_dir=None, fps=2):
    """
    从视频中提取帧并保存
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录，如果为None则自动创建新的imageTest文件夹
        fps: 每秒提取的帧数（默认2，即每0.5秒一帧）
    """
    # 确定输出目录
    if output_dir is None:
        output_dir = find_next_imagetest_folder()
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # 获取视频信息
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    print(f"Video Info:")
    print(f"FPS: {video_fps}")
    print(f"Total Frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Extracting one frame every {1/fps:.1f} seconds")
    
    # 计算帧间隔
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 每0.5秒保存一帧
        if frame_count % frame_interval == 0:
            # 生成文件名（使用时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{timestamp}_{saved_count:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # 保存图像
            cv2.imwrite(filepath, frame)
            print(f"Saved: {filename}")
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"\nExtraction complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames saved: {saved_count}")
    print(f"Frames saved in: {output_dir}")

if __name__ == "__main__":
    # 获取用户输入的视频路径
    video_path = input("请输入视频文件路径: ")
    extract_frames_from_video(video_path)  # 使用默认的fps=2，即每0.5秒一帧 