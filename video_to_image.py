import cv2

def extract_frame(video_path, save_path):
    # 设置为 IMREAD_UNCHANGED 以保持原始格式
    cap = cv2.VideoCapture(video_path)
    
    # 设置原始分辨率
    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_height)
    
    ret, frame = cap.read()
    if ret:
        # 直接保存原始帧，不做任何处理
        cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    cap.release()

# 使用示例
video_path = "your_video.mp4"
save_path = "output.jpg"
extract_frame(video_path, save_path) 