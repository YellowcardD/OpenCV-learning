import cv2 as cv
import time

interval = 60 #捕获图像的间隔，单位：秒
num_frames = 2 #捕获图像的总帧数
out_fps = 24 #输出文件的帧率

#打开默认的相机
cap = cv.VideoCapture(0)

#获取捕获的分辨率
size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

#设置要保存视频的编码，分辨率和帧率
video = cv.VideoWriter(
    'time_lapse.avi',
    cv.VideoWriter_fourcc('M', 'P', '4', '2'),
    out_fps,
    size
)

#对于一些低画质的摄像头，前面的帧可能不稳定，略过
for i in range(42):

    cap.read()

#开始捕获，通过read()函数获取捕获的帧
try:
    for i in range(num_frames):

        _, frame = cap.read()
        video.write(frame)
        print('Frame {} is captured.'.format(i))
        time.sleep(interval)
except KeyboardInterrupt:
    print('Stopped! {}/{} frames captured'.format(i, num_frames))

video.release()
cap.release()