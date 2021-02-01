import torch
from PIL import Image
import  numpy as np
# # Model
# #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# path_or_model='yolov5s_voc_best.pt'
# # Images
# img1 = Image.open('data/images/zidane.jpg')
# img2 = Image.open('data/images/bus.jpg')
# imgs = [img1, img2]  # batched list of images
#
# # Inference
# result = model(imgs)
#
# print("ok")

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='yolov5s.pt')
model = model.autoshape()  # for PIL/cv2/np inputs and NMS
# Images
img1 = Image.open('data/images/zidane.jpg')
img2 = Image.open('data/images/bus.jpg')
imgs = [img1, img2]  # batched list of images

# Inference
result = model(imgs, size=640)  # includes NMS
result.print()


from numpy import random
import cv2

# 获取类别名字
names = model.module.names if hasattr(model, 'module') else model.names
# 设置画框的颜色
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

# 对每一张图片作处理
for i, det in enumerate(result.xyxy):  # detections per image


        im0 = imgs[i]

        im0 = cv2.cvtColor(np.asarray(im0), cv2.COLOR_RGB2BGR)
        # 保存预测结果
        for *xyxy, conf, cls in det:

            # 在原图上画框
            label = '%s %.2f' % (names[int(cls)], conf)
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        cv2.imshow("my{}".format(i),im0)


cv2.waitKey(0)


