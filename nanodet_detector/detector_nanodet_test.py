""" Detect and track blobs in image
"""
import os
import cv2
from detector import NanodetDetector


if __name__ == "__main__":
    model_path = "/home/roee/Projects/nanodet/workspace/nanodet-plus-m_320/model_best/nanodet_model_best.pth"
    config_path = "/home/roee/Projects/nanodet/config/nanodet-plus-m_320_lulav_dit.yml"
    image_path = "/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20250625_coco/images/train/20250416_080932_1744790984435034751.jpg"
    img_resize_method = 'crop'  # 'resize'=resize, 'crop'=crop center, None=no resize
    img_resize = (320, 320)  # (w, h)

    # model_path = "/home/roee/Projects/nanodet/workspace/nanodet-plus-m_320/model_best/nanodet_plus_m_320_lulav_dit_model_best.onnx"
    # config_path = "/home/roee/Projects/nanodet/config/nanodet-plus-m_320_lulav_dit.yml"
    # image_path = "/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20250625_coco/images/train/20250416_080932_1744790984435034751.jpg"
    # # image_path = "/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20250625_coco/images/train/20250408_124029_1744116048392439977_aug1.jpg"
    # # image_path = "/home/roee/Projects/datasets/interceptor_drone/deep_learning_uav_detection_dataset/dataset_20250625_coco/images/train/20250408_124029_1744116048441849541.jpg"
    # img_resize_method = 'crop'  # 'resize'=resize, 'crop'=crop center
    # img_resize = (320, 320)  # (w, h)

    # init model
    nnd_detector = NanodetDetector(model_path, config_path)

    # load image
    if not os.path.isfile(image_path):
        raise Exception('image file: {} not found!'.format(image_path))
    raw_img = cv2.imread(image_path)
    assert raw_img is not None, f"Image not found: {image_path}"

    if img_resize_method == 'crop':
        i0 = int((raw_img.shape[0] - img_resize[1]) / 2)
        i1 = int(i0 + img_resize[1])
        j0 = int((raw_img.shape[1] - img_resize[0]) / 2)
        j1 = int(j0 + img_resize[0])
        resized_img = raw_img[i0:i1, j0:j1, :]
    elif img_resize_method == 'resize':
        resized_img = cv2.resize(raw_img, img_resize)
    elif img_resize_method is None:
        resized_img = raw_img
    else:
        raise Exception('invalid resize method')


    # infer
    results = nnd_detector.detect(resized_img)

    # Draw detections
    for res in results:
        xtl, ytl, w, h = res['bbox']
        score = res['confidence']
        class_id = res['class_id']
        # convert back to original image coordinates
        if img_resize_method == 'crop':
            xtl = xtl + j0
            ytl = ytl + i0
        elif img_resize_method == 'resize':
            resized_img = cv2.resize(raw_img, img_resize)
            resize_ratio_x = raw_img.shape[1] / img_resize[0]
            resize_ratio_y = raw_img.shape[0] / img_resize[1]
            xtl = xtl * resize_ratio_x
            w = w * resize_ratio_x
            ytl = ytl * resize_ratio_y
            h = h * resize_ratio_y
        elif img_resize_method is None:
            pass
        else:
            raise Exception('invalid resize method')

        x1, y1, x2, y2 = map(int, [xtl, ytl, xtl + w, ytl + h])  # convert to (xtl,ytl,xbr,ybr)
        cv2.rectangle(raw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_id}: {score:.2f}"
        cv2.putText(raw_img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # for class_id, detections in results.items():
    #     for det in detections:
    #         xtl, ytl, w, h, score = det[:5]
    #         x1, y1, x2, y2 = map(int, [xtl, ytl, xtl+w, ytl+h])  # convert to (xtl,ytl,xbr,ybr)
    #         cv2.rectangle(raw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         label = f"{class_id}: {score:.2f}"
    #         cv2.putText(raw_img, label, (x1, y1 - 5),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Result", raw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
