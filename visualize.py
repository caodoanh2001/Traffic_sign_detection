import json
from tqdm import tqdm
import os
import cv2
import numpy as np
import argparse

def visualize(test_dir, outdir):
    list_img = os.listdir(test_dir)
    outdir = outdir + '/'

    with open('./result_image/submission.json') as json_file:
        data_predict = json.load(json_file)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    dict_bbox_predict = {}
    for image in data_predict['annotations']:
        if image['image_id'] in dict_bbox_predict:
          dict_bbox_predict[image['image_id']].append([image['bbox'], image['category_id']])
        else:
          dict_bbox_predict[image['image_id']] = [[image['bbox'], image['category_id']]]

    for i in tqdm(range(0, len(list_img)-1)):
        index = int(os.path.splitext(list_img[i])[0])
        img = cv2.imread(test_dir + '/' + list_img[i])
        
        if index in dict_bbox_predict:
          for bbox in dict_bbox_predict[index]:
            x, y, w, h = bbox[0]
            cv2.rectangle(img, (int(x),int(y)), (int(x+w),int(y+h)), (0, 255, 0), 2)
            cv2.putText(img, str(bbox[1]), (int(x), int(y-10)), 0, 0.75, (0, 255, 0), 2)

        cv2.imwrite(outdir + list_img[i], img)

def get_args():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--test_dir', type=str, default='./data',
                        help='Train dir', dest='test_dir')
    parser.add_argument('--outdir', type=str, default='./visualize',
                        help='Json dir', dest='outdir')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    visualize(args.test_dir, args.outdir)
