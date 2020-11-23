import json
import argparse
import os
import numpy as np
import pickle
from skimage import color
from skimage.feature import hog
from sklearn import svm
import cv2

def feature_hog (img):
  dim = (48, 48)
  ppc = 8
  img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  fd, hog_image = hog(img, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(2, 2),block_norm= 'L2',visualize=True)
  return fd

def export(detect_dir, output):
    test_dir = './data'
    list_npy = os.listdir(detect_dir)
    dir_npy = detect_dir
    threshold = 0.3

    #Xuat file json chua ten cac image trong thu muc data
    test_dirs = os.listdir('./data')
    test_dir_dict = {}
    test_dir_list = []
    for i in test_dirs:
      i_dict = {}
      i_dict["file_name"] = i
      i_dict["id"] = int(os.path.splitext(i)[0])
      test_dir_list.append(i_dict)
    test_dir_dict["images"] = test_dir_list

    with open('./test_directory.json', 'w') as test_dir_json:
      test_dir_json.write(str(test_dir_dict).replace("'",'"'))


    #In detection ra dict
    all_detection = []
    for npy in list_npy:
      image_id = os.path.splitext(npy)[0]
      detection = np.load(dir_npy + npy, allow_pickle=True)
      for detect in detection:
        detect_dict = {}
        detect_dict["image_id"] = int(image_id)
        detect_dict["category_id"] = detect[2] + 1
        detect_dict["bbox"] = [detect[0][0], detect[0][1], detect[0][2] - detect[0][0], detect[0][3] - detect[0][1]]
        detect_dict["score"] = detect[1]
        all_detection.append(detect_dict)

    export_json = str(all_detection).replace("'", '"')

    with open(output + 'submission.json' , 'w') as out:
      out.write(export_json)

    with open('./test_directory.json') as json_file:
      test_image = json.load(json_file)
  
    out_visualize_val = {}
    out_visualize_val["annotations"] = all_detection
    out_visualize_val["images"] = test_image["images"]

    with open('./result_image/submission.json', "w") as out:
      out.write(str(out_visualize_val).replace("'",'"'))

    pkl_filename = './model/classify/classfier.pkl'

    with open(pkl_filename, 'rb') as file:
      pickle_model = pickle.load(file)
    
    with open('./result_image/submission.json') as json_file:
      data_predict = json.load(json_file)

    for detection in data_predict["annotations"]:
      image_file = str(detection["image_id"]) + '.png'
      x, y, w, h = detection["bbox"]
      score =  detection["score"]
      category = detection["category_id"]
      if category == 3 or category == 4 or category == 5:
          img = cv2.imread(test_dir + '/' + image_file)
          traffic_sign = img[int(y) : int(y + h) , int(x) : int(x + w), :]
          hog_im = feature_hog(traffic_sign)
          new_score = max(pickle_model.predict_proba(hog_im.reshape(1,-1))[0])
          new_category = pickle_model.predict(hog_im.reshape(1,-1))[0]
          if new_score > threshold:
            detection["score"] = new_score
            detection["category_id"] = int(new_category)

    export_json = str(data_predict["annotations"]).replace("'", '"')

    with open('result/submission.json', "w") as out:
        out.write(export_json)


def get_args():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--detect_dir', type=str,
                        default='./detect/',
                        help='detect', dest='detect_dir')
    parser.add_argument('--output', type=str,
                        default='result/',)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    export(args.detect_dir, args.output)
