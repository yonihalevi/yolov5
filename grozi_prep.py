import os
import shutil

import numpy as np
import cv2
from sklearn.model_selection import train_test_split

import pyexcel_ods3 as ods
from google_drive_downloader import GoogleDriveDownloader as gdd
from zipfile import ZipFile

grozi_drive_id = '1Fx9lvmjthe3aOqjvKc6MJpMuLF22I1Hp'
grozi_path = '../'
grozi_dir = 'grozi'
grozi_ziptmp = 'grozi.zip'
resized_dir = grozi_path + grozi_dir + "/src/408/"
original_dir = grozi_path + grozi_dir + "/src/3264/"

def fetch_dataset():
    gdd.download_file_from_google_drive(grozi_drive_id, grozi_path + grozi_ziptmp, unzip=False)
    if os.path.isdir(grozi_path + grozi_dir):
        shutil.rmtree(grozi_path + grozi_dir)
    zf = ZipFile(grozi_path + grozi_ziptmp)
    zf.extractall("../")
    zf.close()


def resize_images():
    if os.path.isdir(resized_dir):
        shutil.rmtree(resized_dir)

    os.mkdir(resized_dir)

    original_images = [original_dir + img for img in os.listdir(original_dir) if img[-3:] == 'jpg']

    for oi in original_images:
        image = cv2.imread(oi)
        dim = (int(image.shape[1]/8), int(image.shape[0]/8))
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(oi.replace("3264","408",1),resized)

def create_annotation_txt():
    data = ods.read_data(grozi_path + grozi_dir + "/classes/grozi.ods")
    sheet = data['grozi']
    bboxes = dict()

    # format of ods file
    # sheet
    #   gtbboxid,classid,imageid,lx,rx,ty,by,difficult,split
    tags = ['gtbboxid', 'classid', 'imageid', 'lx', 'rx', 'ty', 'by']

    for bbox in sheet[1:]:

        if len(bbox) != 9:
            continue

        bboxes[bbox[0]] = dict()
        b = bboxes[bbox[0]]

        for i in range(len(tags)):
            b[tags[i]] = bbox[i]

        assert b['classid'] < 1223

        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b['box_center_x'] = (b['lx'] + b['rx']) / 2
        b['box_center_y'] = (b['ty'] + b['by']) / 2
        b['box_width'] = (b['rx'] - b['lx'])
        b['box_height'] = (b['by'] - b['ty'])

        #print(bboxes[bbox[0]])

    images = [resized_dir + img for img in os.listdir(resized_dir) if img[-3:] == 'jpg']

    for img in images:
        print_buffer = []
        for b in bboxes.values():
            if b['imageid'] == int(img[img.rfind('/') + 1:-4]):
                print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(b['classid'], b['box_center_x'],
                                                                            b['box_center_y'], b['box_width'],
                                                                            b['box_height']))
        save_file = img.replace("jpg", "txt")
        print("\n".join(print_buffer), file=open(save_file, "w"))


def show_annotation(imageid):
    annotation_path = resized_dir + str(imageid) + '.txt'
    image_path = annotation_path.replace("txt", "jpg")

    # load the annotation file into an numpy array
    afile = open(annotation_path, "r")

    # load into an array of floats using comprehensions
    annotations = [x.split(" ") for x in afile.read().split("\n")[:-1]]
    annotations = [[float(y) for y in x] for x in annotations]

    # load image
    img = cv2.imread(image_path)
    w = img.shape[1]
    h = img.shape[0]

    # now we convert annotations to a np array an manipulat eit from cx cy w h to x0 y0 x1 y1 bounding box
    np_annotations = np.array(annotations)
    scaled_annotations = np.copy(np_annotations)
    scaled_annotations[:, [1, 3]] = np_annotations[:, [1, 3]] * w
    scaled_annotations[:, [2, 4]] = np_annotations[:, [2, 4]] * h
    scaled_annotations[:, 1] = scaled_annotations[:, 1] - scaled_annotations[:, 3] / 2
    scaled_annotations[:, 2] = scaled_annotations[:, 2] - scaled_annotations[:, 4] / 2
    scaled_annotations[:, 3] = scaled_annotations[:, 1] + scaled_annotations[:, 3]
    scaled_annotations[:, 4] = scaled_annotations[:, 2] + scaled_annotations[:, 4]

    with open('classes_list.txt') as f:
        classes = [line[line.find("-")+1:].strip() for line in f]

    for ann in scaled_annotations:
        cls, left, top, right, bottom = (round(x) for x in ann)

        label_size, base_line = cv2.getTextSize(classes[cls], cv2.FONT_HERSHEY_SIMPLEX, 0.25 , 1)

        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), cv2.LINE_4)
        cv2.rectangle(img, (left, top - round(1.5 * label_size[1])),
                           (left + round(1.5 * label_size[0]), top + base_line),
                           (255, 0, 0), cv2.FILLED
                      )

        cv2.putText(img, classes[cls], (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), thickness=1)

    cv2.imshow(image_path, img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


def create_segmentation_of_data():

    images = [resized_dir + img for img in os.listdir(resized_dir)if img[-3:] == 'jpg']

    annotations = [x.replace('jpg', 'txt') for x in images]

    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size= 0.2, random_state=1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size= 0.5, random_state=1)

    path = grozi_path + grozi_dir
    os.mkdir(path +'/images')
    os.mkdir(path +'/labels')

    moves = {'trn_im' : (train_images, '/images/train'),
             'vld_im' : (val_images, '/images/val'),
             'tst_im' : (test_images, '/images/test'),
             'trn_lbl' : (train_annotations, '/labels/train'),
             'vld_lbl' : (val_annotations, '/labels/val'),
             'tst_lbl' : (test_annotations, '/labels/test'),
             }

    for m in moves.values():

        os.mkdir(path + m[1])

        for f in m[0]:
            shutil.copy(f, path + m[1])

def etl():
    fetch_dataset()
    resize_images()
    create_annotation_txt()
    create_segmentation_of_data()

if __name__ == "__main__":
    etl()
