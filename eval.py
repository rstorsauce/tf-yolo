import sys, os

sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
import cv2
import numpy as np

classes_name = ["backpack", "person"]
#["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

ckpt = sys.argv[1]
evaldir = sys.argv[2]

def process_predicts(predicts):
  p_classes = predicts[0, :, :, 0:2]
  C = predicts[0, :, :, 2:4]
  coordinate = predicts[0, :, :, 4:]

  p_classes = np.reshape(p_classes, (7, 7, 1, 2))
  C = np.reshape(C, (7, 7, 2, 1))

  P = C * p_classes

  #print P[5,1, 0, :]

  output = []

  #index = np.argmax(P)
  #indeces = np.argpartition(P, -4)[-4:]

  flat = np.ndarray.flatten(P)
  ind = np.argpartition(flat, -10) [-10:]
  print(ind)

  for index in ind:

    if flat[index] > 0.12:


      print(flat[index])
      index = np.unravel_index(index, P.shape)

      #print(P[a, b, c, d])

      class_num = index[3]

      coordinate = np.reshape(coordinate, (7, 7, 2, 4))

      max_coordinate = coordinate[index[0], index[1], index[2], :]

      xcenter = max_coordinate[0]
      ycenter = max_coordinate[1]
      w = max_coordinate[2]
      h = max_coordinate[3]

      xcenter = (index[1] + xcenter) * (448/7.0)
      ycenter = (index[0] + ycenter) * (448/7.0)

      w = w * 448
      h = h * 448

      xmin = xcenter - w/2.0
      ymin = ycenter - h/2.0

      xmax = xmin + w
      ymax = ymin + h

      output.append((xmin, ymin, xmax, ymax, class_num))

  return output

common_params = {'image_size': 448, 'num_classes': 2, 
                'batch_size':1}
net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, 448, 448, 3))
predicts = net.inference(image)

sess = tf.Session()

saver = tf.train.Saver(net.trainable_collection)

saver.restore(sess, ckpt)

for i in os.scandir(evaldir):
  if ".jpg" in i.name:
    print(i.name)
    np_img = cv2.imread(i.path)
    resized_img = cv2.resize(np_img, (448, 448))
    np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)


    np_img = np_img.astype(np.float32)

    np_img = np_img / 255.0 * 2 - 1
    np_img = np.reshape(np_img, (1, 448, 448, 3))


    np_predict = sess.run(predicts, feed_dict={image: np_img})

    output = process_predicts(np_predict)
    for out in output:
      xmin, ymin, xmax, ymax, class_num = out
      print(xmin, ymin, xmax, ymax, class_num)
      class_name = classes_name[class_num]
      cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
      cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
    outpath = str.split(i.path, ".")
    outpath[0] += "_out"
    outimg = ".".join(outpath)
    cv2.imwrite(outimg, resized_img)
sess.close()
