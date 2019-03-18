import tensorflow as tf

from keras import backend as K
from keras.models import Model, load_model
from TextImageGenerator import TextImageGenerator
import Utils
import detectNumberPlace
import delfCars
import cv2
sess = tf.Session()
K.set_session(sess)

#todo save letters in db
##############################
c_val = Utils.get_counter('lp1/train/', 'val')
c_train = Utils.get_counter('lp1/train/', 'train')
letters_train = set(c_train.keys())
letters_val = set(c_val.keys())
if letters_train == letters_val:
    print('Letters in train and val do match')
else:
    raise Exception()
# print(len(letters_train), len(letters_val), len(letters_val | letters_train))
letters = sorted(list(letters_train))
print('Letters:', ' '.join(letters))
############################

model = load_model('lp1/model1.h5', custom_objects={'<lambda>': lambda y1, y2: y2})
model.summary()
#img = Utils.build_data("lp1/test1/img/T305XK54.png", 128, 64)
#img0 = detectNumberPlace.detectedLicensePlates("vesta.png")
#cv2.imwrite("temp/tempCar.png",img0)
img = Utils.build_data("temp/tempCar.png", 128, 64)
net_inp = model.get_layer(name='the_input').input
net_out = model.get_layer(name='softmax').output

for inp_value, _ in Utils.next_batch(1, 128, 64, 8, letters, 4, img):
    bs = inp_value['the_input'].shape[0]
    X_data = inp_value['the_input']
    net_out_value = sess.run(net_out, feed_dict={net_inp: X_data})
    pred_texts = Utils.decode_batch(net_out_value, letters)
    print('Predicted: %s' % (pred_texts[0]))
    labels = inp_value['the_labels']
    texts = []
   #
    break

#delfCars.findImage('vesta.png')