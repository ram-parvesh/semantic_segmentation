import os

# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import numpy as np
import torch
import yaml
from skimage.transform import resize

from skimage import io, transform
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from PIL import Image
from u2net import U2NET  # full size version 173.6 MB


import datetime, time


def normalize(img):
    image = img
    # print(image)

    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
    image = image / np.max(image)
    if image.shape[2] == 1:
        tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
    else:
        tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
        tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
    tmpImg = tmpImg.transpose((2, 0, 1))

    return torch.from_numpy(tmpImg)


def rescale(sample, output_size):
    image = sample

    h, w = image.shape[:2]

    if isinstance(output_size, int):
        if h > w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size

    new_h, new_w = int(new_h), int(new_w)

    img = transform.resize(image, (output_size, output_size), mode='constant')

    return img

def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized

def load_image(fname, mode='RGB', return_orig=False):
    img = cv2.imread(fname)
    img = crop_square(img, 1500)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = np.array(img.convert(mode))
    # print("image shape",img.shape)
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def main(image):
    # --------- 1. get image path and name ---------
    model_name = 'u2net'

    model_dir = 'saved_models/u2net/u2net.pth'

    # resize and normalize to shape for model as input: torch.Size([3, 320, 320])
    x = normalize(rescale(image, 320))
    print("size of image for prediction sent into model",x.shape)
    
    inputs_test = torch.unsqueeze(x, 0)
    print("reshaped to torch.Size([1,3, 320, 320]) ",inputs_test.shape)

    # --------- 3. model define ---------
    if (model_name == 'u2net'):
        net = U2NET(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference image ---------
    inputs_test = inputs_test.type(torch.FloatTensor)
    print(inputs_test.shape)

    if torch.cuda.is_available():
        start_time=datetime.datetime.utcnow()
        inputs_test = Variable(inputs_test.cuda())
        end_time=datetime.datetime.utcnow()
        print("total time for inference in the model : ",end_time-start_time)
    else:
        inputs_test = Variable(inputs_test)

    d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

    # normalization
    pred = d1[:, 0, :, :]
    pred = normPRED(pred)

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')

    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    pb_np_mask = np.array(imo)
    res = np.array(imo)
    del d1, d2, d3, d4, d5, d6, d7
    return res
if __name__== "__main__" :
    start_time=datetime.datetime.utcnow()
    img=cv2.imread("/home/ram/U-2-Net/green.jpg")
    result=main(img)
    cv2.imwrite('sample.png', result)
    end_time=datetime.datetime.utcnow()
    print("total time for inference : ",end_time-start_time)