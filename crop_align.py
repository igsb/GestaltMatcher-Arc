## crop_align.py
# Run the cropper model on every image in the desired directory
# followed by aligning them based on some landmarks
# and lastly saving the aligned image(s) in the designated directory

import argparse
from glob import glob

import cv2
import os
import random
import time

import numpy as np
import torch
from lib.models.retinaface import RetinaFace
import torch.backends.cudnn as cudnn
from skimage import transform as trans

from lib.utils.prior_box import PriorBox
from lib.utils.box_utils import decode, decode_landm
from lib.utils.py_cpu_nms import py_cpu_nms


# cropper / align
cfg = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description='Crop and align faces')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')

    parser.add_argument('-m', '--cropper_model', default='./saved_models/Resnet50_Final.pth',
                        type=str, help='Trained state_dict file path to open')

    parser.add_argument('--use_subdirectories', action="store_true", default=False,
                        help='When set saves images in dirs in the "images_dir" (e.g. for CASIA)')

    parser.add_argument('--single_align', action="store_true", default=False,
                        help='Originally we corrected rotation once before alignment and once during, setting this '
                             'flag aligns just once.')

    parser.add_argument('--data', default=['data/cases'], dest='data', nargs='+',
                        help='Path to the data directory containing the images or single image to run the model on.')
    parser.add_argument('--save_dir', default='data/cases_align', dest='save_dir',
                        help='Path to the data directory containing the images to run the model on, single file not supported.')

    return parser.parse_args()
args = parse_args()


def face_align_crop(net, img_paths, device):
    save_dir = args.save_dir
    # img_paths = [y for x in os.walk(data) for y in glob(os.path.join(x[0], '*.*'))]
    aligned_imgs = []
    skipped_imgs = []
    img_names = []

    print(f"Cropping and aligning ~{len(img_paths)} ...")
    for img_path in img_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_names.append(img_name)

        if args.single_align:
            img, coords = detect(net, img_path, img_name, device, save_dir=save_dir)
        else:
            img_corrected = detect(net, img_path, img_name, device, save_dir=save_dir, first=True)
            img, coords = detect(net, img_corrected, img_name, device, save_dir=save_dir)

        # if we found a face in the image, then align it
        if len(coords) > 0:
            aligned_img = align(img, coords, img_name, save_dir=save_dir)
            aligned_imgs.append(aligned_img)
        else:
            skipped_imgs.append(img_name)
            continue

    print(f"{'All images were successfully cropped and aligned.' if len(skipped_imgs) == 0 else f'Cropper was unsuccessful for the following images: {skipped_imgs}'}")
    return aligned_imgs, img_names


def detect(net, img_path, img_name, device, save_dir='', first=False):
    resize = 1
    subdir_name = ''
    if type(img_path) == str:
        if args.use_subdirectories:
            subdir_name = os.path.split(os.path.split(img_path)[-2])[-1]
            save_dir = os.path.join(save_dir, subdir_name)
        os.makedirs(save_dir, exist_ok=True)

        ## *.gif format is not supported by cv.imread(..)
        if os.path.splitext(img_path) == ".gif":
            cap = cv2.VideoCapture(img_path)
            ret, img_raw_original = cap.read()
            cap.release()
        else:
            img_raw_original = cv2.imread(img_path)
    else:  # in case we use an image directly
        img_raw_original = img_path

    def resize_square_aspect_cv2(img, desired_size=640):
        old_size = img.shape[0:2]  # (width, height)

        # we crop without resize if desired_size == 0
        if desired_size == 0:
            desired_size = max(old_size)

            # Too large images can cause an OOM-error, hopefully this addresses that...
            if (old_size[0] * old_size[1]) > 10000000:
                desired_size = 2000  # should be large enough

        ratio = float(desired_size) / max(old_size)
        new_size = [int(x * ratio) for x in old_size]
        new_size = tuple(new_size[::-1])

        new_img = cv2.resize(img, new_size)

        return new_img

    if img_path == (None, []):
        print(f"Error averted at {img_name}, skipping.")
        return None, []

    original_size = img_raw_original.shape[0:2]
    img_raw = resize_square_aspect_cv2(img_raw_original, 640)  # Note: some images are too big resulting in an OOM-error

    img_np = np.float32(img_raw)

    im_height, im_width, _ = img_np.shape
    scale = torch.Tensor([img_np.shape[1], img_np.shape[0], img_np.shape[1], img_np.shape[0]])
    img = img_np - (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float().to(device)
    scale = scale.to(device)

    tic = time.time()
    with torch.no_grad():
        loc, conf, landms = net(img)  # forward pass
    # print('net forward time on {}: {:.4f}'.format(img_path, time.time() - tic))

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    # magic value: 0.02 = confidence threshold of cropper
    inds = np.where(scores > 0.02)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    # magic value: 5000 = top_k to keep
    order = scores.argsort()[::-1][:5000]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    # magic value: 0.4 = NMS threshold
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.4)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    # magic value: 750 = keep_top_k
    dets = dets[:750, :]
    landms = landms[:750, :]

    dets = np.concatenate((dets, landms), axis=1)
    # Skip if we don't find any face
    if len(dets) == 0:
        print(f"Did not find any face in image {img_name}.. skipping.")
        return None, []

    # Own addition to keep only the single most confident detection (we want single faces)
    dets = [dets[dets[:, 4].argmax()]]

    for idx, b in enumerate(dets):
        # scale b back to original input size
        b_scaled = b * float(max(original_size)) / max(img_raw.shape[0:2])

        if first:
            def rotate_image(image, landmarks, fill_color=0.):
                # define origin
                origin = landmarks[[5, 6]]
                middle_finger = landmarks[[7, 8]]

                nose = landmarks[[9, 10]]

                # calc angle to rotate
                orientation_vector = middle_finger - origin
                destination_vector = np.array([1., 0.])
                dir_unit_vector = orientation_vector / np.linalg.norm(orientation_vector)
                angle_rad = np.arccos(np.clip(np.dot(destination_vector, dir_unit_vector), -1.0, 1.0))
                angle_degrees = -180 / np.pi * angle_rad
                angle_degrees = angle_degrees if (orientation_vector[1] < 0) else -angle_degrees

                # rot_mat = cv2.getRotationMatrix2D(tuple(np.array(image.shape[1::-1]) / 2), angle_degrees, 1.0)
                rot_mat = cv2.getRotationMatrix2D(tuple(nose), angle_degrees, 1.0)

                # Background fill color is set to 0.5 to 0 when centered around [-1,1]
                fill = int(fill_color * 255)
                result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                                        borderValue=(fill, fill, fill))

                # plt.imshow(result)
                # plt.show()

                return result, angle_degrees

            img_rot, rotation_angle_first = rotate_image(img_raw, b)
            img_original_rot, _ = rotate_image(img_raw_original, b_scaled)
            return img_original_rot

        d = list(map(int, b_scaled))
        # TODO: get relevant coords -> [0:4] and [5::] ?
        coords = d[0:4] + d[5::]

    return img_raw_original, coords[4::]


def align(img, coords, img_name, save_img=True, save_dir=''):
    def estimate_norm(lmk, image_size=112):
        # lmk is prediction; src is template
        assert lmk.shape == (5, 2)
        tform = trans.SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = float('inf')

        # Account for the standard image size of ArcFace
        if image_size == 112:
            src = arcface_src
        else:
            src = float(image_size) / 112 * arcface_src

        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
            M = tform.params[0:2, :]
            results = np.dot(M, lmk_tran.T)
            results = results.T
            error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
            if error < min_error:
                min_error = error
                min_M = M
                min_index = i

        return min_M, min_index
    def norm_crop(img, landmark, image_size=112):
        M, pose_index = estimate_norm(landmark, image_size)
        warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        return warped

    coords = np.array(coords)
    coords = coords.reshape((5, 2)).astype(float)

    # norm crop
    aligned_img = norm_crop(img, coords)

    # save image
    if save_img:
        cv2.imwrite(os.path.join(save_dir, f"{img_name}_aligned.jpg"), aligned_img)

    return aligned_img


def main():
    # Training/cuda settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)

    # cropper data:
    img_paths = []
    if len(args.data) == 1:
        print(f"Single data source found.")
        if os.path.exists(args.data[0]):
            if os.path.isfile(args.data[0]):
                # Data dir is a single file (hopefully image...)
                img_paths = args.data
            else:
                # Data dir is a dir
                img_paths = [y for x in os.walk(args.data[0]) for y in glob(os.path.join(x[0], '*.*'))]
    else:
        print(f"{len(args.data)} data sources found.")
        for d in args.data:
            if os.path.exists(d):
                if os.path.isfile(d):
                    img_paths.append(d)
                else:
                    print("--data can only contain paths or a single directory, not a mix of both.")
                    exit()
            else:
                print(f"file/directory ({d}) in --data does not exist")


    #print(f"{img_paths=}")

    # Load model
    def load_model(model, pretrained_path, load_to_cpu):
        def remove_prefix(state_dict, prefix):
            ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
            print('\tremove prefix \'{}\''.format(prefix))
            f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
            return {f(key): value for key, value in state_dict.items()}
        def check_keys(model, pretrained_state_dict):
            ckpt_keys = set(pretrained_state_dict.keys())
            model_keys = set(model.state_dict().keys())
            used_pretrained_keys = model_keys & ckpt_keys
            unused_pretrained_keys = ckpt_keys - model_keys
            missing_keys = model_keys - ckpt_keys
            print('\tMissing keys:{}'.format(len(missing_keys)))
            print('\tUnused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
            print('\tUsed keys:{}'.format(len(used_pretrained_keys)))
            assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
            return True

        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = remove_prefix(pretrained_dict, 'module.')
        check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.cropper_model, args.no_cuda)
    net.eval()
    print('Finished loading model!')

    cudnn.benchmark = True
    device = torch.device("cpu" if args.no_cuda else "cuda")
    net = net.to(device)

    tic = time.time()
    aligned_imgs, img_names = face_align_crop(net, img_paths, device)
    toc = time.time()
    aligned_img_paths = [y for x in os.walk(args.save_dir) for y in glob(os.path.join(x[0], '*.*'))]
    #print(f"{aligned_img_paths=}")

    if len(img_paths) > 0:
        print(f"Cropping and aligning took {(toc-tic):.2f}s{'.' if len(img_paths) == 1 else f' (~{((toc-tic)/len(img_paths)):.2f}s per image).'}")
    else:
        print("No images were found at the given location.")

if __name__ == '__main__':
    main()
