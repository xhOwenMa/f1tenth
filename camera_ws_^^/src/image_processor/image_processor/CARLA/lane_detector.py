from .camera_geometry import CameraGeometry
import numpy as np
import albumentations as albu
import cv2
import torch
import segmentation_models_pytorch as smp
from lane_detection.model.lanenet.LaneNet import LaneNet


class LaneDetector():
    def __init__(self, cam_geom=CameraGeometry(image_width=1280, image_height=720),
                 model_path='./lane_segmentation_model.pth',
                 encoder='efficientnet-b0', encoder_weights='imagenet'):
        self.cg = cam_geom
        self.cut_v, self.grid = self.cg.precompute_grid()
        self.model = LaneNet()  # Instantiate the model architecture
        # state_dict = torch.load(model_path)
        state_dict = torch.load(model_path)['model_state_dict']
        self.model.load_state_dict(state_dict)  # Apply the state dictionary
        self.model.eval()
        # self.model = self.model.to(self.device)  # Mo
        if torch.cuda.is_available():
            self.device = "cuda"
            # self.model = torch.load(model_path).to(self.device)
            self.model = self.model.to(self.device)
        else:
            # self.model = torch.load(model_path, map_location=torch.device("cpu"))
            self.device = "cpu"
            self.model = self.model.to(self.device)
            # self.device = "cpu"
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        self.to_tensor_func = self._get_preprocessing(preprocessing_fn)

    def _get_preprocessing(self, preprocessing_fn):
        def to_tensor(x, **kwargs):
            return x.transpose(2, 0, 1).astype('float32')

        transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor),
        ]
        return albu.Compose(transform)

    def read_imagefile_to_array(self, filename):
        image = cv2.imread(filename)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def detect_from_file(self, filename):
        img_array = self.read_imagefile_to_array(filename)
        return self.detect(img_array)

    def detect(self, img_array):
        # image_tensor = self.to_tensor_func(image=img_array)["image"]
        # x_tensor = torch.from_numpy(image_tensor).to(self.device).unsqueeze(0)
        # model_output = self.model.forward(x_tensor).cpu().numpy()

        model_output = self.model.forward(img_array)

        segmentation_map = torch.squeeze(model_output['binary_seg_pred']).to('cpu').numpy() * 255
        segmentation_map = segmentation_map.astype(np.uint8)

        # Display the segmentation map
        cv2.imshow('Segmentation', segmentation_map)
        # print(model_output.shape)
        # background, left, right = model_output[0,0,:,:], model_output[0,1,:,:], model_output[0,2,:,:]
        # Assuming model_output is your output tensor

        # segmentation_map = model_output[0, 0, :, :]* 255
        # If needed, apply thresholding to create a binary segmentation map
        # Example: segmentation_map = (segmentation_map > threshold_value).astype(np.uint8)

        # Initialize masks for left and right lanes
        left_lane_mask = np.zeros_like(segmentation_map)
        right_lane_mask = np.zeros_like(segmentation_map)

        # Define criteria for distinguishing left and right lanes
        # This could be based on the distribution of lane pixels across the width of the image
        # For simplicity, let's use the median x-coordinate of lane pixels
        lane_pixels = np.argwhere(segmentation_map > 0)  # Assuming non-zero pixels are lane pixels
        if len(lane_pixels) > 0:
            median_x_coordinate = np.median(lane_pixels[:, 1])  # x-coordinate is the second value

            for y, x in lane_pixels:
                if x < median_x_coordinate:
                    left_lane_mask[y, x] = 1
                else:
                    right_lane_mask[y, x] = 1
        return left_lane_mask, right_lane_mask, segmentation_map
        # return background, left, right

    def detect_and_fit(self, img_array):
        left, right, segmentation = self.detect(img_array)
        left_poly, left_coeffs, coff_check_left = self.fit_poly(left)
        right_poly, right_coeffs, coff_check_right = self.fit_poly(right)
        return left_poly, right_poly, left, right, left_coeffs, right_coeffs, coff_check_left, coff_check_right

    def fit_poly(self, probs):
        probs_flat = np.ravel(probs[self.cut_v:, :])
        mask = probs_flat > 0.3
        coff_check = True
        if not np.any(mask):
            # Handle the empty case (e.g., return a default polynomial or None)
            default_coeffs = [0]
            coff_check = False
            return np.poly1d(default_coeffs), default_coeffs, coff_check

        coeffs = np.polyfit(self.grid[:, 0][mask], self.grid[:, 1][mask], deg=3, w=probs_flat[mask])
        return np.poly1d(coeffs), coeffs, coff_check

    def __call__(self, img):
        if isinstance(img, str):
            img = self.read_imagefile_to_array(img)
            # cv2.namedWindow("Display window", cv2.WINDOW_NORMAL)
            # Display the image
            # cv2.imshow("Display window", img)

        return self.detect_and_fit(img)
