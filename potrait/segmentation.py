import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu


class Segmentation:
    """
    This class is responsible for generating the segmentation mask for potrait segmentation.
    """
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = self.get_preprocessing(smp.encoders.get_preprocessing_fn('mobilenet_v2', 'imagenet'))
        self.model = smp.Unet("mobilenet_v2")
        self.model.load_state_dict(torch.load("models/aug58.pth"))
        self.model.to(self.device)

    def get_mask(self, image):
        """
        :param image:
        :return: returns the binary segmentation mask of the person.
        """
        image = cv2.resize(image, (224, 448))
        preprocessor = self.transform(image=image)
        preprocessed_img = torch.tensor(preprocessor['image']).unsqueeze(0)
        preprocessed_img = preprocessed_img.to(self.device)
        result = self.model(preprocessed_img).detach().cpu().numpy()
        result = np.transpose(result[0], (1, 2, 0))
        result = np.uint8(result * 0.5)
        return result

    def to_tensor(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def get_preprocessing(self, preprocessing_fn):
        transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=self.to_tensor, mask=self.to_tensor),
        ]
        return albu.Compose(transform)


if __name__ == "__main__":
    img = cv2.imread("../images/potrait/4.jpg", 1)
    obj = Segmentation()
    results = obj.get_mask(img)
    cv2.imshow("result", results)
    cv2.waitKey(0)



