from typing import List, Union, Tuple
from PIL import Image
from abc import abstractmethod, ABCMeta
import random
from math import pi as PI
import cv2
import torch
from torch import Tensor
from torchvision.transforms.functional import pil_to_tensor, affine, \
    adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue, \
    resize, to_pil_image, hflip, center_crop, pad, rgb_to_grayscale
from albumentations.augmentations.transforms import RandomGamma,MotionBlur,Sharpen,Blur,GaussianBlur,Equalize,ISONoise,RandomToneCurve, RandomContrast
from albumentations import ElasticTransform,ShiftScaleRotate,CLAHE
import albumentations as A
import numpy as np
from torchvision import transforms


class Transform(metaclass=ABCMeta):
    def __init__(self, p: float = 1.0) -> None:
        self._p = p

    @property
    def p(self) -> float:
        return self._p

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'

    @abstractmethod
    def __call__(self, image: Union[Image.Image, Tensor],
                 ) -> Union[Image.Image, Tensor]:
        pass


class Compose:
    def __init__(self, transforms: List[Transform]) -> None:
        self._transforms = transforms

    def __call__(self, image: Image.Image
                 ) ->  Union[Image.Image, Tensor]:
        for transform in self._transforms:
            p = random.random()
            if p < transform.p:
                image = transform(image=image)
        return image

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self._transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class PILToTensor(Transform):
    def __init__(self, p: float = 1) -> None:
        super().__init__(p=p)

    def __call__(self, image: Image.Image
                 ) -> Tensor:
        return pil_to_tensor(image)

class ImageToTensor(Transform):
    def __init__(self, p: float = 1) -> None:
        super().__init__(p=p)
        self._transform= transforms.ToTensor()

    def __call__(self, image: Image.Image
                 ) -> Tensor:
        return self._transform(image)

class RandomAffine(Transform):
    """
    degrees: Tuple[int, int]: -180 to 180 degree for rotation. Clockwise
    translate: Tuple[int, int, int, int]: First two value is the range of
        horizontal translation. Last two value is the range of vertical
        translation. Applied after rotation.
    scale: Tuple[float, float]: Zoom/scale range, 0 - 1: zoom out.
        Above 1: zoom in
    shear: Tuple[float, float, float, float]: First two value is the range of
        horizontal shear. Last two value is the range of vertical
        shear.
    """

    def __init__(self, degrees: Tuple[int, int] = (0, 0),
                 translate: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 scale: Tuple[float, float] = (1, 1),
                 shear: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 resample: Image = Image.NEAREST,
                 p: float = 1.0) -> None:
        super().__init__(p)
        self._degrees = degrees
        self._translate = translate
        self._scale = scale
        self._shear = shear
        self._resample = resample

    def __call__(self, image: Union[Image.Image, Tensor]
                 ) -> Union[Image.Image, Tensor]:
        rotate_degree = random.randint(self._degrees[0], self._degrees[1])
        horizontal_translation = random.randint(
            self._translate[0], self._translate[1])
        vertical_translation = random.randint(
            self._translate[2], self._translate[3])
        scale = random.uniform(self._scale[0], self._scale[1])
        horizontal_shear = random.uniform(self._shear[0], self._shear[1])
        vertical_shear = random.uniform(self._shear[2], self._shear[3])

        image = affine(
            image, rotate_degree,
            [horizontal_translation, vertical_translation],
            scale, [horizontal_shear, vertical_shear]
        )

        return image


class ColorJitter(Transform):
    """
    brightness: 0 black image. 1 original image. Non negative
    contrast: 0 solid gray image. 1 original image. Non negative
    saturation: 0 black and white image. 1 original image. Non negative
    hue: 0 original image. Range within [-0.5, 0.5]
    """

    def __init__(self, brightness: Union[float, Tuple[float, float]] = 1.0,
                 contrast: Union[float, Tuple[float, float]] = 1.0,
                 saturation: Union[float, Tuple[float, float]] = 1.0,
                 hue: Union[float, Tuple[float, float]] = 0.0,
                 p: float = 1.0) -> None:
        super().__init__(p)
        self._brightness = brightness
        self._contrast = contrast
        self._saturation = saturation
        self._hue = hue

    def __call__(self, image: Union[Image.Image, Tensor]
                 ) -> Union[Image.Image, Tensor]:
        brightness = self._brightness if isinstance(self._brightness, float) \
            else random.uniform(self._brightness[0], self._brightness[1])
        contrast = self._contrast if isinstance(self._contrast, float) \
            else random.uniform(self._contrast[0], self._contrast[1])
        saturation = self._saturation if isinstance(self._saturation, float) \
            else random.uniform(self._saturation[0], self._saturation[1])
        hue = self._hue if isinstance(self._hue, float) \
            else random.uniform(self._hue[0], self._hue[1])

        image = adjust_brightness(image, brightness)
        image = adjust_contrast(image, contrast)
        image = adjust_saturation(image, saturation)
        image = adjust_hue(image, hue)

        return image


class Resize(Transform):
    def __init__(self,
                 new_size: Tuple[int, int],
                 p: float = 1.0) -> None:
        super().__init__(p)
        self._size = new_size

    def __call__(self, image: Union[Image.Image, Tensor]
                 ) -> Union[Image.Image, Tensor]:
        return (resize(image, list(self._size)))


class OccludingPatches(Transform):
    def __init__(self, max_holes: int,
                 min_intensity: int,
                 max_intensity: int,
                 max_height: int,
                 max_width: int,
                 p: float = 0.5) -> None:
        """Occuding Patches as described in DRUNet."""
        super().__init__(p=p)
        self._max_holes = max_holes
        self._min_intensity = min_intensity
        self._max_intensity = max_intensity
        self._max_height = max_height
        self._max_width = max_width
        assert(max_width > 60)

    def _f(self,
           image: Tensor,
           fill_val: int) -> Tensor:
        w = image.shape[2]
        rand_x1 = torch.randint(0, w, (self._max_holes,))
        rand_w = torch.randint(60,
                               self._max_width, (self._max_holes,))
        rand_x2 = rand_x1 + rand_w
        rand_x2 = torch.clip(rand_x2, 0, w)
        temp = image.clone()
        for w_i, start, end in zip(rand_w, rand_x1, rand_x2):
            w_i = end - start
            weight = torch.linspace(0, PI, w_i)
            weight = torch.sin(weight)
            weight = weight.unsqueeze(0).unsqueeze(0)
            temp[:, :, start:end] = weight * fill_val \
                + (1 - weight) * temp[:, :, start:end]
        return temp

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:
        # normalize
        image = pil_to_tensor(image)
        # random value
        val = torch.randint(int(self._min_intensity),
                            int(self._max_intensity),
                            (1,))
        occluded_image = self._f(image, val)
        occluded_image = to_pil_image(occluded_image)
        return occluded_image


class NonLinearIntensityShift(Transform):
    def __init__(self,
                 min_val: float,
                 max_val: float,
                 p_min: float,
                 p_max: float,
                 p: float = 0.5) -> None:
        """Non Linear Intensity Shift as per DRUNet"""
        super().__init__(p=p)
        self._min_val = min_val
        self._max_val = max_val
        self._p_min = p_min
        self._p_max = p_max

    def _f(self, image: Tensor) -> Tensor:
        a = torch.randint(int(self._min_val * 100),
                          int(self._max_val * 100), (1,)) / 100
        b = torch.randint(int(self._min_val * 100),
                          int(self._max_val * 100), (1,)) / 100
        p = torch.randint(int(self._p_min * 10),
                          int(self._p_max * 10), (1,)) / 10
        image = (image ** p) * (1 + a + b) - a
        # normalize
        image = image - image.min()
        image = image / image.max()
        return image

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:
        # normalize
        image = pil_to_tensor(image)
        image = self._f(image)
        image = to_pil_image(image)
        return image


class HorizontalFlip(Transform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:
        image = hflip(image)
        return image


class Zoom(Transform):
    def __init__(self,
                 scale: Tuple[float, float] = (0.5, 1.5),
                 p: float = 0.5) -> None:
        super().__init__(p=p)
        self._scale = scale

    def _f(self, image: Tensor, scale: float) -> Tensor:
        c, h_old, w_old = image.shape
        h_new = int(scale * h_old)
        w_new = int(scale * w_old)
        image = resize(image, [h_new, w_new], interpolation=Image.NEAREST)
        if h_new > h_old and w_new > w_old:
            image = center_crop(image, [h_old, w_old])
        elif h_new <= h_old and w_new <= w_old:
            h_pad = (h_old - h_new) // 2
            w_pad = (w_old - w_new) // 2
            image = pad(image, padding=[h_pad, w_pad])
        else:
            raise ValueError(
                'You should either scale up or down in both x-y direction.')
        return image

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:
        scale = random.randint(
            int(self._scale[0] * 10),
            int(self._scale[1] * 10)
        ) / 10
        image = pil_to_tensor(image)
        image = self._f(image, scale)
        image = to_pil_image(image)
        return image


class CenterCrop(Transform):
    def __init__(self,
                 height: int,
                 width: int,
                 p: float = 0.5) -> None:
        super().__init__(p=p)
        self._height = height
        self._width = width

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:
        image = center_crop(image, [self._height, self._width])
        return image


class RGBToGrayscale(Transform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:
        image = rgb_to_grayscale(image, num_output_channels=1)
        return image


class QualityReduction(Transform):
    def __init__(self,
                 quality: Union[Tuple[float, float], float] = 1.0,
                 p: float = 0.5) -> None:
        super().__init__(p=p)
        if isinstance(quality, float):
            assert((quality > 0.) and (quality <= 1.0))
        else:
            assert((quality[0] > 0.) and (quality[1] <= 1.0))
        self._quality = quality

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:
        factor_h, factor_w = (self._quality, self._quality) \
            if isinstance(self._quality, float) \
            else self._quality
        w, h = image.size
        h_new = int(factor_h * h)
        w_new = int(factor_w * w)
        image = resize(image, [h_new, w_new], interpolation=Image.NEAREST)
        image = resize(image, [h, w], interpolation=Image.NEAREST)
        return image


class GaussianNoise(Transform):
    def __init__(self,
                 mean: float = 0.0,
                 std: float = 0.1,
                 p: float = 0.5) -> None:
        super().__init__(p=p)
        self._mean = mean
        self._std = std

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:
        image = pil_to_tensor(image).type(torch.float32)
        c, h, w = image.shape
        image -= image.view((c, -1)).mean(-1).reshape(-1, 1, 1)
        image /= image.view((c, -1)).std(-1).reshape(-1, 1, 1)
        c, h, w = image.shape
        gauss = self._mean + self._std * torch.randn(c, h, w)
        image += gauss
        image -= image.view((c, -1)).min(-1).values.reshape(-1, 1, 1)
        image /= image.view((c, -1)).max(-1).values.reshape(-1, 1, 1)
        image = to_pil_image(image)
        return image


class SpekcleNoise(Transform):
    def __init__(self,
                 mean: float = 0.0,
                 std: float = 0.1,
                 p: float = 0.5) -> None:
        super().__init__(p=p)
        self._mean = mean
        self._std = std

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:
        image = pil_to_tensor(image).type(torch.float32)
        c, h, w = image.shape
        image -= image.view((c, -1)).mean(-1).reshape(-1, 1, 1)
        image /= image.view((c, -1)).std(-1).reshape(-1, 1, 1)
        c, h, w = image.shape
        gauss = self._mean + self._std * torch.randn(c, h, w)
        image += gauss * image
        image -= image.view((c, -1)).min(-1).values.reshape(-1, 1, 1)
        image /= image.view((c, -1)).max(-1).values.reshape(-1, 1, 1)
        image = to_pil_image(image)
        return image


# class CLAHE(Transform):
#     def __init__(self,
#                  clip_limit: float = 2.0,
#                  tile_grid_size: Tuple[int, int] = (8, 8),
#                  p: float = 1.0):
#         super().__init__(p=p)
#         self._clip_limit = clip_limit
#         self._tile_grid_size = tile_grid_size

#     def __call__(self,
#                  image: Union[Image.Image, Tensor],
#           ) \
#             ->  Union[Image.Image, Tensor]:
#         image = pil_to_tensor(image)
#         image = image.transpose(0, 1).transpose(1, 2).numpy()
#         assert(image.shape[-1] == 3)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         clahe = cv2.createCLAHE(
#             clipLimit=self._clip_limit,
#             tileGridSize=self._tile_grid_size)
#         image = clahe.apply(image)
#         image = Image.fromarray(image)
#         return image, mask

class ResizeandPad(Transform):
    def __init__(self, new_size: Tuple[int,int]=(175,299), p: float = 1) -> None:
        super().__init__(p=p)
        self._newsize=new_size
        self._pad=int((new_size[1]-new_size[0])//2)
        self._transforms= transforms.Compose([transforms.Resize(new_size), 
                                            transforms.Pad((0,self._pad,0,self._pad))])

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:

        
        image=self._transforms(image)

        return image


class Gamma(Transform):
    def __init__(self,gamma_limit: Union[float, Tuple[float,float]]=(80,120), p: float = 0.5) -> None:
        super().__init__(p=p)
        self._gamma_limit=gamma_limit

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:
        transform=RandomGamma(self._gamma_limit,p=1)
        image=np.array(image)
        image,mask = transform(image=image,mask=image).values()
        image=Image.fromarray(image)
        return image

class ElasticDeformation(Transform):
    def __init__(self,sigma:float = 50 , alpha_affine: float = 50, p: float = 0.5) -> None:
        super().__init__(p=p)
        self._sigma=sigma
        self._alpha_affine=alpha_affine

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:
        transform=ElasticTransform(1,self._sigma,self._alpha_affine,interpolation=cv2.INTER_NEAREST,p=1)
        image=np.array(image)
        image,mask = transform(image=image,mask=image).values()
        image=Image.fromarray(image)
        return image


class Motion_Blur(Transform):
    def __init__(self,blur_limit: int=100, p: float = 0.5) -> None:
        super().__init__(p=p)
        self._blur_limit=blur_limit

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:
        transform=MotionBlur(self._blur_limit,p=1)
        image=np.array(image)
        image,mask = transform(image=image,mask=image).values()
        image=Image.fromarray(image)
        return image

class Ghosting(Transform):
    def __init__(self,shift_limit: float = 0.1, p: float = 0.5) -> None:
        super().__init__(p=p)
        self._shift_limit=shift_limit

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:

        transform=ShiftScaleRotate(shift_limit=self._shift_limit,scale_limit=0,rotate_limit=0,p=1)
        image=np.array(image)
        shifted_image,mask = transform(image=image,mask=image).values()
        image=Image.fromarray(image)
        shifted_image=Image.fromarray(shifted_image)
        image=Image.blend(image,shifted_image,0.2)
        return image

class Blurring(Transform):
    def __init__(self,blurlimit: int=20, p: float = 0.5) -> None:
        super().__init__(p=p)
        self._blurlimit=blurlimit

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:
        transform=Blur(blur_limit=self._blurlimit,p=1)
        image=np.array(image)
        image,mask = transform(image=image,mask=image).values()
        image=Image.fromarray(image)
        return image

class Sharpening(Transform):
    def __init__(self,alpha: Tuple[float,float]=(0.9,1.0),lightness: Tuple[float,float]=(0.9, 1.0), p: float = 0.5) -> None:
        super().__init__(p=p)
        self._alpha=alpha
        self._lightness=lightness

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:
        transform=Sharpen(alpha=self._alpha, lightness=self._lightness, p=1)
        image=np.array(image)
        image,mask = transform(image=image,mask=image).values()
        image=Image.fromarray(image)
        return image

class CLAHE_aug(Transform):
    def __init__(self,clip_limit: Tuple[float,float]=(0.75,2.0), p: float = 0.5) -> None:
        super().__init__(p=p)
        self._clip_limit=clip_limit

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:
        transform=CLAHE(clip_limit=self._clip_limit, p=1)
        image=np.array(image)
        image,mask = transform(image=image,mask=image).values()
        image=Image.fromarray(image)
        return image


class BiasField(Transform):
    def __init__(self,gammarange: Tuple[float,float,float]=(0.3,0.7,2.5),transitionpoint:Tuple[float,float]=(0.4,0.6), p: float = 0.5) -> None:
        super().__init__(p=p)
        self._gammarange=gammarange
        self._transitionpoint=transitionpoint

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:

        transform=ShiftScaleRotate(shift_limit=0,scale_limit=0,rotate_limit=180,p=1,interpolation=cv2.INTER_NEAREST)


        image=image.convert('L')
        image=np.array(image)
        gm=image.shape[1]
        gml=[[]]
        transition=random.uniform(self._transitionpoint[0],self._transitionpoint[1])

        for col in range(image.shape[1]):
            if col<int(image.shape[1]*transition):
                gamma=self._gammarange[0]+(self._gammarange[1]-self._gammarange[0])*(col/(gm*transition))
            else:
                gamma=self._gammarange[1]+(self._gammarange[2]-self._gammarange[1])*((col-int(gm*transition))/(gm*transition))
            invgama= 1.0 / gamma
            gml[0].append(invgama)

        gml=np.repeat(gml,image.shape[0],axis=0)
        gml,tmp=transform(image=gml,mask=gml).values()
        image=(image+0.0001)/255.0
        image=(image**gml)*255
        image.astype(int)

        image=Image.fromarray(image) 

        return image


class CraniumShadow(Transform):
    def __init__(self,thickness: Tuple[int,int]=(70,100),brightness:int=10,rotate:Tuple[float,float]=(10,20), p: float = 0.5) -> None:
        super().__init__(p=p)
        self._thickness=thickness
        self._brightness=brightness
        self._rotate=rotate

    def __call__(self,
                 image: Union[Image.Image, Tensor],
                 mask: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:

        transform1=ShiftScaleRotate(shift_limit=0,scale_limit=0,rotate_limit=[-1*self._rotate[0],-1*self._rotate[1]],p=1,interpolation=cv2.INTER_NEAREST)
        transform2=ShiftScaleRotate(shift_limit=0,scale_limit=0,rotate_limit=self._rotate,p=1,interpolation=cv2.INTER_NEAREST)

        image=np.array(image)
        mask=np.array(mask)
        mask1 = np.zeros(image.shape[:2], dtype="uint8")
        mask2 = np.zeros(image.shape[:2], dtype="uint8")

        cranium=np.argwhere(mask==1)
        cranium=cranium[cranium[:, 1].argsort()]
        left=cranium[0]
        left[0]+=random.randint(50,75) 
        right=cranium[-1]
        right[0]+=random.randint(50,75)

        w1=random.randint(self._thickness[0],self._thickness[1])
        w2=random.randint(self._thickness[0],self._thickness[1])

        bright=random.randint(0,self._brightness)
        for col in range(left[1],left[1]+w1):
            for row in range(left[0],image.shape[0]):
                mask1[row][col]=bright
        mask1,tmp=transform1(image=mask1,mask=mask1).values()

        bright=random.randint(0,self._brightness)
        for col in range(right[1]-w1,right[1]):
            for row in range(right[0],image.shape[0]):
                mask2[row][col]=bright
        mask2,tmp=transform2(image=mask2,mask=mask2).values()

        alphamask=mask1+mask2
        alphamask[alphamask==0]=255

        alphamask=GaussianBlur(blur_limit=(75, 95), sigma_limit=0,p=1,always_apply=True)(image=alphamask,mask=alphamask)['image']

        alphamask=Image.fromarray(alphamask)
        image=Image.fromarray(image)
        image.putalpha(alphamask)
        background = Image.new("RGB", image.size, (0, 0, 0))
        background.paste(image, mask = image.split()[3])
        # image=image.convert("RGB")
        mask=Image.fromarray(mask)

        return background


class StackedTransforms(Transform):
    def __init__(self, p: float = 1) -> None:
        super().__init__(p=p)
        self._transforms=[CLAHE(clip_limit=2.0,p=1),
                        RandomGamma(gamma_limit=(80,120),p=1),
                        Blur(blur_limit=7,p=1),
                        Sharpen(alpha=(0.9,1.0),lightness=(0.9,1.0),p=1)]

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:

        
        image=np.array(image)
        image=image.reshape((image.shape[0],image.shape[1],1))

        transform_stack=[image]

        for transform in self._transforms:
            aug_image,tmp = transform(image=image,mask=image).values()
            transform_stack.append(aug_image)
        
        image=np.concatenate(transform_stack, axis=2)
        # print(image.shape)
        return image


class StackedStratified(Transform):
    def __init__(self, p: float = 1) -> None:
        super().__init__(p=p)
        self._transforms=[CLAHE(clip_limit=(1.0,4.0),p=1),
                        CLAHE(clip_limit=(4.0,7.0),p=1),
                        CLAHE(clip_limit=(7.0,10.0),p=1),
                        Equalize(p=1),
                        RandomGamma(gamma_limit=(50,70),p=1),
                        RandomGamma(gamma_limit=(70,130),p=1),
                        RandomGamma(gamma_limit=(130,150),p=1),
                        RandomContrast(limit=(-0.5,0.0),p=1),
                        RandomContrast(limit=(0.0,0.5),p=1),
                        RandomContrast(limit=(0.5,1.0),p=1),
                        RandomToneCurve(scale=0.2,p=1),
                        RandomToneCurve(scale=0.2,p=1),
                        RandomToneCurve(scale=0.2,p=1),
                        GaussianBlur(blur_limit=(3,3),sigma_limit=(0,0),p=1),
                        GaussianBlur(blur_limit=(5,5),sigma_limit=(0,0),p=1),
                        GaussianBlur(blur_limit=(7,7),sigma_limit=(0,0),p=1),
                        Sharpen(alpha=(0.1,0.4),lightness=(0.9,1.0),p=1),
                        Sharpen(alpha=(0.4,0.7),lightness=(0.9,1.0),p=1),
                        Sharpen(alpha=(0.7,1.0),lightness=(0.9,1.0),p=1)]

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:

        
        image=np.array(image)
        image=image.reshape((image.shape[0],image.shape[1],1))

        transform_stack=[image]

        for transform in self._transforms:
            aug_image,tmp = transform(image=image,mask=image).values()
            transform_stack.append(aug_image)
        
        image=np.concatenate(transform_stack, axis=2)
        # print(image.shape)
        return image


class CompoundStackedTransforms(Transform):
    def __init__(self, p: float = 1) -> None:
        super().__init__(p=p)
        self._transforms=[CLAHE(clip_limit=2.0,p=1),
                        RandomGamma(gamma_limit=(80,120),p=1),
                        Blur(blur_limit=7,p=0.25),
                        Sharpen(alpha=(0.9,1.0),lightness=(0.9,1.0),p=1)]

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            -> Union[Image.Image, Tensor]:

        
        image=np.array(image)

        for transform in self._transforms:
            image,mask = transform(image=image,mask=image).values()

        image=Image.fromarray(image)
        return image

class CompoundStackedTransformsv2(Transform):
    def __init__(self, p: float = 1) -> None:
        super().__init__(p=p)
        self._transforms=[A.OneOf([RandomGamma(gamma_limit=(50,70),p=1),
                                RandomGamma(gamma_limit=(70,130),p=1),
                                RandomGamma(gamma_limit=(130,150),p=1)],p=0.5),
                        A.OneOf([CLAHE(clip_limit=(1.0,4.0),p=1),
                                    CLAHE(clip_limit=(4.0,7.0),p=1),
                                    CLAHE(clip_limit=(7.0,10.0),p=1),] ,p=0.5),
                        Equalize(p=0.5),
                        A.OneOf([RandomContrast(limit=(-0.5,0.0),p=1),
                                RandomContrast(limit=(0.0,0.5),p=1),
                                RandomContrast(limit=(0.5,1.0),p=1)]),
                        A.OneOf([GaussianBlur(blur_limit=(3,3),sigma_limit=(0,0),p=1),
                                GaussianBlur(blur_limit=(5,5),sigma_limit=(0,0),p=1),
                                GaussianBlur(blur_limit=(7,7),sigma_limit=(0,0),p=1)],p=0.5),
                        A.OneOf([Sharpen(alpha=(0.1,0.4),lightness=(0.9,1.0),p=1),
                                Sharpen(alpha=(0.4,0.7),lightness=(0.9,1.0),p=1),
                                Sharpen(alpha=(0.7,1.0),lightness=(0.9,1.0),p=1)],p=0.25)
                        ]

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:

        
        image=np.array(image)

        for transform in self._transforms:
            image,mask = transform(image=image,mask=image).values()

        image=Image.fromarray(image)
        return image

class CompoundAffine(Transform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)
        self._transforms=[RandomAffine(translate= [40, 60, 40, 60],p=0.5),
                        RandomAffine(degrees= [-20, 20],p=0.5),
                        RandomAffine(shear= [-20, 20, -20, 20],p=0.5),
                        Zoom(scale=(0.6,1.2),p=0.5),
                        HorizontalFlip(p=0.5)]

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            -> Union[Image.Image, Tensor]:


        for transform in self._transforms:
            image,mask = transform(image=image,mask=image)
        
        return image

class Shadows(Transform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)
        self._transforms=[CraniumShadow(thickness=(70,100),brightness=10,rotate=(10,20),p=1),
                        OccludingPatches(min_intensity=0,max_intensity=10,max_height=1,max_width=200,max_holes=1,p=1)]

    def __call__(self,
                 image: Union[Image.Image, Tensor]) \
            ->  Union[Image.Image, Tensor]:
        
        
        if random.random()<0.5:
            image, mask=self._transforms[0](image,image)
        else:
            image, mask=self._transforms[1](image,image)
        
        
        return image

