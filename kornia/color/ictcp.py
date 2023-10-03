import numpy as np
import torch
from torch import nn

from kornia.core import Tensor


def _apply_transform(transform: Tensor, image: Tensor) -> Tensor:
    transform = transform.type(image.dtype)
    size = image.shape
    if len(image.shape) == 3:
        image = image.reshape(1, 3, -1)
    else:
        image = image.reshape(image.shape[0], 3, -1)

    # r: torch.Tensor = image[..., 0, :, :].flatten()
    # g: torch.Tensor = image[..., 1, :, :].flatten()
    # b: torch.Tensor = image[..., 2, :, :].flatten()
    # print(r)
    #
    # i = torch.vstack((r, g, b))
    # print(i.shape, i)

    transformed = torch.matmul(transform, image)
    return transformed.reshape(size)


def _tensor_to_power(image: Tensor, pow: float) -> Tensor:
    return torch.sign(image) * torch.abs(image) ** pow


def _normalize(image: Tensor) -> Tensor:
    m_1 = 2610 / 4096 * (1 / 4)
    m_2 = 2523 / 4096 * 128
    c_1 = 3424 / 4096
    c_2 = 2413 / 4096 * 32
    c_3 = 2392 / 4096 * 32

    Y_p = _tensor_to_power(image / 10000, m_1)
    N = _tensor_to_power((c_1 + c_2 * Y_p) / (c_3 * Y_p + 1), m_2)
    return N


def _unnormalize(image: Tensor) -> Tensor:
    m_1 = 2610 / 4096 * (1 / 4)
    m_2 = 2523 / 4096 * 128
    c_1 = 3424 / 4096
    c_2 = 2413 / 4096 * 32
    c_3 = 2392 / 4096 * 32

    V_p = _tensor_to_power(image, 1 / m_2)
    n = np.maximum(0, V_p - c_1)
    L = _tensor_to_power((n / (c_2 - c_3 * V_p)), 1 / m_1)
    C = 10000 * L
    return C


def rgb_to_ictcp(image: Tensor) -> Tensor:
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    rgb_to_lms_transform = torch.tensor([[1688, 2146, 262],
                                         [683, 2951, 462],
                                         [99, 309, 3688]]) / 4096

    lms_p_to_ictcp_transform = torch.tensor([[2048, 2048, 0],
                                             [6610, -13613, 7003],
                                             [17933, -17390, -543]]) / 4096

    lms = _apply_transform(rgb_to_lms_transform, image)
    lms_p = _normalize(lms)
    return _apply_transform(lms_p_to_ictcp_transform, lms_p)


def ictcp_to_rgb(image: Tensor) -> Tensor:
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    rgb_to_lms_transform = torch.tensor([[1688, 2146, 262],
                                         [683, 2951, 462],
                                         [99, 309, 3688]]) / 4096

    lms_p_to_ictcp_transform = torch.tensor([[2048, 2048, 0],
                                             [6610, -13613, 7003],
                                             [17933, -17390, -543]]) / 4096

    lms_to_rgb_transform = torch.inverse(rgb_to_lms_transform)
    ictcp_to_lms_p_transform = torch.inverse(lms_p_to_ictcp_transform)

    lms_p = _apply_transform(ictcp_to_lms_p_transform, image)
    lms = _unnormalize(lms_p)
    return _apply_transform(lms_to_rgb_transform, lms)


class RgbToICtCp(nn.Module):
    r"""Convert an image from Rgb to ICtCp.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        IctCp version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> ictcp = RgbToICtCp()
        >>> output = ictcp(input)  # 2x3x4x5
    """

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_ictcp(image)


class ICtCpToRgb(nn.Module):
    r"""Convert an image from ICtCp to Rgb.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = ICtCpToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def forward(self, image: Tensor) -> Tensor:
        return ictcp_to_rgb(image)


if __name__ == "__main__":
    # rgb = torch.tensor([0.4562, 0.0308, 0.0409], dtype=torch.float64).reshape(3, 1, 1)
    # rgb = torch.tensor([1, 0, 0], dtype=torch.float64).reshape(3, 1, 1)
    # print(rgb)

    # ictcp = rgb_to_ictcp(rgb)
    # print(ictcp)

    # rgb_new = ictcp_to_rgb(ictcp)
    # print(rgb_new)

    # print(rgb - rgb_new)
    # assert (torch.allclose(rgb, rgb_new))

    import colour
    import cv2

    # i = np.linspace(0, 1, 10)
    # ct = np.linspace(-1, 1, 10)
    # cp = np.linspace(-1, 1, 10)
    #
    # numpy_image = np.array([list(zip(i, ct, cp))])
    #
    #
    # # numpy_image = np.array([[[0.4562, 0.0308, 0.0409],
    # #                          [0.4562, 0.0308, 0.0409]]])
    # print(numpy_image.shape)
    # print(colour.RGB_to_ICtCp(numpy_image))
    #
    #
    # tensor = torch.Tensor(numpy_image).permute(2, 0, 1).unsqueeze(0)
    # print(rgb_to_ictcp(tensor))




    img = cv2.imread("/Users/alorthius/Documents/kotyky/fanart.jpg")
    img = cv2.resize(img, (5, 5))
    img = (img / 256).astype(float)
    r = colour.RGB_to_ICtCp(img)
    print(r)
    # np.savetxt("/Users/alorthius/Downloads/colour.txt", r)
    # cv2.imwrite("/Users/alorthius/Downloads/fanart.jpg", img.astype("uint8"))

    import sys
    np.set_printoptions(threshold=sys.maxsize)
    tensor = torch.Tensor(img).permute(2, 0, 1)
    t = rgb_to_ictcp(tensor)
    print(t)
    # print(t.permute(1, 2, 0))
    # np.savetxt("/Users/alorthius/Downloads/my.txt", t)



