import math

import pytest
import torch
from torch.autograd import gradcheck

import kornia
from kornia.testing import BaseTester


class TestICtCpToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.rgb_to_ictcp(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.rgb_to_ictcp(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert kornia.color.rgb_to_ictcp([0.0])

        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_ictcp(img)

        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_ictcp(img)

    def test_unit(self, device, dtype):
        # data = torch.tensor(
        #     [
        #         [
        #             [
        #                 [50.21928787, 23.29810143, 14.98279190, 62.50927353, 72.78904724],
        #                 [70.86846924, 68.75330353, 52.81696701, 76.17090607, 88.63134003],
        #                 [46.87160873, 72.38699341, 37.71450806, 82.57386780, 74.79967499],
        #                 [77.33016968, 47.39180374, 61.76217651, 90.83254242, 86.96239471],
        #             ],
        #             [
        #                 [65.81327057, -3.69859719, 0.16971001, 14.86583614, -65.54960632],
        #                 [-41.03258133, -19.52661896, 64.16155243, -58.53935242, -71.78411102],
        #                 [112.05227661, -60.13330460, 43.07910538, -51.01456833, -58.25787354],
        #                 [-62.37575531, 50.88882065, -39.27450943, 17.00958824, -24.93779755],
        #             ],
        #             [
        #                 [-69.53346252, -73.34986877, -11.47461891, 66.73863220, 70.43983459],
        #                 [51.92737579, 58.77009583, 45.97863388, 24.44452858, 98.81991577],
        #                 [-7.60597992, 78.97976685, -69.31867218, 67.33953857, 14.28889370],
        #                 [92.31149292, -85.91405487, -32.83668518, -23.45091820, 69.99038696],
        #             ],
        #         ]
        #     ],
        #     device=device,
        #     dtype=dtype,
        # )
        #
        # # Reference output generated using skimage: lab2rgb(data)
        #
        # expected = torch.tensor(
        #     [
        #         [
        #             [
        #                 [0.63513142, 0.0, 0.10660624, 0.79048697, 0.26823414],
        #                 [0.48903025, 0.64529494, 0.91140099, 0.15877841, 0.45987959],
        #                 [1.0, 0.36069696, 0.29236125, 0.55744393, 0.0],
        #                 [0.41710863, 0.3198324, 0.0, 0.94256868, 0.82748892],
        #             ],
        #             [
        #                 [0.28210726, 0.26080003, 0.15027717, 0.54540429, 0.80323837],
        #                 [0.748392, 0.68774842, 0.24204415, 0.83695682, 0.9902132],
        #                 [0.0, 0.79101603, 0.26633725, 0.89223337, 0.82301254],
        #                 [0.84857086, 0.34455393, 0.66555314, 0.86168397, 0.8948667],
        #             ],
        #             [
        #                 [0.94172458, 0.66390044, 0.21043296, 0.02453515, 0.04169043],
        #                 [0.28233233, 0.20235374, 0.19803933, 0.55069441, 0.0],
        #                 [0.50205101, 0.0, 0.79745394, 0.25376936, 0.6114783],
        #                 [0.0, 1.0, 0.80867314, 1.0, 0.28778443],
        #             ],
        #         ]
        #     ],
        #     device=device,
        #     dtype=dtype,
        # )

        expected = [[[0.12059001, 0.02577267,  0.01494905],
                     [0.12817414, 0.02898962,  0.016383],
                     [0.12036308, 0.02591178,  0.01536138],
                     [0.12528643, 0.02432277,  0.01463714],
                     [0.08077499, 0.01287576,  0.00900849]],

                    [[0.13092962, 0.02537485,  0.01495433]
            [0.08808649 - 0.01626149  0.00954006]
        [0.08300461 - 0.01418583 0.01043896]
        [0.12193512 - 0.02502472  0.01494394]
        [0.12748005 - 0.02627225 0.01596024]]

        [[0.0964701 - 0.01257366  0.00734427]
         [0.13089199 - 0.02540577  0.01585178]
        [0.13372969 - 0.02623934
        0.01586302]
        [0.09633954 - 0.00559128  0.00357232]
        [0.03600475 - 0.00978714 - 0.00024434]]

        [[0.13250504 - 0.02548079  0.01388817]
         [0.02632138 - 0.01559767 - 0.00761916]
         [0.1306375 - 0.02601374  0.01512429]
        [0.10226605 - 0.01940479
        0.0127325]
        [0.09441516 - 0.01554101  0.01041815]]

        [[0.05544724  0.00565235 - 0.03712839]
         [0.08854769 - 0.00421003  0.0036087]
        [0.11299569 - 0.00360864
        0.00211407]
        [0.10788765 - 0.00440173  0.00331131]
        [0.12932168 - 0.02427474
        0.01466639]]]

        # self.assert_close(kornia.color.lab_to_rgb(data), expected)
        # self.assert_close(kornia.color.lab_to_rgb(data, clip=False), expected_unclipped)

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_ictcp, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.rgb_to_ictcp
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))

    def test_module(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.RgbToICtCp().to(device, dtype)
        fcn = kornia.color.rgb_to_ictcp
        self.assert_close(ops(img), fcn(img))

    def test_forth_and_back(self, device, dtype):
        data = torch.rand(3, 4, 5, device=device, dtype=dtype)
        ictcp = kornia.color.rgb_to_ictcp
        rgb = kornia.color.ictcp_to_rgb

        data_out = rgb(ictcp(data))
        self.assert_close(data_out, data, low_tolerance=True)  # low_tolerance for float32


# class TestRgbToictcp(BaseTester):
#     def test_smoke(self, device, dtype):
#         C, H, W = 3, 4, 5
#         img = torch.rand(C, H, W, device=device, dtype=dtype)
#         out = kornia.color.rgb_to_ictcp(img)
#         assert out.device == img.device
#         assert out.dtype == img.dtype
#
#     def test_smoke_byte(self, device):
#         C, H, W = 3, 4, 5
#         img = torch.randint(0, 255, (C, H, W), device=device, dtype=torch.uint8)
#         out = kornia.color.rgb_to_ictcp(img)
#         assert out.device == img.device
#         assert out.dtype == img.dtype
#
#     @pytest.mark.parametrize("batch_size, height, width", [(1, 3, 4), (2, 2, 4), (3, 4, 1)])
#     def test_cardinality(self, device, dtype, batch_size, height, width):
#         img = torch.ones(batch_size, 3, height, width, device=device, dtype=dtype)
#         assert kornia.color.rgb_to_ictcp(img).shape == (batch_size, 1, height, width)
#
#     def test_exception(self, device, dtype):
#         with pytest.raises(TypeError):
#             assert kornia.color.rgb_to_ictcp([0.0])
#
#         with pytest.raises(ValueError):
#             img = torch.ones(1, 1, device=device, dtype=dtype)
#             assert kornia.color.rgb_to_ictcp(img)
#
#         with pytest.raises(ValueError):
#             img = torch.ones(2, 1, 1, device=device, dtype=dtype)
#             assert kornia.color.rgb_to_ictcp(img)
#
#         with pytest.raises(ValueError):
#             img = torch.ones(3, 1, 1, device=device, dtype=dtype)
#             assert kornia.color.rgb_to_ictcp(img)
#
#     def test_opencv(self, device, dtype):
#         data = torch.tensor(
#             [
#                 [
#                     [0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912],
#                     [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008],
#                     [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481],
#                     [0.0086008, 0.8288748, 0.9647092, 0.8922020, 0.7614344],
#                     [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593],
#                 ],
#                 [
#                     [0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274],
#                     [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.0632100],
#                     [0.6171775, 0.8624780, 0.4126036, 0.7600935, 0.7279997],
#                     [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165],
#                     [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426],
#                 ],
#                 [
#                     [0.2869580, 0.4700376, 0.2743714, 0.8135023, 0.2229074],
#                     [0.9306560, 0.3734594, 0.4566821, 0.7599275, 0.7557513],
#                     [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.1315770],
#                     [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012],
#                     [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532],
#                 ],
#             ],
#             device=device,
#             dtype=dtype,
#         )
#
#         # Output data generated with OpenCV 4.1.1: cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
#         expected = torch.tensor(
#             [
#                 [
#                     [0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
#                     [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
#                     [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
#                     [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
#                     [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805],
#                 ]
#             ],
#             device=device,
#             dtype=dtype,
#         )
#
#         img_gray = kornia.color.rgb_to_ictcp(data)
#         self.assert_close(img_gray, expected)
#
#     def test_custom_rgb_weights(self, device, dtype):
#         B, C, H, W = 2, 3, 4, 4
#         img = torch.ones(B, C, H, W, device=device, dtype=dtype)
#         img_gray = kornia.color.rgb_to_ictcp(img)
#         assert img_gray.device == device
#         assert img_gray.dtype == dtype
#
#     @pytest.mark.grad()
#     def test_gradcheck(self, device, dtype):
#         B, C, H, W = 2, 3, 4, 4
#         img = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
#         assert gradcheck(kornia.color.rgb_to_ictcp, (img,), raise_exception=True, fast_mode=True)
#
#     def test_module(self, device, dtype):
#         B, C, H, W = 2, 3, 4, 4
#         img = torch.ones(B, C, H, W, device=device, dtype=dtype)
#         gray_ops = kornia.color.RgbToICtCp().to(device, dtype)
#         gray_fcn = kornia.color.rgb_to_ictcp
#         self.assert_close(gray_ops(img), gray_fcn(img))
