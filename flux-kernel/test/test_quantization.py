from __future__ import annotations

import unittest

import torch

from flux_kernel import static_scaled_fp8_quant


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class StaticScaledFp8QuantTest(unittest.TestCase):
    def test_matches_torch_and_zero_fills_padding(self):
        for dtype in (torch.bfloat16, torch.float16):
            input_ = torch.linspace(
                -12, 12, 21 * 64, device="cuda", dtype=dtype
            ).reshape(21, 64)
            scale = torch.tensor(0.02, device="cuda", dtype=torch.float32)

            actual = static_scaled_fp8_quant(input_, scale, padded_rows=25)
            expected = torch.clamp(input_.float() / scale, -448, 448).to(
                torch.float8_e4m3fn
            )

            torch.cuda.synchronize()
            self.assertTrue(torch.equal(actual[:21].float(), expected.float()))
            self.assertEqual(torch.count_nonzero(actual[21:].float()).item(), 0)

    def test_rejects_invalid_dtype_scale_and_padding(self):
        input_ = torch.ones(4, 8, device="cuda", dtype=torch.bfloat16)
        scale = torch.ones(1, device="cuda", dtype=torch.float32)

        with self.assertRaisesRegex(TypeError, "input must use"):
            static_scaled_fp8_quant(input_.to(torch.int32), scale)
        with self.assertRaisesRegex(TypeError, "input must use"):
            static_scaled_fp8_quant(input_.to(torch.float32), scale)
        with self.assertRaisesRegex(ValueError, "single float32"):
            static_scaled_fp8_quant(input_, scale.to(torch.float16))
        with self.assertRaisesRegex(ValueError, "single float32"):
            static_scaled_fp8_quant(input_, scale.repeat(2))
        with self.assertRaisesRegex(ValueError, "at least"):
            static_scaled_fp8_quant(input_, scale, padded_rows=3)


if __name__ == "__main__":
    unittest.main()
