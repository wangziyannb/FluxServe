from __future__ import annotations

import unittest

import torch

from flux_kernel import (
    interleave_nvfp4_block_scale,
    static_scaled_fp8_quant,
    static_scaled_nvfp4_quant,
)


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


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class StaticScaledNvfp4QuantTest(unittest.TestCase):
    @staticmethod
    def _encode_reference(values: torch.Tensor) -> torch.Tensor:
        magnitude = values.abs()
        code = sum(
            (magnitude > midpoint).to(torch.uint8)
            for midpoint in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
        )
        code += sum(
            (magnitude == midpoint).to(torch.uint8)
            for midpoint in (0.75, 1.75, 3.5)
        )
        return code | ((values < 0).to(torch.uint8) << 3)

    def test_matches_torch_reference_and_blocked_scale_layout(self):
        for dtype in (torch.bfloat16, torch.float16):
            torch.manual_seed(7)
            input_ = torch.randn(5, 64, device="cuda", dtype=dtype)
            input_[0, :16] = 0
            global_scale = torch.tensor(0.02, device="cuda", dtype=torch.float32)

            packed, natural_scale = static_scaled_nvfp4_quant(
                input_, global_scale, blocked_scale=False
            )
            blocked_packed, blocked_scale = static_scaled_nvfp4_quant(
                input_, global_scale, blocked_scale=True
            )

            blocks = input_.float().reshape(5, 4, 16)
            expected_scale = blocks.abs().amax(dim=-1) / (6.0 * global_scale)
            expected_scale = torch.where(
                expected_scale == 0, torch.ones_like(expected_scale), expected_scale
            )
            expected_scale = expected_scale.clamp(2**-9, 448).to(
                torch.float8_e4m3fn
            )
            normalized = (
                blocks / (expected_scale.float().unsqueeze(-1) * global_scale)
            ).clamp(-6, 6)
            code = self._encode_reference(normalized).reshape(5, 64)
            expected_packed = code[:, 0::2] | (code[:, 1::2] << 4)

            torch.cuda.synchronize()
            # Fast division can move a value sitting within float epsilon of an
            # E2M1 midpoint to the neighboring representable value.
            packed_mismatches = torch.count_nonzero(
                packed.view(torch.uint8) != expected_packed
            ).item()
            self.assertLessEqual(packed_mismatches, 2)
            self.assertTrue(
                torch.equal(natural_scale.float(), expected_scale.float())
            )
            self.assertTrue(
                torch.equal(
                    blocked_packed.view(torch.uint8), packed.view(torch.uint8)
                )
            )
            self.assertTrue(
                torch.equal(
                    blocked_scale.float(),
                    interleave_nvfp4_block_scale(expected_scale).float(),
                )
            )

    def test_rounds_e2m1_midpoints_to_even(self):
        normalized = torch.tensor(
            [
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5.0,
                -0.75,
                -1.75,
                -3.5,
                6,
                0,
                0,
                0,
                0,
                0,
            ],
            device="cuda",
            dtype=torch.bfloat16,
        ).reshape(1, 16)
        global_scale = torch.ones(1, device="cuda", dtype=torch.float32)
        packed, _ = static_scaled_nvfp4_quant(
            normalized, global_scale, blocked_scale=False
        )

        # The per-block scale is one, so these are the direct E2M1 codes.
        expected_codes = self._encode_reference(normalized)
        expected = expected_codes[:, 0::2] | (expected_codes[:, 1::2] << 4)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(packed.view(torch.uint8), expected))

    def test_rejects_invalid_inputs(self):
        input_ = torch.ones(4, 32, device="cuda", dtype=torch.bfloat16)
        scale = torch.ones(1, device="cuda", dtype=torch.float32)

        with self.assertRaisesRegex(TypeError, "input must use"):
            static_scaled_nvfp4_quant(
                input_.float(), scale, blocked_scale=False
            )
        with self.assertRaisesRegex(ValueError, "2-D"):
            static_scaled_nvfp4_quant(
                input_.reshape(2, 2, 32), scale, blocked_scale=False
            )
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            static_scaled_nvfp4_quant(
                input_[:, :-1], scale, blocked_scale=False
            )
        with self.assertRaisesRegex(ValueError, "single float32"):
            static_scaled_nvfp4_quant(
                input_, scale.half(), blocked_scale=False
            )


if __name__ == "__main__":
    unittest.main()
