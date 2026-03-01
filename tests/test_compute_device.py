from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch


class ComputeDeviceTests(TestCase):
    def test_auto_prefers_cpu_when_cuda_unavailable(self) -> None:
        from src.integrations.storage import resolve_compute_device

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict("sys.modules", {"torch": fake_torch}):
            device, warning = resolve_compute_device("auto")

        self.assertEqual(device, "cpu")
        self.assertIsNone(warning)

    def test_gpu_request_falls_back_to_cpu_with_warning(self) -> None:
        from src.integrations.storage import resolve_compute_device

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict("sys.modules", {"torch": fake_torch}):
            device, warning = resolve_compute_device("gpu")

        self.assertEqual(device, "cpu")
        self.assertIn("Falling back to CPU", str(warning))
