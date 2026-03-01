from __future__ import annotations

import importlib.util
from unittest import TestCase, skipUnless
from unittest.mock import patch

LANGCHAIN_COMMUNITY_AVAILABLE = importlib.util.find_spec("langchain_community") is not None
LANGCHAIN_NVIDIA_AVAILABLE = (
    importlib.util.find_spec("langchain_nvidia_ai_endpoints") is not None
)


@skipUnless(LANGCHAIN_COMMUNITY_AVAILABLE, "langchain_community is not installed")
class StorageSingletonTests(TestCase):
    def setUp(self) -> None:
        from src.integrations.storage import _build_store_cached, _get_embeddings_cached

        _build_store_cached.cache_clear()
        _get_embeddings_cached.cache_clear()

    @patch("src.integrations.storage.SentenceTransformerEmbeddings")
    def test_embeddings_cached_for_same_model(self, mock_embeddings) -> None:
        from src.integrations.storage import get_embeddings

        mock_embeddings.side_effect = lambda **_: object()

        first = get_embeddings("model-a")
        second = get_embeddings("model-a")

        self.assertIs(first, second)
        mock_embeddings.assert_called_once_with(
            model_name="model-a",
            model_kwargs={"device": "cpu"},
        )

    @patch("src.integrations.storage.SentenceTransformerEmbeddings")
    def test_embeddings_vary_by_model(self, mock_embeddings) -> None:
        from src.integrations.storage import get_embeddings

        mock_embeddings.side_effect = lambda **_: object()

        first = get_embeddings("model-a")
        second = get_embeddings("model-b")

        self.assertIsNot(first, second)
        self.assertEqual(mock_embeddings.call_count, 2)

    @patch("src.integrations.storage.resolve_compute_device")
    @patch("src.integrations.storage.SentenceTransformerEmbeddings")
    def test_embeddings_vary_by_device(self, mock_embeddings, mock_resolve_device) -> None:
        from src.integrations.storage import get_embeddings

        mock_resolve_device.side_effect = [("cpu", None), ("cuda", None)]
        mock_embeddings.side_effect = lambda **_: object()

        cpu_embeddings = get_embeddings("model-a", device="cpu")
        gpu_embeddings = get_embeddings("model-a", device="gpu")

        self.assertIsNot(cpu_embeddings, gpu_embeddings)
        self.assertEqual(mock_embeddings.call_count, 2)

    @patch("src.integrations.storage.Path.mkdir")
    @patch("src.integrations.storage.Chroma")
    @patch("src.integrations.storage.get_embeddings")
    def test_query_store_cached_for_same_inputs(
        self,
        mock_get_embeddings,
        mock_chroma,
        mock_mkdir,
    ) -> None:
        from src.integrations.storage import get_query_cache_store

        mock_get_embeddings.return_value = object()
        mock_chroma.side_effect = lambda **_: object()

        first = get_query_cache_store("C:\\cache-dir", embeddings_model_name="embed-a")
        second = get_query_cache_store("C:\\cache-dir", embeddings_model_name="embed-a")
        different = get_query_cache_store("C:\\cache-dir-2", embeddings_model_name="embed-a")

        self.assertIs(first, second)
        self.assertIsNot(first, different)
        self.assertEqual(mock_chroma.call_count, 2)
        self.assertGreaterEqual(mock_mkdir.call_count, 2)


@skipUnless(LANGCHAIN_NVIDIA_AVAILABLE, "langchain_nvidia_ai_endpoints is not installed")
class NvidiaSingletonTests(TestCase):
    def setUp(self) -> None:
        from src.integrations.nvidia import _get_nvidia_llm_cached

        _get_nvidia_llm_cached.cache_clear()

    @patch("src.integrations.nvidia.ChatNVIDIA")
    def test_llm_cached_for_same_model_and_key(self, mock_chat_nvidia) -> None:
        from src.integrations.nvidia import get_nvidia_llm

        mock_chat_nvidia.side_effect = lambda **_: object()

        first = get_nvidia_llm(model_name="model-a", api_key="key-a")
        second = get_nvidia_llm(model_name="model-a", api_key="key-a")

        self.assertIs(first, second)
        mock_chat_nvidia.assert_called_once()

    @patch("src.integrations.nvidia.ChatNVIDIA")
    def test_llm_varies_by_model(self, mock_chat_nvidia) -> None:
        from src.integrations.nvidia import get_nvidia_llm

        mock_chat_nvidia.side_effect = lambda **_: object()

        first = get_nvidia_llm(model_name="model-a", api_key="key-a")
        second = get_nvidia_llm(model_name="model-b", api_key="key-a")

        self.assertIsNot(first, second)
        self.assertEqual(mock_chat_nvidia.call_count, 2)
