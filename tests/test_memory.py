import tempfile
import unittest
from pathlib import Path

from fomo.memory.embeddings import EmbeddingUnavailable, LocalSentenceTransformerEmbedder
from fomo.memory.long_term import LongTermMemory
from fomo.memory.retrieval import MemoryRetriever
from fomo.memory.short_term import BoundedContext
from fomo.memory.vector_store import SQLiteVectorStore, cosine_similarity


class BoundedContextTests(unittest.TestCase):
    def test_evicts_oldest_messages_to_obey_both_limits(self):
        context = BoundedContext(max_messages=2, max_characters=8)
        context.append("user", "one")
        context.append("assistant", "two")
        context.append("user", "three")
        self.assertEqual(
            context.snapshot(),
            [{"role": "assistant", "content": "two"}, {"role": "user", "content": "three"}],
        )
        self.assertLessEqual(context.character_count, 8)

    def test_preserves_system_message_and_validates_inputs(self):
        context = BoundedContext(max_messages=1, max_characters=12, system_message="policy")
        context.append("user", "hello")
        context.append("assistant", "ok")
        self.assertEqual(context.snapshot()[0], {"role": "system", "content": "policy"})
        self.assertEqual(context.snapshot()[-1]["content"], "ok")
        with self.assertRaises(ValueError):
            context.append("system", "not allowed")
        with self.assertRaises(ValueError):
            context.append("user", "too long for system and message")


class PersistentMemoryTests(unittest.TestCase):
    def test_long_term_memories_are_scoped_and_deletable(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = LongTermMemory(Path(directory) / "memories.sqlite3")
            first = memory.add("alice", "prefers concise answers", {"kind": "preference"})
            memory.add("bob", "private memory")
            self.assertEqual(memory.list("alice")[0]["id"], first)
            self.assertEqual(memory.list("alice")[0]["metadata"], {"kind": "preference"})
            self.assertFalse(memory.delete("bob", first))
            self.assertTrue(memory.delete("alice", first))
            self.assertEqual(memory.delete_user("bob"), 1)

    def test_vector_store_ranks_with_cosine_similarity_and_isolates_users(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteVectorStore(Path(directory) / "vectors.sqlite3")
            store.add("alice", "same direction", [3, 0], item_id="top")
            store.add("alice", "orthogonal", [0, 1], item_id="middle")
            store.add("alice", "opposite", [-1, 0], item_id="bottom")
            store.add("bob", "other user", [1, 0], item_id="private")
            matches = store.search("alice", [1, 0])
            self.assertEqual([item["id"] for item in matches], ["top", "middle", "bottom"])
            self.assertAlmostEqual(matches[0]["score"], 1)
            self.assertAlmostEqual(matches[1]["score"], 0)
            self.assertAlmostEqual(matches[2]["score"], -1)
            self.assertTrue(store.delete("alice", "top"))
            self.assertFalse(store.delete("alice", "private"))

    def test_cosine_similarity_rejects_invalid_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1, 1], [1, 0]), 2 ** -0.5)
        self.assertAlmostEqual(
            cosine_similarity([1e308, 1e308], [1e308, 0]), 2 ** -0.5
        )
        for left, right in (([], []), ([1], [1, 2]), ([0], [1]), ([float("nan")], [1])):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                cosine_similarity(left, right)

    def test_retrieval_uses_injected_embedder_without_fake_production_fallback(self):
        class TestEmbedder:
            def embed(self, text):
                return [1.0, 0.0] if "FOMO" in text else [0.0, 1.0]

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteVectorStore(Path(directory) / "vectors.sqlite3")
            retrieval = MemoryRetriever(TestEmbedder(), store)
            retrieval.remember("alice", "FOMO model", {"source": "test fixture"})
            self.assertEqual(
                retrieval.retrieve("alice", "FOMO query")[0]["content"], "FOMO model"
            )

    def test_missing_embedding_model_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "not-installed"
            with self.assertRaisesRegex(EmbeddingUnavailable, "does not exist"):
                LocalSentenceTransformerEmbedder(missing)

    def test_invalid_and_cross_user_queries_are_rejected_or_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteVectorStore(Path(directory) / "vectors.sqlite3")
            store.add("alice", "content", [1.0, 0.0])
            self.assertEqual(store.search("bob", [1.0, 0.0]), [])
            with self.assertRaises(ValueError):
                store.search("alice", [1.0, 0.0, 0.0])
            with self.assertRaises(ValueError):
                store.search("alice", [1.0, 0.0], limit=0)


if __name__ == "__main__":
    unittest.main()