import os
import sys
import tempfile
import unittest

# Ensure we can import ml/api.py as a module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class ModelPackagesApiTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "registry.sqlite")
        self.packages_dir = os.path.join(self.tmpdir.name, "packages")

        os.environ["TRADER_REGISTRY_DB_PATH"] = self.db_path
        os.environ["MODEL_PACKAGES_DIR"] = self.packages_dir

        # Import after env vars so api.py picks them up
        from api import app  # noqa: WPS433
        from fastapi.testclient import TestClient  # noqa: WPS433

        self.client = TestClient(app)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_list_and_upload(self):
        # default package must exist
        r = self.client.get("/model-packages")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(any(p["id"] == "ppo_v1" for p in data))

        files = {
            "forecast_model": ("forecast_model.pt", b"dummy-pt", "application/octet-stream"),
            "ppo_model": ("ppo_policy.zip", b"dummy-zip", "application/zip"),
        }
        r2 = self.client.post("/model-packages/upload", data={"name": "My Test Model"}, files=files)
        self.assertEqual(r2.status_code, 200)
        created = r2.json()
        self.assertIn("id", created)

        rid = created["id"]
        r3 = self.client.get("/model-packages")
        self.assertEqual(r3.status_code, 200)
        data3 = r3.json()
        self.assertTrue(any(p["id"] == rid for p in data3))

        # files should be saved
        forecast_path = os.path.join(self.packages_dir, rid, "forecast_model.pt")
        ppo_path = os.path.join(self.packages_dir, rid, "ppo_policy.zip")
        self.assertTrue(os.path.exists(forecast_path))
        self.assertTrue(os.path.exists(ppo_path))


if __name__ == "__main__":
    unittest.main()
