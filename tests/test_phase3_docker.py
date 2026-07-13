import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DOCKERFILE = REPO_ROOT / "Dockerfile"
JETSON_DOCKERFILE = REPO_ROOT / "Dockerfile.jetson"
PHASE3_REQUIREMENTS = REPO_ROOT / "requirements.phase3.txt"
PHASE3_DOC = REPO_ROOT / "PHASE3_DOCKER_SUBMISSION.md"
WINDOWS_BUNDLE = REPO_ROOT / "tools" / "create_windows_offline_bundle.ps1"
JETSON_BUNDLE = REPO_ROOT / "tools" / "make_jetson_offline_bundle.sh"
RUN_JETSON = REPO_ROOT / "run_jetson.sh"


class TestPhase3DockerContract(unittest.TestCase):
    def test_submission_dockerfile_uses_required_phase3_base(self):
        text = SUBMISSION_DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn("nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04", text)
        self.assertIn("ARG TRT_VER=10.16.1.11-1+cuda13.2", text)
        self.assertIn("https://download.pytorch.org/whl/cu130", text)
        self.assertIn("torch==2.9.1", text)
        self.assertIn("torchvision==0.24.1", text)
        self.assertIn("torchaudio==2.9.1", text)
        self.assertIn("COPY . .", text)
        self.assertIn("tools/verify_phase3_requirements.py", text)

    def test_phase3_requirements_use_compatible_onnx(self):
        text = PHASE3_REQUIREMENTS.read_text(encoding="utf-8")
        self.assertIn("onnx==1.19.1", text)
        self.assertNotIn("torch==", text)

    def test_jetson_dockerfile_is_separate_and_preserved(self):
        text = JETSON_DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn("nvcr.io/nvidia/pytorch:25.06-py3-igpu", text)
        self.assertIn("tools/verify_jetson_requirements.py", text)

    def test_jetson_helpers_build_dockerfile_jetson(self):
        for path in (WINDOWS_BUNDLE, JETSON_BUNDLE, RUN_JETSON):
            text = path.read_text(encoding="utf-8")
            self.assertIn("Dockerfile.jetson", text)

    def test_phase3_doc_points_to_submission_dockerfile(self):
        text = PHASE3_DOC.read_text(encoding="utf-8")
        self.assertIn("Dockerfile", text)
        self.assertIn("linux/amd64", text)
        self.assertIn("Do not submit `Dockerfile.jetson`", text)


if __name__ == "__main__":
    unittest.main()
