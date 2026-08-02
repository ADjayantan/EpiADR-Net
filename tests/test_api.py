from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

def test_api_root_and_health():
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["service"] == "EpiADR-Net REST Microservice"

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

def test_api_organs_and_adr_classes():
    res_organs = client.get("/organs")
    assert res_organs.status_code == 200
    assert "Liver" in res_organs.json()["supported_organs"]

    res_adr = client.get("/adr-classes")
    assert res_adr.status_code == 200
    assert len(res_adr.json()["meddra_terms"]) == 10

def test_api_predict_endpoint():
    payload = {
        "smiles": "CC(=O)NC1=CC=C(O)C=C1",
        "organ": "Liver",
        "mc_samples": 5
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["conditioned_organ"] == "Liver"
    assert len(data["predictions"]) == 10
    assert "xai_explanation" in data

def test_api_explain_endpoint():
    payload = {
        "smiles": "CC(=O)NC1=CC=C(O)C=C1",
        "top_k": 3
    }
    res = client.post("/explain", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["top_toxic_hotspots"]) <= 3
