from fastapi import APIRouter
from pydantic import BaseModel
from app.models.predictors import get_top_k_predictions_with_confidence
from app.utils.session_manager import append_symptoms, reset_session
from app.core.config import DEFAULT_TOP_K

router = APIRouter()

class PredictRequest(BaseModel):
    session_id: str
    symptom_text: str
    top_k: int = DEFAULT_TOP_K

@router.post("/predict")
def predict_symptoms(payload: PredictRequest):
    updated_text = append_symptoms(payload.session_id, payload.symptom_text)
    predictions, disclaimer = get_top_k_predictions_with_confidence(updated_text, k=payload.top_k)

    return {
        "session_id": payload.session_id,
        "aggregated_symptoms": updated_text,
        "predictions": predictions,
        "disclaimer": disclaimer
    }

@router.post("/reset")
def reset_session_route(payload: PredictRequest):
    reset_session(payload.session_id)
    return {"message": f"Session '{payload.session_id}' has been reset."}

@router.get("/health")
def health_check():
    return {"status": "ok"}
