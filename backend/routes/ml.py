"""
Machine Learning API Endpoints
Handles ML model training, prediction, and evaluation with user-friendly explanations
"""
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import json
import tempfile
import joblib
import os
from datetime import datetime

from services.ml_engine import MLEngine
from utils.explanations import explain_ml_results, suggest_next_ml_steps
from utils.file_handler import save_dataset, load_dataset

router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# In-memory storage for user sessions (in production, use Redis/Database)
ml_sessions = {}

@router.post("/initialize")
async def initialize_ml_session(
    dataset_id: str = Form(...),
    target_column: str = Form(...),
    problem_type: str = Form(...)  # 'classification', 'regression', 'clustering'
):
    """
    Initialize an ML session with user's dataset and target variable
    Returns ML-specific analysis and suggestions
    """
    try:
        # Load the dataset
        df = load_dataset(dataset_id)
        
        if target_column not in df.columns:
            return JSONResponse({
                "status": "error",
                "message": f"Target column '{target_column}' not found in dataset",
                "available_columns": list(df.columns)
            })
        
        # Initialize ML Engine
        ml_engine = MLEngine(df, target_column, problem_type)
        
        # Generate initial ML analysis
        analysis = ml_engine.analyze_for_ml()
        
        # Create session
        session_id = f"ml_{datetime.now().timestamp()}_{dataset_id}"
        ml_sessions[session_id] = {
            "engine": ml_engine,
            "dataset_id": dataset_id,
            "target_column": target_column,
            "problem_type": problem_type,
            "models_trained": [],
            "current_model": None
        }
        
        # Provide user-friendly explanations
        explanations = {
            "data_assessment": explain_ml_results("data_assessment", analysis),
            "recommended_models": explain_ml_results("model_recommendations", analysis.get('model_recommendations', [])),
            "potential_issues": explain_ml_results("potential_issues", analysis.get('issues', [])),
            "next_steps": suggest_next_ml_steps(analysis)
        }
        
        return JSONResponse({
            "status": "success",
            "session_id": session_id,
            "analysis": analysis,
            "explanations": explanations,
            "message": "ML session initialized successfully. Review the analysis below before proceeding."
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train")
async def train_model(
    session_id: str = Form(...),
    model_type: str = Form(...),
    test_size: float = Form(0.2),
    random_state: int = Form(42),
    hyperparameters: Optional[str] = Form(None)
):
    """
    Train a selected ML model with user confirmation
    """
    try:
        if session_id not in ml_sessions:
            raise HTTPException(status_code=404, detail="ML session not found")
        
        session = ml_sessions[session_id]
        ml_engine = session["engine"]
        
        # Parse hyperparameters if provided
        params = json.loads(hyperparameters) if hyperparameters else {}
        
        # Get model options explanation before training
        model_options = ml_engine.get_model_options(model_type)
        
        # Train the model
        training_result = ml_engine.train_model(
            model_type=model_type,
            test_size=test_size,
            random_state=random_state,
            **params
        )
        
        # Store trained model in session
        session["current_model"] = training_result
        session["models_trained"].append({
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "metrics": training_result["metrics"]
        })
        
        # Generate user-friendly explanations
        explanations = {
            "training_summary": explain_ml_results("training_summary", training_result),
            "model_performance": explain_ml_results("model_performance", training_result["metrics"]),
            "feature_importance": explain_ml_results("feature_importance", training_result.get("feature_importance", [])),
            "what_this_means": explain_ml_results("interpretation", training_result["metrics"])
        }
        
        # Provide actionable recommendations
        recommendations = []
        if training_result["metrics"].get("accuracy", 0) < 0.7:
            recommendations.append("Consider feature engineering or trying different models")
        if training_result.get("training_time", 0) > 10:
            recommendations.append("Model training is slow. Consider dimensionality reduction")
        
        return JSONResponse({
            "status": "success",
            "model_trained": model_type,
            "metrics": training_result["metrics"],
            "explanations": explanations,
            "recommendations": recommendations,
            "downloadable": {
                "model": f"/ml/download/model/{session_id}",
                "predictions": f"/ml/download/predictions/{session_id}"
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare")
async def compare_models(
    session_id: str = Form(...),
    models: List[str] = Form(...)
):
    """
    Compare multiple ML models side by side
    """
    try:
        if session_id not in ml_sessions:
            raise HTTPException(status_code=404, detail="ML session not found")
        
        session = ml_sessions[session_id]
        ml_engine = session["engine"]
        
        comparison_results = ml_engine.compare_models(models)
        
        # Generate comparison explanation
        explanations = explain_ml_results("model_comparison", comparison_results)
        
        return JSONResponse({
            "status": "success",
            "comparison": comparison_results,
            "explanations": explanations,
            "recommended_model": comparison_results.get("best_model"),
            "reason": comparison_results.get("best_model_reason")
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
async def make_predictions(
    session_id: str = Form(...),
    input_data: Optional[UploadFile] = File(None),
    use_example: bool = Form(False)
):
    """
    Make predictions using the trained model
    Accepts new data file or uses test data
    """
    try:
        if session_id not in ml_sessions:
            raise HTTPException(status_code=404, detail="ML session not found")
        
        session = ml_sessions[session_id]
        ml_engine = session["engine"]
        current_model = session.get("current_model")
        
        if not current_model:
            raise HTTPException(status_code=400, detail="No model trained yet. Please train a model first.")
        
        if input_data:
            # Read uploaded prediction data
            content = await input_data.read()
            df_new = pd.read_csv(pd.io.common.BytesIO(content))
        elif use_example:
            # Use test split for example predictions
            df_new = ml_engine.get_test_data()
        else:
            raise HTTPException(status_code=400, detail="Either upload data or select use_example")
        
        # Make predictions
        predictions = ml_engine.predict(current_model["model"], df_new)
        
        # Add predictions to dataframe
        if isinstance(predictions, np.ndarray):
            df_new["predictions"] = predictions
        else:
            df_new["predictions"] = predictions.tolist()
        
        # Save predictions
        pred_file = save_dataset(df_new, f"predictions_{session_id}")
        
        # Explain predictions
        explanations = explain_ml_results("predictions", {
            "sample_predictions": df_new[["predictions"]].head().to_dict(),
            "statistics": {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions))
            }
        })
        
        return JSONResponse({
            "status": "success",
            "predictions_count": len(predictions),
            "sample_predictions": df_new[["predictions"]].head().to_dict(orient="records"),
            "explanations": explanations,
            "download_url": f"/download/{pred_file}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/model/{session_id}")
async def download_model(session_id: str):
    """
    Download trained model as pickle file
    """
    if session_id not in ml_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = ml_sessions[session_id]
    current_model = session.get("current_model")
    
    if not current_model:
        raise HTTPException(status_code=400, detail="No model available for download")
    
    # Save model temporarily
    model_path = f"temp_model_{session_id}.pkl"
    joblib.dump(current_model["model"], model_path)
    
    # In production, use proper file serving
    return FileResponse(
        path=model_path,
        filename=f"model_{session['model_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
        media_type='application/octet-stream'
    )

@router.get("/session/{session_id}")
async def get_session_status(session_id: str):
    """
    Get current ML session status
    """
    if session_id not in ml_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = ml_sessions[session_id]
    return {
        "session_id": session_id,
        "dataset_id": session["dataset_id"],
        "target_column": session["target_column"],
        "problem_type": session["problem_type"],
        "models_trained": session["models_trained"],
        "current_model": session["current_model"]["model_type"] if session["current_model"] else None
    }

@router.post("/explain/{explanation_type}")
async def get_explanation(
    explanation_type: str,
    data: Dict[str, Any]
):
    """
    Get plain English explanations for ML concepts
    """
    try:
        explanation = explain_ml_results(explanation_type, data)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot explain: {str(e)}")