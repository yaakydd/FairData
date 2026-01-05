"""
Core Machine Learning Engine
Handles model training, evaluation, and comparison with automated explanations
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
import warnings
import time
from typing import Dict, List, Any, Optional, Tuple
import json

warnings.filterwarnings('ignore')

class MLEngine:
    def __init__(self, dataframe: pd.DataFrame, target_column: str, problem_type: str):
        """
        Initialize ML Engine with dataset
        
        Args:
            dataframe: Input DataFrame
            target_column: Target variable column name
            problem_type: Type of ML problem ('classification', 'regression', 'clustering')
        """
        self.df = dataframe.copy()
        self.target_column = target_column
        self.problem_type = problem_type
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoders = {}
        self.models = {}
        
        # Prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for ML training"""
        # Separate features and target
        if self.target_column in self.df.columns:
            self.X = self.df.drop(columns=[self.target_column])
            self.y = self.df[self.target_column]
        else:
            self.X = self.df
            self.y = None
        
        # Handle categorical variables
        self._encode_categorical()
        
        # Handle missing values
        self._handle_missing_values()
    
    def _encode_categorical(self):
        """Encode categorical variables"""
        for column in self.X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            self.X[column] = le.fit_transform(self.X[column].astype(str))
            self.label_encoders[column] = le
        
        # Encode target for classification
        if self.problem_type == 'classification' and self.y is not None:
            if self.y.dtype in ['object', 'category']:
                le_target = LabelEncoder()
                self.y = le_target.fit_transform(self.y)
                self.label_encoders['target'] = le_target
    
    def _handle_missing_values(self):
        """Handle missing values in features"""
        # For numerical columns, fill with median
        num_cols = self.X.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if self.X[col].isnull().any():
                self.X[col].fillna(self.X[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        cat_cols = self.X.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            if self.X[col].isnull().any():
                self.X[col].fillna(self.X[col].mode()[0], inplace=True)
    
    def analyze_for_ml(self) -> Dict[str, Any]:
        """
        Analyze dataset for ML readiness
        Returns insights and recommendations
        """
        analysis = {
            "dataset_shape": self.df.shape,
            "target_column": self.target_column,
            "problem_type": self.problem_type,
            "features_count": len(self.X.columns),
            "categorical_features": list(self.X.select_dtypes(exclude=[np.number]).columns),
            "numerical_features": list(self.X.select_dtypes(include=[np.number]).columns),
            "issues": [],
            "recommendations": [],
            "model_recommendations": []
        }
        
        # Check for issues
        if self.y is not None:
            if self.problem_type == 'classification':
                class_distribution = pd.Series(self.y).value_counts()
                analysis["class_distribution"] = class_distribution.to_dict()
                
                # Check for class imbalance
                if len(class_distribution) > 0:
                    imbalance_ratio = class_distribution.max() / class_distribution.min()
                    if imbalance_ratio > 10:
                        analysis["issues"].append("Severe class imbalance detected")
                        analysis["recommendations"].append("Consider using class_weight='balanced' or SMOTE")
            
            elif self.problem_type == 'regression':
                analysis["target_statistics"] = {
                    "mean": float(self.y.mean()),
                    "std": float(self.y.std()),
                    "min": float(self.y.min()),
                    "max": float(self.y.max())
                }
        
        # Check for multicollinearity
        corr_matrix = self.X.corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": float(corr_matrix.iloc[i, j])
                    })
        
        if high_corr_pairs:
            analysis["issues"].append("High correlation between features detected")
            analysis["high_correlation_pairs"] = high_corr_pairs[:5]  # Show top 5
            analysis["recommendations"].append("Consider removing highly correlated features")
        
        # Recommend models based on problem type and data size
        n_samples = len(self.X)
        
        if self.problem_type == 'classification':
            if n_samples < 1000:
                analysis["model_recommendations"] = ["Logistic Regression", "Decision Tree", "Random Forest"]
            else:
                analysis["model_recommendations"] = ["Random Forest", "XGBoost", "SVM"]
        
        elif self.problem_type == 'regression':
            if n_samples < 1000:
                analysis["model_recommendations"] = ["Linear Regression", "Decision Tree", "Random Forest"]
            else:
                analysis["model_recommendations"] = ["Random Forest", "XGBoost", "SVR"]
        
        elif self.problem_type == 'clustering':
            analysis["model_recommendations"] = ["KMeans", "DBSCAN"]
        
        return analysis
    
    def train_model(self, model_type: str, test_size: float = 0.2, 
                   random_state: int = 42, **hyperparams) -> Dict[str, Any]:
        """
        Train a machine learning model
        
        Returns:
            Dictionary containing model, metrics, and training info
        """
        # Split data
        if self.y is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state
            )
        else:
            # For clustering, use all data
            self.X_train = self.X
            self.X_test = None
            self.y_train = None
            self.y_test = None
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        if self.X_test is not None:
            self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Select model
        model = self._get_model(model_type, hyperparams)
        
        # Train model
        start_time = time.time()
        
        if self.problem_type == 'clustering':
            model.fit(self.X_train_scaled)
            training_time = time.time() - start_time
            
            # For clustering, calculate inertia
            metrics = {
                "inertia": float(model.inertia_) if hasattr(model, 'inertia_') else None,
                "n_clusters": len(set(model.labels_)) if hasattr(model, 'labels_') else None
            }
            predictions = model.labels_
            
        else:
            model.fit(self.X_train_scaled, self.y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            predictions = y_pred
            
            # Calculate metrics
            metrics = self._calculate_metrics(self.y_test, y_pred)
        
        # Feature importance for tree-based models
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(self.X.columns, model.feature_importances_))
            feature_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Store model
        self.models[model_type] = model
        
        result = {
            "model": model,
            "model_type": model_type,
            "metrics": metrics,
            "training_time": training_time,
            "feature_importance": feature_importance,
            "predictions": predictions,
            "test_size": test_size,
            "hyperparameters": hyperparams
        }
        
        return result
    
    def _get_model(self, model_type: str, hyperparams: Dict) -> Any:
        """Get model instance based on type"""
        model_map = {
            'classification': {
                'logistic_regression': LogisticRegression(**hyperparams),
                'random_forest': RandomForestClassifier(**hyperparams),
                'decision_tree': DecisionTreeClassifier(**hyperparams),
                'svm': SVC(**hyperparams),
                'xgboost': XGBClassifier(**hyperparams),
                'naive_bayes': GaussianNB(**hyperparams)
            },
            'regression': {
                'linear_regression': LinearRegression(**hyperparams),
                'random_forest': RandomForestRegressor(**hyperparams),
                'decision_tree': DecisionTreeRegressor(**hyperparams),
                'svr': SVR(**hyperparams),
                'xgboost': XGBRegressor(**hyperparams)
            },
            'clustering': {
                'kmeans': KMeans(**hyperparams),
                'dbscan': DBSCAN(**hyperparams)
            }
        }
        
        model_key = model_type.lower().replace(" ", "_")
        return model_map[self.problem_type].get(model_key, model_map[self.problem_type]['random_forest'])
    
    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate appropriate metrics based on problem type"""
        if self.problem_type == 'classification':
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            }
        elif self.problem_type == 'regression':
            return {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred)),
                "mae": float(np.mean(np.abs(y_true - y_pred)))
            }
        return {}
    
    def compare_models(self, model_types: List[str]) -> Dict[str, Any]:
        """Compare multiple models side by side"""
        comparison = {
            "models": [],
            "best_model": None,
            "best_score": -float('inf')
        }
        
        for model_type in model_types:
            try:
                result = self.train_model(model_type, test_size=0.2, random_state=42)
                
                model_info = {
                    "model_type": model_type,
                    "metrics": result["metrics"],
                    "training_time": result["training_time"],
                    "hyperparameters": result["hyperparameters"]
                }
                
                comparison["models"].append(model_info)
                
                # Determine best model based on primary metric
                score_key = "accuracy" if self.problem_type == "classification" else "r2"
                current_score = result["metrics"].get(score_key, -float('inf'))
                
                if current_score > comparison["best_score"]:
                    comparison["best_score"] = current_score
                    comparison["best_model"] = model_type
                    comparison["best_model_reason"] = f"Highest {score_key}: {current_score:.3f}"
                    
            except Exception as e:
                print(f"Error training {model_type}: {str(e)}")
                continue
        
        return comparison
    
    def predict(self, model, new_data: pd.DataFrame):
        """Make predictions on new data"""
        # Preprocess new data
        new_data_processed = new_data.copy()
        
        # Apply same preprocessing
        for column, encoder in self.label_encoders.items():
            if column in new_data_processed.columns and column != 'target':
                new_data_processed[column] = encoder.transform(new_data_processed[column].astype(str))
        
        # Scale features
        if self.scaler:
            new_data_scaled = self.scaler.transform(new_data_processed)
        else:
            new_data_scaled = new_data_processed
        
        # Make predictions
        predictions = model.predict(new_data_scaled)
        
        return predictions
    
    def get_test_data(self):
        """Get test data for example predictions"""
        if self.X_test is not None:
            test_df = self.X_test.copy()
            if self.y_test is not None:
                test_df[self.target_column] = self.y_test
            return test_df
        return None
    
    def get_model_options(self, model_type: str) -> Dict[str, Any]:
        """Get configuration options for a model type"""
        options = {
            "hyperparameters": {},
            "recommendations": [],
            "considerations": []
        }
        
        if model_type == "Random Forest":
            options["hyperparameters"] = {
                "n_estimators": "Number of trees (default: 100)",
                "max_depth": "Maximum tree depth (default: None)",
                "min_samples_split": "Min samples to split (default: 2)"
            }
            options["recommendations"] = [
                "Use n_estimators=100-200 for good performance",
                "Set max_depth to prevent overfitting"
            ]
        
        elif model_type == "Logistic Regression":
            options["recommendations"] = [
                "Good for binary classification problems",
                "Fast and interpretable"
            ]
        
        return options