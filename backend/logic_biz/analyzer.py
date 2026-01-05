"""
Data Analyzer Service
Performs exploratory data analysis, statistical summaries, and pattern detection
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple, Union, Literal, cast
import json
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize DataAnalyzer with a DataFrame
        
        Args:
            dataframe: Pandas DataFrame to analyze
        """
        self.df = dataframe.copy()
        
        # Use numpy numeric types directly to avoid complex numbers
        self.numerical_cols = []
        for col in self.df.columns:
            try:
                # Check if column dtype is in numpy's numeric types (excluding complex)
                dtype = self.df[col].dtype
                if (np.issubdtype(dtype, np.number) and 
                    not np.issubdtype(dtype, np.complexfloating)):
                    self.numerical_cols.append(col)
            except:
                continue
        
        # Categorical columns
        self.categorical_cols = self.df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        # Datetime columns - handle carefully
        self.datetime_cols = []
        for col in self.df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    self.datetime_cols.append(col)
                elif pd.api.types.is_timedelta64_dtype(self.df[col]):
                    self.datetime_cols.append(col)
            except:
                continue
        
        self.analysis_history = []
    
    def quick_analysis(self) -> Dict[str, Any]:
        """
        Perform quick analysis on the dataset
        
        Returns:
            Dictionary with basic analysis results
        """
        result = {
            "overview": self._get_dataset_overview(),
            "data_quality": self._assess_data_quality(),
            "statistical_summary": self._get_statistical_summary(),
            "key_insights": self._generate_key_insights(),
            "recommendations": self._generate_recommendations()
        }
        
        # Record analysis
        self.analysis_history.append({
            "type": "quick",
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
        
        return result
    
    def comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis including correlations and patterns
        
        Returns:
            Dictionary with detailed analysis results
        """
        result = {
            "overview": self._get_dataset_overview(),
            "data_quality": self._assess_data_quality(),
            "statistical_summary": self._get_statistical_summary(),
            "correlation_analysis": self._get_correlation_analysis(),
            "pattern_analysis": self._detect_patterns_comprehensive(),
            "key_insights": self._generate_key_insights(),
            "recommendations": self._generate_recommendations()
        }
        
        self.analysis_history.append({
            "type": "comprehensive",
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
        
        return result
    
    def statistical_analysis(self) -> Dict[str, Any]:
        """
        Focus on statistical analysis
        
        Returns:
            Statistical analysis results
        """
        result = {
            "descriptive_stats": self._get_descriptive_statistics(),
            "distribution_analysis": self._analyze_distributions(),
            "outlier_analysis": self._detect_outliers(),
            "normality_tests": self._run_normality_tests()
        }
        
        self.analysis_history.append({
            "type": "statistical",
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
        
        return result
    
    def correlation_analysis(self) -> Dict[str, Any]:
        """
        Perform correlation analysis
        
        Returns:
            Correlation analysis results
        """
        corr_matrix = self.get_correlation_matrix()
        result = {
            "correlation_matrix": corr_matrix.to_dict() if not corr_matrix.empty else {},
            "high_correlations": self.find_high_correlations(0.7),
            "correlation_insights": self._generate_correlation_insights()
        }
        
        return result
    
    def outlier_analysis(self) -> Dict[str, Any]:
        """
        Detect and analyze outliers
        
        Returns:
            Outlier analysis results
        """
        outliers = self._detect_outliers_detailed()
        
        result = {
            "total_outliers": sum(outliers.get("counts", {}).values()),
            "affected_columns": list(outliers.get("counts", {}).keys()),
            "outlier_details": outliers,
            "recommendations": self._generate_outlier_recommendations(outliers)
        }
        
        return result
    
    def _get_dataset_overview(self) -> Dict[str, Any]:
        """Get basic dataset overview"""
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "memory_usage": f"{self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "numerical_columns": self.numerical_cols,
            "categorical_columns": self.categorical_cols,
            "datetime_columns": self.datetime_cols
        }
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess data quality and identify issues"""
        issues = []
        missing_percentages = {}
        quality_score = 100
        
        # Check for missing values
        missing_counts = self.df.isnull().sum()
        total_cells = len(self.df) * len(self.df.columns)
        
        for col, missing in missing_counts.items():
            if missing > 0:
                percent_missing = (missing / len(self.df)) * 100
                missing_percentages[col] = round(percent_missing, 2)
                
                if percent_missing > 50:
                    issues.append(f"Column '{col}' has {percent_missing:.1f}% missing values (critical)")
                    quality_score -= 15
                elif percent_missing > 20:
                    issues.append(f"Column '{col}' has {percent_missing:.1f}% missing values")
                    quality_score -= 10
                elif percent_missing > 5:
                    issues.append(f"Column '{col}' has {percent_missing:.1f}% missing values (minor)")
                    quality_score -= 5
        
        # Check for duplicates
        duplicate_count = self.df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percent = (duplicate_count / len(self.df)) * 100
            issues.append(f"Found {duplicate_count} duplicate rows ({duplicate_percent:.1f}%)")
            quality_score -= 10
        
        # Check for data type issues
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Check if object column might actually be numeric
                try:
                    pd.to_numeric(self.df[col], errors='raise')
                    issues.append(f"Column '{col}' appears to be numeric but stored as text")
                    quality_score -= 5
                except:
                    pass
        
        # Ensure quality score doesn't go below 0
        quality_score = max(0, quality_score)
        
        return {
            "quality_score": round(quality_score),
            "issues": issues,
            "missing_percentages": missing_percentages,
            "duplicate_rows": int(duplicate_count),
            "total_missing_cells": int(missing_counts.sum()),
            "missing_percentage_total": round((missing_counts.sum() / total_cells) * 100, 2) if total_cells > 0 else 0
        }
    
    def _get_statistical_summary(self) -> Dict[str, Any]:
        """Get statistical summary for numerical columns"""
        if not self.numerical_cols:
            return {"message": "No numerical columns found for statistical analysis"}
        
        summary = {}
        for col in self.numerical_cols:
            # Convert to numeric safely
            col_data_clean = self._safe_to_numeric(self.df[col])
            if len(col_data_clean) > 0:
                # Explicitly cast to float to satisfy type checker
                col_values = col_data_clean.values.astype(float)
                summary[col] = {
                    "count": int(len(col_data_clean)),
                    "mean": float(np.mean(col_values)),
                    "std": float(np.std(col_values)),
                    "min": float(np.min(col_values)),
                    "25%": float(np.percentile(col_values, 25)),
                    "median": float(np.median(col_values)),
                    "75%": float(np.percentile(col_values, 75)),
                    "max": float(np.max(col_values)),
                    "range": float(np.max(col_values) - np.min(col_values)),
                    "iqr": float(np.percentile(col_values, 75) - np.percentile(col_values, 25)),
                    "skewness": float(stats.skew(col_values)),
                    "kurtosis": float(stats.kurtosis(col_values))
                }
        
        return summary
    
    def _safe_to_numeric(self, series: pd.Series) -> pd.Series:
        """Safely convert series to numeric, handling complex numbers"""
        try:
            # First try direct conversion
            numeric = pd.to_numeric(series, errors='coerce')
            # Check for complex numbers by checking dtype
            if hasattr(numeric.dtype, 'kind') and numeric.dtype.kind == 'c':
                # Complex numbers found, return empty
                return pd.Series([], dtype=np.float64)
            # Ensure we return float64
            return numeric.astype(np.float64).dropna()
        except Exception:
            return pd.Series([], dtype=np.float64)
    
    def _get_descriptive_statistics(self) -> Dict[str, Any]:
        """Get detailed descriptive statistics"""
        return {
            "numerical_summary": self._get_statistical_summary(),
            "categorical_summary": self._get_categorical_summary(),
            "missing_values_summary": self._get_missing_values_summary()
        }
    
    def _get_categorical_summary(self) -> Dict[str, Any]:
        """Get summary for categorical columns"""
        summary = {}
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()
            summary[col] = {
                "unique_values": int(self.df[col].nunique()),
                "most_common": value_counts.head(5).to_dict(),
                "missing_values": int(self.df[col].isnull().sum())
            }
        return summary
    
    def _get_missing_values_summary(self) -> Dict[str, Any]:
        """Get detailed missing values summary"""
        missing_counts = self.df.isnull().sum()
        return {
            "missing_by_column": missing_counts[missing_counts > 0].to_dict(),
            "total_missing": int(missing_counts.sum()),
            "percentage_missing": round((missing_counts.sum() / (len(self.df) * len(self.df.columns))) * 100, 2) if len(self.df) > 0 else 0
        }
    
    def _analyze_distributions(self) -> Dict[str, Any]:
        """Analyze distributions of numerical columns"""
        distributions = {}
        for col in self.numerical_cols:
            col_data_clean = self._safe_to_numeric(self.df[col])
            if len(col_data_clean) > 10:  # Need enough data
                try:
                    col_values = col_data_clean.values.astype(float)
                    # Calculate histogram
                    hist, bin_edges = np.histogram(col_values, bins='auto')
                    
                    distributions[col] = {
                        "is_normal": self._is_normal_distribution(col_data_clean),
                        "distribution_type": self._identify_distribution_type(col_data_clean),
                        "histogram": {
                            "counts": hist.tolist(),
                            "bin_edges": bin_edges.tolist()
                        },
                        "skewness": float(stats.skew(col_values)),
                        "kurtosis": float(stats.kurtosis(col_values))
                    }
                except Exception:
                    continue
        
        return distributions
    
    def _is_normal_distribution(self, data: pd.Series) -> bool:
        """Check if data follows normal distribution using Shapiro-Wilk test"""
        if len(data) < 3 or len(data) > 5000:
            return False
        
        try:
            # Ensure data is numeric
            data_numeric = self._safe_to_numeric(data)
            if len(data_numeric) < 3:
                return False
            
            data_values = data_numeric.values.astype(float)
            _, p_value = stats.shapiro(data_values)
            return p_value > 0.05
        except Exception:
            return False
    
    def _identify_distribution_type(self, data: pd.Series) -> str:
        """Identify the type of distribution"""
        try:
            # Ensure numeric data
            data_numeric = self._safe_to_numeric(data)
            if len(data_numeric) == 0:
                return "unknown"
            
            data_values = data_numeric.values.astype(float)
            skewness = stats.skew(data_values)
            
            if abs(skewness) < 0.5:
                return "approximately normal"
            elif skewness > 0:
                return "right-skewed"
            else:
                return "left-skewed"
        except Exception:
            return "unknown"
    
    def _detect_outliers(self) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        outliers = {}
        for col in self.numerical_cols:
            col_data_clean = self._safe_to_numeric(self.df[col])
            if len(col_data_clean) > 0:
                col_values = col_data_clean.values.astype(float)
                Q1 = float(np.percentile(col_values, 25))
                Q3 = float(np.percentile(col_values, 75))
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers using numpy for type safety
                outlier_count = int(np.sum((col_values < lower_bound) | (col_values > upper_bound)))
                
                if outlier_count > 0:
                    outliers[col] = {
                        "count": outlier_count,
                        "percentage": float((outlier_count / len(col_values)) * 100),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound)
                    }
        
        return outliers
    
    def _detect_outliers_detailed(self) -> Dict[str, Any]:
        """Detailed outlier detection with multiple methods"""
        outliers_iqr = self._detect_outliers()
        
        # Also use Z-score method for comparison
        outliers_zscore = {}
        for col in self.numerical_cols:
            col_data_clean = self._safe_to_numeric(self.df[col])
            if len(col_data_clean) > 0:
                try:
                    col_values = col_data_clean.values.astype(float)
                    # Use numpy for z-score calculation
                    z_scores = np.abs((col_values - np.mean(col_values)) / np.std(col_values))
                    outlier_count = int(np.sum(z_scores > 3))
                    
                    if outlier_count > 0:
                        outliers_zscore[col] = {
                            "count": outlier_count,
                            "percentage": float((outlier_count / len(col_values)) * 100)
                        }
                except Exception:
                    continue
        
        return {
            "iqr_method": outliers_iqr,
            "zscore_method": outliers_zscore,
            "counts": {col: outliers_iqr.get(col, {}).get("count", 0) for col in self.numerical_cols}
        }
    
    def _run_normality_tests(self) -> Dict[str, Any]:
        """Run normality tests on numerical columns"""
        normality_results = {}
        for col in self.numerical_cols:
            col_data_clean = self._safe_to_numeric(self.df[col])
            if 3 <= len(col_data_clean) <= 5000:
                try:
                    col_values = col_data_clean.values.astype(float)
                    # Shapiro-Wilk test
                    shapiro_stat, shapiro_p = stats.shapiro(col_values)
                    
                    # D'Agostino's K² test
                    k2_stat, k2_p = stats.normaltest(col_values)
                    
                    normality_results[col] = {
                        "shapiro_wilk": {
                            "statistic": float(shapiro_stat),
                            "p_value": float(shapiro_p),
                            "is_normal": shapiro_p > 0.05
                        },
                        "dagostino_k2": {
                            "statistic": float(k2_stat),
                            "p_value": float(k2_p),
                            "is_normal": k2_p > 0.05
                        }
                    }
                except Exception:
                    continue
        
        return normality_results
    
    def get_correlation_matrix(self, method: Literal['pearson', 'spearman', 'kendall'] = "pearson") -> pd.DataFrame:
        """
        Calculate correlation matrix
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
        
        Returns:
            Correlation matrix DataFrame
        """
        # Use only numerical columns for correlation
        if len(self.numerical_cols) < 2:
            return pd.DataFrame()
        
        # Create a clean numeric dataframe
        numeric_data = {}
        for col in self.numerical_cols:
            numeric_data[col] = self._safe_to_numeric(self.df[col])
        
        numeric_df = pd.DataFrame(numeric_data)
        
        if len(numeric_df.columns) < 2:
            return pd.DataFrame()
        
        return numeric_df.corr(method=method)
    
    def find_high_correlations(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find highly correlated feature pairs
        
        Args:
            threshold: Absolute correlation threshold
        
        Returns:
            List of high correlation pairs
        """
        high_correlations = []
        corr_matrix = self.get_correlation_matrix()
        
        if corr_matrix.empty:
            return high_correlations
        
        # Get upper triangle of correlation matrix
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find pairs with correlation above threshold
        for col in upper_triangle.columns:
            for idx in upper_triangle.index:
                corr_value = upper_triangle.loc[idx, col]
                if pd.notna(corr_value) and abs(float(corr_value)) >= threshold:
                    high_correlations.append({
                        "feature1": idx,
                        "feature2": col,
                        "correlation": float(corr_value),
                        "type": "positive" if corr_value > 0 else "negative"
                    })
        
        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return high_correlations
    
    def _get_correlation_analysis(self) -> Dict[str, Any]:
        """Perform correlation analysis"""
        corr_matrix = self.get_correlation_matrix()
        
        return {
            "matrix_size": corr_matrix.shape,
            "high_correlations": self.find_high_correlations(0.7),
            "strongest_correlation": self._find_strongest_correlation(corr_matrix),
            "method_used": "pearson"
        }
    
    def _find_strongest_correlation(self, corr_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Find the strongest correlation in the matrix"""
        if corr_matrix.empty:
            return {}
        
        # Get absolute values for strongest correlation
        abs_corr = corr_matrix.abs()
        
        # Set diagonal to NaN to ignore self-correlations
        np.fill_diagonal(abs_corr.values, np.nan)
        
        # Find max correlation
        max_corr_val = abs_corr.max().max()
        
        if pd.isna(max_corr_val):
            return {}
        
        max_corr = float(max_corr_val)
        
        # Find the pair with this correlation
        pairs = np.where(corr_matrix.values == max_corr)
        if len(pairs[0]) > 0:
            feature1 = corr_matrix.index[pairs[0][0]]
            feature2 = corr_matrix.columns[pairs[1][0]]
            
            return {
                "feature1": feature1,
                "feature2": feature2,
                "correlation": max_corr,
                "type": "positive" if corr_matrix.loc[feature1, feature2] > 0 else "negative"
            }
        
        return {}
    
    def _detect_patterns_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive pattern detection"""
        patterns = {
            "temporal_patterns": self._detect_temporal_patterns(),
            "categorical_patterns": self._detect_categorical_patterns(),
            "numerical_patterns": self._detect_numerical_patterns(),
            "missing_patterns": self._detect_missing_patterns()
        }
        
        return patterns
    
    def detect_patterns(self, pattern_type: str = "all") -> List[str]:
        """
        Detect patterns in the dataset
        
        Args:
            pattern_type: Type of patterns to detect
        
        Returns:
            List of detected patterns
        """
        patterns = []
        
        if pattern_type in ["all", "temporal"]:
            temporal_patterns = self._detect_temporal_patterns()
            patterns.extend(temporal_patterns)
        
        if pattern_type in ["all", "categorical"]:
            categorical_patterns = self._detect_categorical_patterns()
            patterns.extend(categorical_patterns)
        
        if pattern_type in ["all", "numerical"]:
            numerical_patterns = self._detect_numerical_patterns()
            patterns.extend(numerical_patterns)
        
        if pattern_type in ["all", "missing"]:
            missing_patterns = self._detect_missing_patterns()
            patterns.extend(missing_patterns)
        
        return patterns
    
    def _detect_temporal_patterns(self) -> List[str]:
        """Detect temporal patterns in datetime columns"""
        patterns = []
        
        for col in self.datetime_cols:
            if len(self.df[col].dropna()) > 1:
                # Check for trends
                try:
                    sorted_dates = self.df[col].dropna().sort_values()
                    time_diffs = sorted_dates.diff().dropna()
                    
                    if len(time_diffs) > 0:
                        # Handle timedelta safely
                        avg_gap = time_diffs.mean()
                        # Convert to total seconds for comparison
                        if hasattr(avg_gap, 'total_seconds'):
                            avg_gap_seconds = avg_gap.total_seconds()
                        else:
                            # If it's already a numeric type (like int or float)
                            avg_gap_seconds = float(avg_gap)
                        
                        avg_gap_days = avg_gap_seconds / (24 * 3600)
                        
                        if 0.9 < avg_gap_days < 1.1:
                            patterns.append(f"Column '{col}' appears to have daily frequency")
                        elif 6.5 < avg_gap_days < 7.5:
                            patterns.append(f"Column '{col}' appears to have weekly frequency")
                        elif 28 < avg_gap_days < 32:
                            patterns.append(f"Column '{col}' appears to have monthly frequency")
                except Exception:
                    continue
        
        return patterns
    
    def _detect_categorical_patterns(self) -> List[str]:
        """Detect patterns in categorical columns"""
        patterns = []
        
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()
            total = len(self.df[col].dropna())
            
            if total > 0:
                # Check for dominant categories
                top_category_percent = (value_counts.iloc[0] / total) * 100
                if top_category_percent > 80:
                    patterns.append(f"Column '{col}' is dominated by one category ({top_category_percent:.1f}%)")
                
                # Check for uniform distribution
                if len(value_counts) > 5:
                    uniform_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                    if uniform_ratio < 2:
                        patterns.append(f"Column '{col}' has relatively uniform distribution")
        
        return patterns
    
    def _detect_numerical_patterns(self) -> List[str]:
        """Detect patterns in numerical columns"""
        patterns = []
        
        for col in self.numerical_cols:
            col_data_clean = self._safe_to_numeric(self.df[col])
            if len(col_data_clean) > 10:
                col_values = col_data_clean.values.astype(float)
                # Check for round numbers (common in human-entered data)
                round_numbers = np.sum(col_values % 1 == 0)
                if (round_numbers / len(col_values)) > 0.8:
                    patterns.append(f"Column '{col}' contains mostly whole numbers")
                
                # Check for constant values
                if np.unique(col_values).shape[0] == 1:
                    patterns.append(f"Column '{col}' has constant value")
                elif np.unique(col_values).shape[0] == 2:
                    patterns.append(f"Column '{col}' appears to be binary")
        
        return patterns
    
    def _detect_missing_patterns(self) -> List[str]:
        """Detect patterns in missing values"""
        patterns = []
        
        # Check if missing values are correlated
        missing_matrix = self.df.isnull()
        missing_sum = missing_matrix.sum().sum()
        
        if missing_sum > 0:
            # Check if missing values occur together
            try:
                missing_corr = missing_matrix.corr()
                
                for i, col1 in enumerate(missing_corr.columns):
                    for j, col2 in enumerate(missing_corr.columns):
                        if i < j and missing_corr.loc[col1, col2] > 0.8:
                            patterns.append(f"Missing values in '{col1}' and '{col2}' often occur together")
            except Exception:
                pass
        
        return patterns
    
    def _generate_key_insights(self) -> List[str]:
        """Generate key insights from the analysis"""
        insights = []
        
        # Dataset size insight
        if len(self.df) > 10000:
            insights.append("Large dataset suitable for complex modeling")
        elif len(self.df) < 100:
            insights.append("Small dataset - consider collecting more data for robust analysis")
        
        # Missing values insight
        total_missing = self.df.isnull().sum().sum()
        total_cells = len(self.df) * len(self.df.columns)
        missing_percent = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        
        if missing_percent > 20:
            insights.append(f"High missing data ({missing_percent:.1f}%) - consider imputation or removal")
        elif missing_percent > 5:
            insights.append(f"Moderate missing data ({missing_percent:.1f}%) - may need treatment")
        else:
            insights.append("Minimal missing data - good data quality")
        
        # Numerical columns insight
        if len(self.numerical_cols) > 10:
            insights.append(f"Many numerical features ({len(self.numerical_cols)}) - consider dimensionality reduction")
        elif len(self.numerical_cols) > 0:
            insights.append(f"Suitable numerical features ({len(self.numerical_cols)}) for statistical analysis")
        
        # Categorical columns insight
        if len(self.categorical_cols) > 5:
            insights.append(f"Multiple categorical variables ({len(self.categorical_cols)}) - may need encoding for ML")
        
        # Correlation insight
        high_corrs = self.find_high_correlations(0.8)
        if high_corrs:
            insights.append(f"Found {len(high_corrs)} highly correlated feature pairs - consider removing redundancy")
        
        # Outlier insight
        outliers = self._detect_outliers()
        outlier_cols = [col for col, info in outliers.items() if info.get("percentage", 0) > 5]
        if outlier_cols:
            insights.append(f"Outliers detected in {len(outlier_cols)} columns - may need treatment")
        
        return insights
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Data quality recommendations
        quality = self._assess_data_quality()
        if quality["quality_score"] < 70:
            recommendations.append("Address data quality issues before further analysis")
        
        if quality["duplicate_rows"] > 0:
            recommendations.append(f"Remove {quality['duplicate_rows']} duplicate rows")
        
        # Missing data recommendations
        missing_cols = [col for col, perc in quality["missing_percentages"].items() if perc > 20]
        if missing_cols:
            recommendations.append(f"Consider removing columns with >20% missing values: {', '.join(missing_cols[:3])}")
        
        # Outlier recommendations
        outliers = self._detect_outliers()
        outlier_cols = [col for col, info in outliers.items() if info.get("percentage", 0) > 5]
        if outlier_cols:
            recommendations.append(f"Investigate outliers in columns: {', '.join(outlier_cols[:3])}")
        
        # Correlation recommendations
        high_corrs = self.find_high_correlations(0.9)
        if high_corrs:
            recommendations.append("Remove highly correlated features to avoid multicollinearity")
        
        # General recommendations
        if len(self.numerical_cols) > 0:
            recommendations.append("Consider scaling numerical features for machine learning")
        
        if len(self.categorical_cols) > 0:
            recommendations.append("Encode categorical variables before modeling")
        
        recommendations.append("Split data into training and testing sets for model validation")
        
        return recommendations
    
    def _generate_correlation_insights(self) -> List[Dict[str, Any]]:
        """Generate insights from correlation analysis"""
        insights = []
        high_corrs = self.find_high_correlations(0.7)
        
        for corr in high_corrs[:5]:  # Limit to top 5
            insights.append({
                "features": f"{corr['feature1']} and {corr['feature2']}",
                "correlation": corr["correlation"],
                "type": corr["type"],
                "interpretation": f"Strong {corr['type']} relationship: as one increases, the other tends to {'increase' if corr['type'] == 'positive' else 'decrease'}"
            })
        
        return insights
    
    def _generate_outlier_recommendations(self, outliers: Dict[str, Any]) -> List[str]:
        """Generate recommendations for outlier treatment"""
        recommendations = []
        
        iqr_outliers = outliers.get("iqr_method", {})
        for col, info in iqr_outliers.items():
            outlier_percentage = info.get("percentage", 0)
            if outlier_percentage > 10:
                recommendations.append(f"Column '{col}' has {outlier_percentage:.1f}% outliers - consider robust statistical methods")
            elif outlier_percentage > 5:
                recommendations.append(f"Column '{col}' has moderate outliers ({outlier_percentage:.1f}%) - investigate if they're valid")
        
        if not recommendations:
            recommendations.append("Outliers are minimal - no treatment needed")
        
        return recommendations
    
    def get_column_basic_stats(self, column_name: str) -> Dict[str, Any]:
        """
        Get basic statistics for a specific column
        
        Args:
            column_name: Name of the column
        
        Returns:
            Basic statistics
        """
        if column_name not in self.df.columns:
            return {"error": f"Column '{column_name}' not found"}
        
        col_data = self.df[column_name].dropna()
        
        if column_name in self.numerical_cols:
            # Convert to numeric safely
            col_data_clean = self._safe_to_numeric(self.df[column_name])
            if len(col_data_clean) > 0:
                col_values = col_data_clean.values.astype(float)
                return {
                    "type": "numerical",
                    "count": int(len(col_values)),
                    "mean": float(np.mean(col_values)),
                    "std": float(np.std(col_values)),
                    "min": float(np.min(col_values)),
                    "max": float(np.max(col_values)),
                    "median": float(np.median(col_values)),
                    "missing": int(self.df[column_name].isnull().sum()),
                    "missing_percentage": float((self.df[column_name].isnull().sum() / len(self.df)) * 100) if len(self.df) > 0 else 0
                }
            else:
                return {
                    "type": "numerical",
                    "error": "Cannot convert column to numeric values",
                    "count": 0,
                    "missing": int(self.df[column_name].isnull().sum()),
                    "missing_percentage": float((self.df[column_name].isnull().sum() / len(self.df)) * 100) if len(self.df) > 0 else 0
                }
        else:
            # Categorical column
            value_counts = self.df[column_name].value_counts()
            return {
                "type": "categorical",
                "count": int(len(col_data)),
                "unique_values": int(self.df[column_name].nunique()),
                "most_common": value_counts.head(3).to_dict(),
                "missing": int(self.df[column_name].isnull().sum()),
                "missing_percentage": float((self.df[column_name].isnull().sum() / len(self.df)) * 100) if len(self.df) > 0 else 0
            }
    
    def get_column_detailed_stats(self, column_name: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a specific column
        
        Args:
            column_name: Name of the column
        
        Returns:
            Detailed statistics
        """
        basic_stats = self.get_column_basic_stats(column_name)
        
        if basic_stats.get("type") == "numerical" and "error" not in basic_stats:
            col_data_clean = self._safe_to_numeric(self.df[column_name])
            
            if len(col_data_clean) > 0:
                col_values = col_data_clean.values.astype(float)
                # Calculate percentiles
                percentiles = {}
                for p in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
                    percentiles[f"p{p*100:.0f}"] = float(np.percentile(col_values, p * 100))
                
                # Detect outliers
                Q1 = float(np.percentile(col_values, 25))
                Q3 = float(np.percentile(col_values, 75))
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = int(np.sum((col_values < lower_bound) | (col_values > upper_bound)))
                
                basic_stats.update({
                    "percentiles": percentiles,
                    "skewness": float(stats.skew(col_values)),
                    "kurtosis": float(stats.kurtosis(col_values)),
                    "outliers": {
                        "count": outlier_count,
                        "percentage": float((outlier_count / len(col_values)) * 100),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound)
                    }
                })
        
        return basic_stats
    
    def get_column_distribution(self, column_name: str) -> Dict[str, Any]:
        """
        Get distribution information for a column
        
        Args:
            column_name: Name of the column
        
        Returns:
            Distribution information
        """
        if column_name not in self.df.columns:
            return {"error": f"Column '{column_name}' not found"}
        
        col_data = self.df[column_name].dropna()
        
        if column_name in self.numerical_cols:
            # Convert to numeric safely
            col_data_clean = self._safe_to_numeric(self.df[column_name])
            if len(col_data_clean) > 10:
                col_values = col_data_clean.values.astype(float)
                hist, bin_edges = np.histogram(col_values, bins='auto')
                
                return {
                    "type": "numerical",
                    "is_normal": self._is_normal_distribution(col_data_clean),
                    "distribution_type": self._identify_distribution_type(col_data_clean),
                    "histogram": {
                        "counts": hist.tolist(),
                        "bin_edges": bin_edges.tolist()
                    },
                    "skewness": float(stats.skew(col_values)),
                    "kurtosis": float(stats.kurtosis(col_values))
                }
            else:
                return {
                    "type": "numerical",
                    "error": "Not enough data for distribution analysis"
                }
        else:
            # Categorical column
            value_counts = self.df[column_name].value_counts()
            return {
                "type": "categorical",
                "value_counts": value_counts.to_dict()
            }  