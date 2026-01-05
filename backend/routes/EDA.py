import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy.stats import linregress
from scipy import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, KBinsDiscretizer, PolynomialFeatures, FunctionTransformer

class DataAnalyzer:
    """
    Performs comprehensive Exploratory Data Analysis (EDA).
    
    Plain English: This is your data detective - it investigates
    your data and finds patterns, relationships, and insights.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
    def generate_full_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive EDA report.
        
        Plain English: A complete health checkup for your data.
        """
        
        report = {
            "dataset_overview": self._get_overview(),
            "column_analysis": self._analyze_columns(),
            "statistical_summary": self._get_statistics(),
            "correlations": self._get_correlations(),
            "distribution_analysis": self._analyze_distributions(),
            "category_analysis": self._analyze_categories(),
            "data_quality_score": self._calculate_quality_score(),
            "insights": self._generate_insights(),
            "ml_readiness": self._assess_ml_readiness()
        }
        
        return report
    
    def _get_overview(self) -> Dict[str, Any]:
        """Basic dataset information"""
        
        return {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "numeric_columns": len(self.numeric_cols),
            "categorical_columns": len(self.categorical_cols),
            "datetime_columns": len(self.datetime_cols),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / (1024**2),
            "column_names": list(self.df.columns),
            "explanation": (
                f"Your dataset is like a spreadsheet with {len(self.df)} rows (records) "
                f"and {len(self.df.columns)} columns (fields). "
                f"\n\n Breakdown:"
                f"\n• {len(self.numeric_cols)} numeric columns (numbers you can calculate with)"
                f"\n• {len(self.categorical_cols)} categorical columns (groups/categories)"
                f"\n• {len(self.datetime_cols)} date/time columns"
                f"\n\nThink of it like an address book with {len(self.df)} contacts, "
                f"each with {len(self.df.columns)} pieces of information."
            )
        }
    
    def _analyze_columns(self) -> List[Dict[str, Any]]:
        """
        Detailed analysis of each column.
        
        Plain English: A profile for each column - what it contains,
        how it behaves, and what makes it unique.
        """
        
        column_profiles = []
        
        for col in self.df.columns:
            profile = {
                "name": col,
                "data_type": str(self.df[col].dtype),
                "non_null_count": int(self.df[col].count()),
                "null_count": int(self.df[col].isnull().sum()),
                "null_percentage": round(float(self.df[col].isnull().sum() / len(self.df)) * 100, 2),
                "unique_values": int(self.df[col].nunique()),
                "uniqueness_ratio": round(float(self.df[col].nunique() / len(self.df)), 4)
            }
            
            # Type-specific analysis
            if col in self.numeric_cols:
                profile.update(self._analyze_numeric_column(col))
            elif col in self.categorical_cols:
                profile.update(self._analyze_categorical_column(col))
            elif col in self.datetime_cols:
                profile.update(self._analyze_datetime_column(col))
            
            column_profiles.append(profile)
        
        return column_profiles
    
    def _analyze_numeric_column(self, col: str) -> Dict[str, Any]:
        """Analyze a numeric column in detail"""
        
        col_data = self.df[col].dropna()
        
        if len(col_data) == 0:
            return {
                "column_type": "numeric",
                "explanation": f"The '{col}' column is entirely empty - no values to analyze!"
            }
        
        # Calculate statistics - convert to float to handle complex numbers
        try:
            col_data_float = pd.to_numeric(col_data, errors='coerce')
            
            # Convert statistics to numbers safely
            mean_val = self._convert_to_number(col_data_float.mean())
            median_val = self._convert_to_number(col_data_float.median())
            std_val = self._convert_to_number(col_data_float.std())
            min_val = self._convert_to_number(col_data_float.min())
            max_val = self._convert_to_number(col_data_float.max())
            
            # Check for skewness
            skewness_val = self._convert_to_number(col_data_float.skew())
            
            # Detect outliers using IQR
            Q1_val = self._convert_to_number(col_data_float.quantile(0.25))
            Q3_val = self._convert_to_number(col_data_float.quantile(0.75))
            IQR = Q3_val - Q1_val
            outliers = col_data_float[(col_data_float < Q1_val - 1.5*IQR) | (col_data_float > Q3_val + 1.5*IQR)]
            
        except (TypeError, ValueError) as e:
            return {
                "column_type": "numeric",
                "error": f"Cannot analyze column '{col}': {str(e)}",
                "explanation": f"The '{col}' column contains values that cannot be analyzed as numbers."
            }
        
        # Generate explanation
        if abs(mean_val - median_val) < std_val * 0.1:
            distribution_type = "symmetric (balanced)"
            dist_explanation = (
                f"The values are evenly spread around the middle. "
                f"Like heights in a classroom - most people are average height, "
                f"with fewer very tall or very short people."
            )
        elif skewness_val > 1.0:
            distribution_type = "right-skewed (tail on right)"
            dist_explanation = (
                f"Most values are low, with a few very high values pulling the average up. "
                f"Like income - most people earn moderate amounts, but billionaires "
                f"pull the average way up."
            )
        elif skewness_val < -1.0:
            distribution_type = "left-skewed (tail on left)"
            dist_explanation = (
                f"Most values are high, with a few very low values pulling the average down. "
                f"Like test scores where most students do well, but a few struggle."
            )
        else:
            distribution_type = "slightly skewed"
            dist_explanation = "Values are fairly balanced with a slight lean to one side."
        
        return {
            "column_type": "numeric",
            "statistics": {
                "mean": round(mean_val, 2),
                "median": round(median_val, 2),
                "std_dev": round(std_val, 2),
                "min": round(min_val, 2),
                "max": round(max_val, 2),
                "range": round(max_val - min_val, 2)
            },
            "distribution": {
                "type": distribution_type,
                "skewness": round(skewness_val, 2)
            },
            "outliers": {
                "count": len(outliers),
                "percentage": round((len(outliers) / len(col_data_float)) * 100, 2)
            },
            "explanation": (
                f" '{col}' is a numeric column with values ranging from "
                f"{min_val:.2f} to {max_val:.2f}."
                f"\n\n Center Point:"
                f"\n• Average (mean): {mean_val:.2f}"
                f"\n• Middle value (median): {median_val:.2f}"
                f"\n\n Distribution: {dist_explanation}"
                f"\n\n Outliers: Found {len(outliers)} unusual values "
                f"({(len(outliers)/len(col_data_float)*100):.1f}% of data)."
            )
        }
    
    def _analyze_categorical_column(self, col: str) -> Dict[str, Any]:
        """Analyze a categorical column"""
        
        col_data = self.df[col].dropna()
        
        if len(col_data) == 0:
            return {
                "column_type": "categorical",
                "explanation": f"The '{col}' column is entirely empty!"
            }
        
        value_counts = col_data.value_counts()
        top_5 = value_counts.head(5)
        
        # Calculate concentration
        top_value_pct = float((value_counts.iloc[0] / len(col_data)) * 100)
        
        if top_value_pct > 50.0:
            concentration = "highly concentrated"
            conc_explanation = (
                f"One value ('{value_counts.index[0]}') appears {top_value_pct:.1f}% of the time. "
                f"Think of it like a survey where most people give the same answer."
            )
        elif top_value_pct > 25.0:
            concentration = "moderately concentrated"
            conc_explanation = (
                f"The most common value ('{value_counts.index[0]}') appears "
                f"{top_value_pct:.1f}% of the time."
            )
        else:
            concentration = "well distributed"
            conc_explanation = "Values are fairly evenly spread across categories."
        
        return {
            "column_type": "categorical",
            "unique_count": len(value_counts),
            "most_common": {
                "value": str(value_counts.index[0]),
                "count": int(value_counts.iloc[0]),
                "percentage": round(top_value_pct, 2)
            },
            "top_5_values": {
                str(k): int(v) for k, v in top_5.items()
            },
            "concentration": concentration,
            "explanation": (
                f"📊 '{col}' is a categorical column with {len(value_counts)} different values."
                f"\n\n🏆 Most Common: '{value_counts.index[0]}' appears "
                f"{value_counts.iloc[0]} times ({top_value_pct:.1f}%)"
                f"\n\n💡 Distribution: {conc_explanation}"
                f"\n\n📋 Top 5 Categories:"
            ) + "\n" + "\n".join([
                f"  {i+1}. {k}: {v} occurrences"
                for i, (k, v) in enumerate(top_5.items())
            ])
        }
    
    def _analyze_datetime_column(self, col: str) -> Dict[str, Any]:
        """Analyze a datetime column"""
        
        col_data = self.df[col].dropna()
        
        if len(col_data) == 0:
            return {
                "column_type": "datetime",
                "explanation": f"The '{col}' column is entirely empty!"
            }
        
        min_date = col_data.min()
        max_date = col_data.max()
        date_range = (max_date - min_date).days
        
        return {
            "column_type": "datetime",
            "earliest": str(min_date),
            "latest": str(max_date),
            "range_days": int(date_range),
            "range_years": round(float(date_range) / 365.25, 2),
            "explanation": (
                f"📅 '{col}' contains dates ranging from {min_date.date()} "
                f"to {max_date.date()}."
                f"\n\n⏰ Time Span: {date_range} days ({date_range/365.25:.1f} years)"
                f"\n\nThink of it like a calendar showing events over "
                f"{date_range/365.25:.1f} years."
            )
        }
    
    def _get_statistics(self) -> Dict[str, Any]:
        """Get statistical summary for numeric columns"""
        
        if len(self.numeric_cols) == 0:
            return {
                "message": "No numeric columns to analyze",
                "explanation": (
                    "Your dataset has no numeric columns. It's like trying to calculate "
                    "the average of names - doesn't make sense! All your columns are text or categories."
                )
            }
        
        stats_summary = self.df[self.numeric_cols].describe().to_dict()
        
        return {
            "statistics": stats_summary,
            "explanation": (
                f"📊 Statistical Summary for {len(self.numeric_cols)} numeric columns:"
                f"\n\nThese statistics help you understand the 'shape' of your numbers:"
                f"\n• Count: How many non-empty values"
                f"\n• Mean: Average value"
                f"\n• Std: Standard deviation (how spread out values are)"
                f"\n• Min/Max: Smallest and largest values"
                f"\n• 25%/50%/75%: Quartiles (dividing data into 4 equal parts)"
                f"\n\nThink of it like a report card for each numeric column!"
            )
        }
    
    def _get_correlations(self) -> Dict[str, Any]:
        """
        Calculate correlations between numeric columns.
        
        Plain English: Find which columns move together.
        Like discovering that ice cream sales and temperature are connected!
        """
        
        if len(self.numeric_cols) < 2:
            return {
                "message": "Need at least 2 numeric columns for correlation",
                "explanation": (
                    "Correlation needs at least 2 numeric columns to compare. "
                    "It's like trying to see if two things are related when you only have one thing!"
                )
            }
        
        # Calculate correlation matrix - ensure numeric conversion
        numeric_df = self.df[self.numeric_cols].apply(pd.to_numeric, errors='coerce')
        corr_matrix = numeric_df.corr()
        
        # Find strong correlations (excluding diagonal)
        strong_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                # Convert corr_value safely with explicit type checking
                if pd.notna(corr_value):
                    try:
                        # Use safe conversion
                        corr = self._convert_to_number(corr_value)
                    except (TypeError, ValueError):
                        corr = None
                else:
                    corr = None
                
                # Now work ONLY with clean floats
                if corr is not None and abs(corr) > 0.7:
                    strong_correlations.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": round(corr, 3),
                        "strength": "strong positive" if corr > 0.7 else "strong negative",
                        "explanation": self._explain_correlation(col1, col2, corr)
                    })
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_correlations,
            "explanation": (
                f"🔗 Correlation Analysis:"
                f"\n\nCorrelation measures how two things move together. "
                f"Values range from -1 to +1:"
                f"\n• +1: Perfect positive (when one goes up, other goes up)"
                f"\n• 0: No relationship"
                f"\n• -1: Perfect negative (when one goes up, other goes down)"
                f"\n\n💡 Example: Height and weight are positively correlated - "
                f"taller people tend to weigh more."
                f"\n\n🎯 Found {len(strong_correlations)} strong correlations in your data."
            )
        }
    
    def _explain_correlation(self, col1: str, col2: str, corr: float) -> str:
        """Generate plain English explanation for a correlation"""
        
        if corr > 0.7:
            return (
                f"'{col1}' and '{col2}' have a strong positive relationship ({corr:.2f}). "
                f"When {col1} increases, {col2} tends to increase too. "
                f"Think of it like temperature and ice cream sales - both go up together!"
            )
        elif corr < -0.7:
            return (
                f"'{col1}' and '{col2}' have a strong negative relationship ({corr:.2f}). "
                f"When {col1} increases, {col2} tends to decrease. "
                f"Think of it like heating and electricity bills - as temperature rises, "
                f"heating costs go down!"
            )
        else:
            return f"'{col1}' and '{col2}' have a moderate relationship ({corr:.2f})."
    
    def _analyze_distributions(self) -> Dict[str, Any]:
        """Analyze the distribution of numeric columns"""
        
        if len(self.numeric_cols) == 0:
            return {"message": "No numeric columns to analyze"}
        
        distributions = []
        
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            
            if len(col_data) > 0:
                # Convert to float for normaltest
                try:
                    col_data_float = pd.to_numeric(col_data, errors='coerce').dropna()
                    if len(col_data_float) > 0:
                        # Test for normality (bell curve)
                        _, p_value = stats.normaltest(col_data_float)
                        is_normal = p_value > 0.05
                        
                        distributions.append({
                            "column": col,
                            "is_normal": is_normal,
                            "p_value": self._convert_to_number(p_value),
                            "explanation": (
                                f"'{col}' follows a normal distribution (bell curve). "
                                f"Most values cluster around the average."
                                if is_normal else
                                f"'{col}' does NOT follow a normal distribution. "
                                f"Values are skewed or have unusual patterns."
                            )
                        })
                except Exception as e:
                    distributions.append({
                        "column": col,
                        "is_normal": False,
                        "p_value": None,
                        "explanation": f"Cannot test normality for column '{col}': {str(e)}"
                    })
        
        return {
            "distributions": distributions,
            "explanation": (
                "📊 Distribution Analysis:"
                f"\n\nA 'normal distribution' looks like a bell curve - "
                f"most values in the middle, fewer at extremes."
                f"\n\nExample: Human heights follow a bell curve - most people are "
                f"average height, with fewer very tall or very short people."
                f"\n\n💡 Why it matters: Many ML algorithms work better with "
                f"normally distributed data!"
            )
        }

    
    
    def _analyze_categories(self) -> Dict[str, Any]:
        """Analyze categorical columns for ML preparation"""
        
        if len(self.categorical_cols) == 0:
            return {"message": "No categorical columns to analyze"}
        
        category_analysis = []
        
        for col in self.categorical_cols:
            unique_count = self.df[col].nunique()
            total_count = len(self.df)
            cardinality_ratio = float(unique_count) / float(total_count)
            
            # Classify cardinality
            if cardinality_ratio > 0.5:
                cardinality = "very high"
                recommendation = "Consider grouping or removing - too many categories"
                explanation = (
                    f"'{col}' has {unique_count} unique values out of {total_count} rows. "
                    f"This is TOO SPECIFIC - like having a unique category for each person. "
                    f"ML models struggle with this!"
                )
            elif cardinality_ratio > 0.1:
                cardinality = "high"
                recommendation = "May need encoding or grouping for ML"
                explanation = (
                    f"'{col}' has {unique_count} unique values. This is manageable but "
                    f"may need to be converted to numbers for ML (called 'encoding')."
                )
            else:
                cardinality = "good"
                recommendation = "Perfect for ML - can be one-hot encoded"
                explanation = (
                    f"'{col}' has {unique_count} unique values. Great! "
                    f"Not too many, not too few. Easy to use in ML models."
                )
            
            category_analysis.append({
                "column": col,
                "unique_count": unique_count,
                "cardinality": cardinality,
                "cardinality_ratio": round(cardinality_ratio, 4),
                "recommendation": recommendation,
                "explanation": explanation
            })
        
        return {
            "categories": category_analysis,
            "explanation": (
                "📋 Category Analysis for ML:"
                f"\n\n'Cardinality' means how many unique values a category has."
                f"\n\n💡 Why it matters:"
                f"\n• Low cardinality (2-20 values): Perfect! Easy to use in ML"
                f"\n• Medium cardinality (20-100): Needs encoding but manageable"
                f"\n• High cardinality (>100): Problematic - consider grouping"
                f"\n\nExample: 'Gender' (2 values) is low. 'Country' (195 values) is high."
            )
        }
    
    def prepare_visualization_data(self) -> Dict[str, Any]:
        """
        Prepare data structures for frontend visualizations.
    
        Plain English: Format the data so charts can be drawn easily.
        """
    
        viz_data = {}
    
        # 1. Histograms for numeric columns
        viz_data['histograms'] = {}
        for col in self.numeric_cols[:5]:  # Limit to first 5 to avoid overload
            col_data = self.df[col].dropna()
            # Convert to float for histogram
            try:
                col_data_float = pd.to_numeric(col_data, errors='coerce').dropna()
                if len(col_data_float) > 0:
                    hist, bin_edges = np.histogram(col_data_float, bins=20)
                    viz_data['histograms'][col] = {
                        'counts': hist.tolist(),
                        'bin_edges': [self._convert_to_number(edge) for edge in bin_edges.tolist()],
                        'xlabel': col,
                        'ylabel': 'Frequency'
                    }
            except Exception:
                continue  # Skip columns that can't be converted to numeric
    
        # 2. Box plot data for numeric columns
        viz_data['boxplots'] = {}
        for col in self.numeric_cols[:5]:
            col_data = self.df[col].dropna()
            try:
                col_data_float = pd.to_numeric(col_data, errors='coerce').dropna()
                if len(col_data_float) > 0:
                    q1_val = self._convert_to_number(col_data_float.quantile(0.25))
                    q3_val = self._convert_to_number(col_data_float.quantile(0.75))
                    iqr = q3_val - q1_val
                    lower_bound = q1_val - 1.5 * iqr
                    upper_bound = q3_val + 1.5 * iqr
                    
                    # Get outliers as list of floats
                    outliers_mask = (col_data_float < lower_bound) | (col_data_float > upper_bound)
                    outliers_series = col_data_float[outliers_mask]
                    outliers_list = [self._convert_to_number(x) for x in outliers_series.head(50).tolist()]
                    
                    viz_data['boxplots'][col] = {
                        'min': self._convert_to_number(col_data_float.min()),
                        'q1': q1_val,
                        'median': self._convert_to_number(col_data_float.median()),
                        'q3': q3_val,
                        'max': self._convert_to_number(col_data_float.max()),
                        'outliers': outliers_list
                    }
            except Exception:
                continue  # Skip columns that can't be analyzed
    
         # 3. Bar chart data for categorical columns
        viz_data['bar_charts'] = {}
        for col in self.categorical_cols[:5]:
            value_counts = self.df[col].value_counts().head(10)  # Top 10 categories
            viz_data['bar_charts'][col] = {
                'categories': value_counts.index.tolist(),
                'counts': value_counts.values.tolist()
         }
    
        # 4. Correlation heatmap data
        if len(self.numeric_cols) >= 2:
            # Use only columns that can be converted to numeric
            numeric_df = self.df[self.numeric_cols].apply(pd.to_numeric, errors='coerce')
            corr_matrix = numeric_df.corr()
            viz_data['correlation_heatmap'] = {
                'columns': corr_matrix.columns.tolist(),
                'data': corr_matrix.values.tolist()
            }
    
        # 5. Scatter plot pairs (for first 3 numeric columns)
        viz_data['scatter_plots'] = []
        numeric_pairs = [(self.numeric_cols[i], self.numeric_cols[j]) 
                     for i in range(min(3, len(self.numeric_cols))) 
                     for j in range(i+1, min(3, len(self.numeric_cols)))]
    
        for col1, col2 in numeric_pairs[:5]:  # Limit to 5 pairs
            x_data = self.df[col1].dropna()
            y_data = self.df[col2].dropna()
            # Align indices and convert to float
            try:
                x_data_float = pd.to_numeric(x_data, errors='coerce')
                y_data_float = pd.to_numeric(y_data, errors='coerce')
                aligned_data = pd.concat([x_data_float, y_data_float], axis=1).dropna()
                if len(aligned_data) > 0:
                    viz_data['scatter_plots'].append({
                        'x_column': col1,
                        'y_column': col2,
                        'x_data': [self._convert_to_number(x) for x in aligned_data[col1].head(500).tolist()],
                        'y_data': [self._convert_to_number(y) for y in aligned_data[col2].head(500).tolist()]
                    })
            except Exception:
                continue  # Skip pairs that can't be analyzed
    
        return {
            "visualization_data": viz_data,
            "explanation": (
                "📊 Visualization Data Prepared:"
                f"\n\n✓ {len(viz_data.get('histograms', {}))} histograms (distribution shapes)"
                f"\n✓ {len(viz_data.get('boxplots', {}))} box plots (outlier detection)"
                f"\n✓ {len(viz_data.get('bar_charts', {}))} bar charts (category counts)"
                f"\n✓ Correlation heatmap (relationship strength)"
                f"\n✓ {len(viz_data.get('scatter_plots', []))} scatter plots (2D relationships)"
                f"\n\nThink of visualizations as 'seeing' your data instead of just reading numbers!"
            )
        }
    
    def analyze_time_series(self, date_col: str, value_col: str) -> Dict[str, Any]:
        """
        Analyze time-based patterns.
    
        Plain English: Find patterns over time - like seasonal trends.
        """
    
        if date_col not in self.datetime_cols:
            return {"error": f"'{date_col}' is not a datetime column"}
    
        if value_col not in self.numeric_cols:
            return {"error": f"'{value_col}' is not a numeric column"}
    
        # Sort by date
        ts = self.df[[date_col, value_col]].dropna().copy()
        ts[date_col] = pd.to_datetime(ts[date_col])
        ts = ts.sort_values(date_col)
    
        if ts.empty:
            return {"error": "No data for time series"}
    
        # Resample to daily/monthly
        ts.set_index(date_col, inplace=True)
        daily_avg = ts.resample('D').mean()
        monthly_avg = ts.resample('M').mean()
    
        # Detect trend
        try:
            y = pd.to_numeric(daily_avg[value_col].dropna(), errors='coerce').values
            if len(y) < 2:
                slope = 0.0
                r_value = 0.0
            else:
                x = np.arange(len(y))
                try:
                    result = linregress(x, y)
                    # Access attributes properly - use the returned tuple
                    # Fix: Unpack the result into individual variables
                    slope_val, intercept, r_val, p_val, std_err = result
                    slope = self._convert_to_number(slope_val)
                    r_value = self._convert_to_number(r_val)
                except Exception as e:
                    slope = 0.0
                    r_value = 0.0
        except Exception:
            slope = 0.0
            r_value = 0.0
    
        if slope > 0.0:
            trend = "increasing"
        elif slope < 0.0:
            trend = "decreasing"
        else:
            trend = "stable"
    
        # ensure datetime index formatting using DatetimeIndex
        daily_dates = pd.DatetimeIndex(daily_avg.index).strftime('%Y-%m-%d').tolist()
        monthly_dates = pd.DatetimeIndex(monthly_avg.index).strftime('%Y-%m').tolist()
    
        return {
            "time_range": {
                "start": str(ts.index.min()),
                "end": str(ts.index.max()),
                "duration_days": int((ts.index.max() - ts.index.min()).days)
            },
            "trend": {
                "direction": trend,
                "slope": round(slope, 4),
                "strength": round(abs(r_value), 2)
            },
            "daily_data": {
                "dates": daily_dates,
                "values": [self._convert_to_number(x) if not pd.isna(x) else None for x in daily_avg[value_col].tolist()]
            },
            "monthly_data": {
                "dates": monthly_dates,
                "values": [self._convert_to_number(x) if not pd.isna(x) else None for x in monthly_avg[value_col].tolist()]
            },
            "explanation": (
                f"📅 Time Series Analysis:"
                f"\n\nYour data shows a {trend} trend over time. "
                f"Think of it like tracking your weight daily - "
                f"you can see if it's going up, down, or staying the same!"
            )
        }
    
    def _calculate_quality_score(self) -> Dict[str, Any]:
        """
        Calculate an overall data quality score (0-100).
        
        Plain English: Like a grade for your dataset's cleanliness.
        """
        
        scores = {}
        
        # Completeness (no missing values)
        total_cells = float(self.df.size)
        missing_cells = float(self.df.isnull().sum().sum())
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        scores['completeness'] = round(completeness, 2)
        
        # Uniqueness (no duplicates)
        duplicate_rows = float(self.df.duplicated().sum())
        uniqueness = ((len(self.df) - duplicate_rows) / len(self.df)) * 100
        scores['uniqueness'] = round(uniqueness, 2)
        
        # Validity (correct data types)
        # Simple check: are numeric columns actually numeric?
        valid_cols = 0
        for col in self.df.columns:
            if col in self.numeric_cols:
                # Check if can convert to numeric
                try:
                    pd.to_numeric(self.df[col], errors='raise')
                    valid_cols += 1
                except:
                    pass
            else:
                valid_cols += 1  # Assume non-numeric are valid
        
        validity = (valid_cols / len(self.df.columns)) * 100
        scores['validity'] = round(validity, 2)
        
        # Overall score (weighted average)
        overall_score = (
            completeness * 0.4 +  # Completeness is most important
            uniqueness * 0.3 +
            validity * 0.3
        )
        scores['overall'] = round(overall_score, 2)
        
        # Generate grade
        if overall_score >= 90.0:
            grade = "A (Excellent)"
            explanation = "Your data is in great shape! Minimal issues found."
        elif overall_score >= 80.0:
            grade = "B (Good)"
            explanation = "Your data is pretty clean with minor issues."
        elif overall_score >= 70.0:
            grade = "C (Fair)"
            explanation = "Your data has some issues that should be addressed."
        elif overall_score >= 60.0:
            grade = "D (Poor)"
            explanation = "Your data has significant issues. Cleaning is highly recommended."
        else:
            grade = "F (Failing)"
            explanation = "Your data has major quality issues. Extensive cleaning needed!"
        
        return {
            "scores": scores,
            "overall_score": round(overall_score, 2),
            "grade": grade,
            "explanation": (
                f"🎯 Data Quality Score: {overall_score:.1f}/100 (Grade: {grade})"
                f"\n\n{explanation}"
                f"\n\n📊 Breakdown:"
                f"\n• Completeness: {completeness:.1f}% (how much data is filled in)"
                f"\n• Uniqueness: {uniqueness:.1f}% (no duplicate rows)"
                f"\n• Validity: {validity:.1f}% (correct data types)"
                f"\n\nThink of it like a health score - higher is better!"
            )
        }
    
    def _generate_insights(self) -> List[Dict[str, Any]]:
        """
        Generate actionable insights from the data.
        
        Plain English: Smart observations about your data.
        """
        
        insights = []
        
        # Insight 1: Missing data patterns
        missing_pct = (self.df.isnull().sum().sum() / self.df.size) * 100
        if missing_pct > 10.0:
            insights.append({
                "type": "warning",
                "title": "High Missing Data",
                "message": f"{missing_pct:.1f}% of your data is missing",
                "explanation": (
                    f"More than 10% of your cells are empty. This is like trying to solve "
                    f"a jigsaw puzzle with many pieces missing. You should fill or remove "
                    f"these gaps before analysis."
                ),
                "action": "Use data cleaning tools to fill or remove missing values"
            })
        
        # Insight 2: Duplicate detection
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            insights.append({
                "type": "warning",
                "title": "Duplicate Rows Detected",
                "message": f"Found {duplicates} duplicate rows",
                "explanation": (
                    f"You have {duplicates} rows that are exact copies of others. "
                    f"This inflates your numbers and can bias analysis. Remove these duplicates!"
                ),
                "action": "Remove duplicate rows using the cleaning tools"
            })
        
        # Insight 3: High cardinality columns
        high_card_cols = [
            col for col in self.categorical_cols
            if self.df[col].nunique() / len(self.df) > 0.5
        ]
        if high_card_cols:
            insights.append({
                "type": "info",
                "title": "High Cardinality Columns",
                "message": f"Columns with too many unique values: {', '.join(high_card_cols)}",
                "explanation": (
                    f"These columns have so many different values that they're almost unique "
                    f"per row. It's like trying to find patterns in snowflakes - each is unique! "
                    f"Consider grouping them into broader categories."
                ),
                "action": "Group similar values or remove these columns"
            })
        
        # Insight 4: Strong correlations
        if len(self.numeric_cols) >= 2:
            # Use numeric conversion for correlation
            numeric_df = self.df[self.numeric_cols].apply(pd.to_numeric, errors='coerce')
            corr_matrix = numeric_df.corr()
            strong_corr_count = 0
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if pd.notna(corr_value):
                        try:
                            corr_val = self._convert_to_number(corr_value)
                            if abs(corr_val) > 0.8:
                                strong_corr_count += 1
                        except (TypeError, ValueError):
                            continue
            
            if strong_corr_count > 0:
                insights.append({
                    "type": "success",
                    "title": "Strong Correlations Found",
                    "message": f"Found {strong_corr_count} strong relationships between columns",
                    "explanation": (
                        f"Some columns move together very strongly. This is useful for predictions! "
                        f"But too much correlation can also be redundant - like having both "
                        f"'temperature in Celsius' and 'temperature in Fahrenheit'."
                    ),
                    "action": "Review correlation analysis and consider removing redundant columns"
                })
        
        # Insight 5: Imbalanced data (if binary column exists)
        for col in self.categorical_cols:
            if self.df[col].nunique() == 2:
                value_counts = self.df[col].value_counts()
                imbalance_ratio = float(value_counts.iloc[0]) / float(value_counts.iloc[1])
                
                if imbalance_ratio > 5.0:  # 5:1 ratio or worse
                    insights.append({
                        "type": "warning",
                        "title": "Imbalanced Categories",
                        "message": f"Column '{col}' is heavily imbalanced",
                        "explanation": (
                            f"In '{col}', one category appears {imbalance_ratio:.1f}x more than the other. "
                            f"This is like a coin that lands heads 90% of the time - not fair! "
                            f"ML models might just predict the common category every time."
                        ),
                        "action": "Consider resampling techniques (SMOTE) or weighted training"
                    })
                    break  # Only report first imbalanced column
        
        # If no issues found
        if len(insights) == 0:
            insights.append({
                "type": "success",
                "title": "Data Looks Great!",
                "message": "No major issues detected",
                "explanation": (
                    "Your dataset is in excellent shape! No glaring problems found. "
                    "You're ready to proceed with analysis and modeling."
                ),
                "action": "Proceed to machine learning model building"
            })
        
        return insights
    
    def _assess_ml_readiness(self) -> Dict[str, Any]:
        """
        Assess if dataset is ready for machine learning.
        
        Plain English: Check if your data is ready to train AI models.
        """
        
        checks = []
        readiness_score = 100.0
        
        # Check 1: No missing values
        missing_pct = (self.df.isnull().sum().sum() / self.df.size) * 100
        if missing_pct > 0.0:
            checks.append({
                "check": "Missing Values",
                "status": "warning" if missing_pct < 5.0 else "fail",
                "message": f"{missing_pct:.1f}% missing data",
                "impact": "ML models can't handle missing values - must fill or remove"
            })
            readiness_score -= min(missing_pct, 30.0)
        else:
            checks.append({
                "check": "Missing Values",
                "status": "pass",
                "message": "No missing values",
                "impact": "✓ Ready for ML"
            })
        
        # Check 2: No duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            dup_pct = (duplicates / len(self.df)) * 100
            checks.append({
                "check": "Duplicates",
                "status": "warning",
                "message": f"{duplicates} duplicate rows",
                "impact": "Can bias model training - should remove"
            })
            readiness_score -= min(dup_pct * 2.0, 20.0)
        else:
            checks.append({
                "check": "Duplicates",
                "status": "pass",
                "message": "No duplicates",
                "impact": "✓ Ready for ML"
            })
        
        # Check 3: Sufficient data
        if len(self.df) < 100:
            checks.append({
                "check": "Sample Size",
                "status": "warning",
                "message": f"Only {len(self.df)} rows",
                "impact": "Very small dataset - ML models need more data to learn patterns"
            })
            readiness_score -= 25.0
        elif len(self.df) < 1000:
            checks.append({
                "check": "Sample Size",
                "status": "warning",
                "message": f"{len(self.df)} rows (small)",
                "impact": "Adequate but more data would improve model performance"
            })
            readiness_score -= 10.0
        else:
            checks.append({
                "check": "Sample Size",
                "status": "pass",
                "message": f"{len(self.df)} rows",
                "impact": "✓ Sufficient data for ML"
            })
        
       # Check 4: Feature types
        if len(self.numeric_cols) == 0:
            checks.append({
                "check": "Numeric Features",
                "status": "fail",
                "message": "No numeric columns",
                "impact": "ML models need numeric features - must convert categorical to numbers"
            })
            readiness_score -= 30.0
        else:
            checks.append({
                "check": "Numeric Features",
                "status": "pass",
                "message": f"{len(self.numeric_cols)} numeric columns",
                "impact": "✓ Has numeric features for ML"
            })
        
        # Check 5: Categorical encoding needed
        high_card_cats = [
            col for col in self.categorical_cols
            if self.df[col].nunique() > 50
        ]
        if high_card_cats:
            checks.append({
                "check": "Categorical Encoding",
                "status": "warning",
                "message": f"{len(high_card_cats)} high cardinality columns",
                "impact": "Need to encode or reduce categories before ML"
            })
            readiness_score -= 15.0
        elif self.categorical_cols:
            checks.append({
                "check": "Categorical Encoding",
                "status": "warning",
                "message": f"{len(self.categorical_cols)} categorical columns",
                "impact": "Need encoding (one-hot or label encoding) for ML"
            })
            readiness_score -= 5.0
        else:
            checks.append({
                "check": "Categorical Encoding",
                "status": "pass",
                "message": "No categorical columns",
                "impact": "✓ No encoding needed"
            })
        
        # Overall readiness
        if readiness_score >= 85.0:
            readiness_level = "Ready"
            explanation = "Your data is ready for machine learning! Minimal preparation needed."
        elif readiness_score >= 70.0:
            readiness_level = "Nearly Ready"
            explanation = "Your data needs minor cleaning before ML. Should take just a few minutes."
        elif readiness_score >= 50.0:
            readiness_level = "Needs Work"
            explanation = "Your data needs some preparation before ML. Plan for data cleaning."
        else:
            readiness_level = "Not Ready"
            explanation = "Your data needs significant cleaning before ML. Start with data quality fixes."
        
        return {
            "readiness_score": max(0.0, round(readiness_score, 2)),
            "readiness_level": readiness_level,
            "checks": checks,
            "explanation": (
                f"🤖 ML Readiness: {readiness_score:.0f}/100 ({readiness_level})"
                f"\n\n{explanation}"
                f"\n\n💡 What ML models need:"
                f"\n• No missing values (models can't learn from blanks)"
                f"\n• Numeric data (models work with numbers, not text)"
                f"\n• Clean data (no duplicates or errors)"
                f"\n• Enough samples (more data = better learning)"
                f"\n\nThink of ML like teaching a child - you need clear, "
                f"complete examples for them to learn properly!"
            ),
            "next_steps": self._get_ml_next_steps(checks)
        }
    
    def _get_ml_next_steps(self, checks: List[Dict]) -> List[str]:
        """Generate actionable next steps based on readiness checks"""
        
        steps = []
        
        for check in checks:
            if check["status"] == "fail":
                if "Missing Values" in check["check"]:
                    steps.append("Fill or remove missing values using data cleaning tools")
                elif "Numeric Features" in check["check"]:
                    steps.append("Convert categorical columns to numbers (encoding)")
            elif check["status"] == "warning":
                if "Duplicates" in check["check"]:
                    steps.append("Remove duplicate rows")
                elif "Sample Size" in check["check"]:
                    steps.append("Consider collecting more data if possible")
                elif "Categorical" in check["check"]:
                    steps.append("Encode categorical columns (one-hot or label encoding)")
        
        if not steps:
            steps.append("Your data is ready! Proceed to ML model selection")
        
        return steps
    
    def detect_target_variable(self) -> Dict[str, Any]:
        """
        Suggest potential target variables for ML.
        
        Plain English: Find which column you might want to predict.
        """
        
        suggestions = []
        
        # Look for common target variable patterns
        target_keywords = ['target', 'label', 'class', 'category', 'outcome', 
                          'result', 'prediction', 'fraud', 'churn', 'price', 
                          'sales', 'revenue', 'risk']
        
        for col in self.df.columns:
            col_lower = col.lower()
            
            # Check if column name suggests it's a target
            if any(keyword in col_lower for keyword in target_keywords):
                suggestions.append({
                    "column": col,
                    "reason": "Column name suggests it's a prediction target",
                    "type": "classification" if col in self.categorical_cols else "regression",
                    "confidence": "high"
                })
            
            # Binary columns are often targets
            elif self.df[col].nunique() == 2:
                suggestions.append({
                    "column": col,
                    "reason": "Binary column (Yes/No type) - common for classification",
                    "type": "classification",
                    "confidence": "medium",
                    "unique_values": self.df[col].unique().tolist()
                })
            
            # Low cardinality categorical could be multi-class target
            elif col in self.categorical_cols and self.df[col].nunique() <= 10:
                suggestions.append({
                    "column": col,
                    "reason": f"Has {self.df[col].nunique()} categories - good for classification",
                    "type": "classification",
                    "confidence": "medium",
                    "unique_values": self.df[col].unique().tolist()
                })
        
        return {
            "suggestions": suggestions,
            "explanation": (
                "🎯 Target Variable:"
                f"\n\nThis is what you want to PREDICT. Think of it like:"
                f"\n• Email spam detection → Target: 'Is Spam' (Yes/No)"
                f"\n• House price prediction → Target: 'Price' (number)"
                f"\n• Customer churn → Target: 'Will Leave' (Yes/No)"
                f"\n\n💡 I found {len(suggestions)} potential target variables based on:"
                f"\n• Column names (like 'target', 'label', 'outcome')"
                f"\n• Binary columns (Yes/No, True/False, 0/1)"
                f"\n• Low cardinality categories (2-10 unique values)"
                f"\n\nIf none fit, you can choose any column you want to predict!"
            )
        }
    
    def suggest_ml_models(self, target_column: str) -> Dict[str, Any]:
        """
        Suggest appropriate ML models based on target variable.
        
        Plain English: Recommend which AI models to use.
        """
        
        if target_column not in self.df.columns:
            return {
                "error": "Target column not found",
                "explanation": f"The column '{target_column}' doesn't exist in your dataset."
            }
        
        target_unique = self.df[target_column].nunique()
        is_numeric = target_column in self.numeric_cols
        
        suggestions = []
        
        # Determine problem type
        if is_numeric:
            problem_type = "Regression"
            problem_explanation = (
                "This is a REGRESSION problem - predicting a continuous number. "
                "Like predicting house prices, temperature, or sales amounts."
            )
            
            suggestions = [
                {
                    "model": "Linear Regression",
                    "difficulty": "Beginner",
                    "description": "Simple straight-line fitting",
                    "when_to_use": "When target has linear relationship with features",
                    "pros": ["Fast", "Easy to interpret", "Works well for simple patterns"],
                    "cons": ["Can't capture complex patterns", "Sensitive to outliers"],
                    "analogy": "Drawing the best straight line through scattered points"
                },
                {
                    "model": "Random Forest Regressor",
                    "difficulty": "Intermediate",
                    "description": "Ensemble of decision trees",
                    "when_to_use": "General purpose - works well most of the time",
                    "pros": ["Handles non-linear patterns", "Robust to outliers", "Feature importance"],
                    "cons": ["Slower than linear models", "Less interpretable"],
                    "analogy": "Asking many experts and averaging their opinions"
                },
                {
                    "model": "XGBoost Regressor",
                    "difficulty": "Advanced",
                    "description": "Gradient boosting - state of the art",
                    "when_to_use": "When you need maximum accuracy",
                    "pros": ["Highest accuracy", "Handles missing values", "Fast"],
                    "cons": ["Needs parameter tuning", "Can overfit small datasets"],
                    "analogy": "Learning from mistakes iteratively to get better"
                }
            ]
            
        elif target_unique == 2:
            problem_type = "Binary Classification"
            problem_explanation = (
                "This is BINARY CLASSIFICATION - predicting one of two options. "
                "Like spam/not spam, fraud/not fraud, yes/no decisions."
            )
            
            suggestions = [
                {
                    "model": "Logistic Regression",
                    "difficulty": "Beginner",
                    "description": "Simple probability-based classifier",
                    "when_to_use": "When you need interpretability and simplicity",
                    "pros": ["Fast", "Gives probability scores", "Easy to understand"],
                    "cons": ["Assumes linear decision boundary", "Limited complexity"],
                    "analogy": "Drawing a line to separate two groups"
                },
                {
                    "model": "Random Forest Classifier",
                    "difficulty": "Intermediate",
                    "description": "Multiple decision trees voting",
                    "when_to_use": "General purpose binary classification",
                    "pros": ["Accurate", "Handles non-linear patterns", "Low overfitting"],
                    "cons": ["Slower", "Black box"],
                    "analogy": "Committee of experts voting on the answer"
                },
                {
                    "model": "XGBoost Classifier",
                    "difficulty": "Advanced",
                    "description": "Advanced gradient boosting",
                    "when_to_use": "Competitions and maximum accuracy",
                    "pros": ["Best accuracy", "Feature importance", "Fast"],
                    "cons": ["Needs tuning", "Complex"],
                    "analogy": "Elite specialist who learns from every mistake"
                }
            ]
            
        else:  # Multi-class
            problem_type = "Multi-Class Classification"
            problem_explanation = (
                f"This is MULTI-CLASS CLASSIFICATION - predicting one of {target_unique} categories. "
                f"Like classifying images (cat/dog/bird) or rating products (1-5 stars)."
            )
            
            suggestions = [
                {
                    "model": "Logistic Regression (Multi-class)",
                    "difficulty": "Beginner",
                    "description": "One-vs-rest classification",
                    "when_to_use": "Simple multi-class problems",
                    "pros": ["Fast", "Probability outputs", "Simple"],
                    "cons": ["Limited to linear boundaries"],
                    "analogy": "Multiple lines separating different groups"
                },
                {
                    "model": "Random Forest Classifier",
                    "difficulty": "Intermediate",
                    "description": "Forest of decision trees",
                    "when_to_use": "Most multi-class problems",
                    "pros": ["Handles complex patterns", "Robust", "Feature importance"],
                    "cons": ["Slower training", "Memory intensive"],
                    "analogy": "Panel of judges voting on categories"
                },
                {
                    "model": "XGBoost Classifier",
                    "difficulty": "Advanced",
                    "description": "Boosted trees for multi-class",
                    "when_to_use": "Maximum accuracy needed",
                    "pros": ["Top performance", "Handles imbalance", "Fast prediction"],
                    "cons": ["Complex tuning", "Can overfit"],
                    "analogy": "Expert system that learns hierarchically"
                }
            ]
        
        return {
            "problem_type": problem_type,
            "target_info": {
                "column": target_column,
                "unique_values": int(target_unique),
                "is_numeric": is_numeric,
                "sample_values": self.df[target_column].dropna().unique()[:5].tolist()
            },
            "explanation": problem_explanation,
            "recommended_models": suggestions,
            "next_steps": [
                "1. Choose a model based on your needs (accuracy vs speed vs interpretability)",
                "2. Split data into training and testing sets (80/20 split)",
                "3. Train the model on training data",
                "4. Evaluate on test data to check performance",
                "5. Fine-tune hyperparameters if needed"
            ]
        }
    
    def _convert_to_number(self, value) -> float:
        """
        Safely convert any value to a float number.
        Used throughout the code where number conversion is needed.
        """
        # If it's already missing, return 0
        if pd.isna(value):
            return 0.0
        
        try:
            # Try direct float conversion first
            result = float(value)
            
            # Check if we got NaN back
            if np.isnan(result):
                return 0.0
            return result
            
        except (TypeError, ValueError):
            # If direct conversion fails, try other approaches
            
            # For numpy/pandas objects, try to extract the value
            if hasattr(value, 'item'):
                try:
                    extracted = value.item()
                    if not pd.isna(extracted):
                        return float(extracted)
                except:
                    pass
            
            # As a last resort, try string conversion
            try:
                str_value = str(value)
                
                # Handle complex numbers (like 3+4j)
                if 'j' in str_value or 'i' in str_value:
                    # Try to extract just the real part
                    import cmath
                    try:
                        complex_val = complex(str_value)
                        return float(complex_val.real)
                    except:
                        # If complex conversion fails, try to extract numbers
                        cleaned = ''.join(c for c in str_value if c.isdigit() or c in '.-+')
                        if cleaned:
                            # Try to get the real part
                            parts = cleaned.replace('+', ' ').replace('-', ' -').strip().split()
                            if parts:
                                return float(parts[0])
                
                # For regular numbers, clean and convert
                cleaned = ''.join(c for c in str_value if c.isdigit() or c in '.-')
                if cleaned and cleaned != '-':  # Don't convert just a minus sign
                    return float(cleaned)
                    
            except (ValueError, TypeError):
                pass
        
        # If everything fails, return 0
        return 0