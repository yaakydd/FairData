"""
Plain English Explanations Generator
Converts technical analysis results into user-friendly explanations
"""
from typing import Dict, List, Any, Union
import pandas as pd
import numpy as np

def explain_analysis_result(explanation_type: str, data: Any) -> str:
    """
    Generate plain English explanations for analysis results
    
    Args:
        explanation_type: Type of explanation needed
        data: The data/result to explain
    
    Returns:
        Plain English explanation
    """
    # Define explanation functions separately to avoid f-string issues
    def explain_dataset_overview(d):
        rows = d.get('rows', 0)
        columns = d.get('columns', 0)
        memory = d.get('memory_usage', 'N/A')
        
        # Handle dtypes safely
        dtypes_items = list(d.get('dtypes', {}).items())
        dtype_str = ', '.join([f"{k}: {v}" for k, v in dtypes_items[:5]]) if dtypes_items else "No type info"
        
        small_note = "" if rows > 1000 else "relatively small, which is "
        
        return f"""
Your Dataset Overview:

• Size: {rows:,} rows × {columns} columns
• Memory: {memory}
• Data Types: {dtype_str}

This dataset contains {rows:,} records with {columns} different features.
The data appears to be {small_note}suitable for analysis.
"""
    
    def explain_data_quality(d):
        quality_score = d.get('quality_score', 0)
        
        # Issues
        issues = d.get('issues', [])
        issues_str = '\n'.join(f'• {issue}' for issue in issues) if issues else '• No major issues detected'
        
        # Missing values
        missing_percentages = d.get('missing_percentages', {})
        missing_str = ""
        if missing_percentages:
            missing_items = []
            for col, missing in missing_percentages.items():
                if missing > 0:
                    missing_items.append(f'• {col}: {missing}% missing')
                    if len('\n'.join(missing_items)) > 200:
                        break
            missing_str = '\n'.join(missing_items)
        else:
            missing_str = '• No missing values found'
        
        # Quality assessment
        quality_note = "⚠️ Attention: Significant data quality issues detected. Consider cleaning before analysis." if quality_score < 70 else "✓ Good data quality overall."
        
        return f"""
Data Quality Assessment:

Overall Quality Score: {quality_score}/100

Issues Found:
{issues_str}

Missing Values:
{missing_str}

{quality_note}
"""
    
    def explain_key_insights(d):
        if d:
            insights_str = '\n'.join(f'• {insight}' for insight in d)
        else:
            insights_str = '• No specific insights generated from basic analysis. Try more detailed analysis.'
        
        return f"""
Key Insights Discovered:

{insights_str}

These insights can help you:
1. Understand your data better
2. Identify important patterns
3. Make data-driven decisions
4. Plan next steps for analysis or modeling
"""
    
    def explain_recommendations(d):
        if d:
            recs_str = '\n'.join(f'{i+1}. {rec}' for i, rec in enumerate(d))
        else:
            recs_str = '1. Explore your data with different analysis types\n2. Check column statistics for detailed insights\n3. Consider data cleaning if needed'
        
        return f"""
Recommendations for Next Steps:

{recs_str}

Priority Order:
1. Address critical data quality issues first
2. Explore relationships between variables
3. Prepare for machine learning if applicable
"""
    
    def explain_column_stats(d):
        column_name = d.get('column_name', 'Unknown Column')
        stats = d.get('stats', {})
        dtype = d.get('dtype', 'Unknown')
        
        # Basic stats
        if isinstance(stats, dict):
            basic_stats = stats.get('basic_stats', 'No basic stats available')
            dist_info = stats.get('distribution_info', 'Check distribution type for details')
        else:
            basic_stats = 'Statistical analysis performed'
            dist_info = 'Distribution analyzed'
        
        return f"""
Column Analysis: {column_name}

Basic Statistics:
{basic_stats}

Distribution:
{dist_info}

Data Type: {dtype}

What this means:
• These statistics show how values are distributed in this column
• Look for unusual patterns or outliers
• Consider if the data type matches the actual content
"""
    
    def explain_patterns(d):
        pattern_type = d.get('pattern_type', 'general')
        patterns_found = d.get('patterns_found', 0)
        
        # Patterns list
        patterns = d.get('patterns', [])
        if patterns:
            patterns_str = '\n'.join(f'• {pattern}' for pattern in patterns[:5])
        else:
            patterns_str = '• No significant patterns detected'
        
        # Interpretation
        if patterns_found > 0:
            interpretation = 'Strong patterns detected. Consider investigating these relationships further.'
        else:
            interpretation = 'No strong patterns found. The data may be random or complex relationships exist.'
        
        return f"""
Pattern Detection Results:

Type of patterns analyzed: {pattern_type}
Total patterns found: {patterns_found}

Significant Patterns:
{patterns_str}

Interpretation:
{interpretation}
"""
    
    def explain_correlation(d):
        method = d.get('method', 'Pearson')
        threshold = d.get('threshold', 0.5)
        significant_pairs = d.get('significant_pairs', 0)
        matrix_shape = d.get('matrix_shape', 'Unknown')
        
        if significant_pairs > 0:
            corr_note = "Strong relationships detected! These can be valuable for modeling."
        else:
            corr_note = "No strong linear relationships found."
        
        return f"""
Correlation Analysis:

Method used: {method}
Significance threshold: {threshold}
Significant correlations found: {significant_pairs}

Correlation Matrix Size: {matrix_shape}

What correlations tell us:
• Positive correlation (> 0.5): When one variable increases, the other tends to increase
• Negative correlation (< -0.5): When one increases, the other tends to decrease
• Weak correlation (-0.3 to 0.3): Little to no relationship

{corr_note}
"""
    
    def explain_outliers(d):
        total_outliers = d.get('total_outliers', 0)
        affected_columns = d.get('affected_columns', [])
        
        if affected_columns:
            columns_str = ', '.join(affected_columns)
        else:
            columns_str = 'None'
        
        stats = d.get('stats', {})
        if isinstance(stats, dict):
            summary = stats.get('summary', 'No outlier statistics')
        else:
            summary = 'Outlier analysis performed'
        
        if total_outliers > 0:
            rec = 'Consider investigating these unusual values. They could be errors or important anomalies.'
        else:
            rec = 'No significant outliers detected. Your data appears normally distributed.'
        
        return f"""
Outlier Detection:

Total outliers detected: {total_outliers}
Affected columns: {columns_str}

Outlier Summary:
{summary}

Recommendations for outliers:
{rec}
"""
    
    def explain_analysis_summary(d):
        analysis_type = d.get('type', 'Comprehensive analysis')
        dataset_shape = d.get('dataset_shape', (0, 0))
        rows = dataset_shape[0]
        columns_count = dataset_shape[1] if len(dataset_shape) > 1 else 0
        columns_analyzed = d.get('columns_analyzed', [])
        
        return f"""
Analysis Summary:

Type: {analysis_type}
Dataset: {rows:,} rows × {columns_count} columns
Columns analyzed: {len(columns_analyzed)}

What was done:
• Statistical calculations on all columns
• Data quality assessment
• Pattern detection
• Relationship analysis

Analysis complete! Review the detailed results below.
"""
    
    # Map explanation types to functions
    explanations = {
        "dataset_overview": explain_dataset_overview,
        "data_quality": explain_data_quality,
        "key_insights": explain_key_insights,
        "recommendations": explain_recommendations,
        "column_stats": explain_column_stats,
        "patterns": explain_patterns,
        "correlation": explain_correlation,
        "outliers": explain_outliers,
        "analysis_summary": explain_analysis_summary
    }
    
    if explanation_type in explanations:
        try:
            return explanations[explanation_type](data)
        except Exception as e:
            return f"Explanation generated with some issues: {str(e)}"
    else:
        # Truncate long data for unknown types
        data_str = str(data)
        if len(data_str) > 500:
            data_str = data_str[:497] + "..."
        return f"Analysis Results for {explanation_type}:\n{data_str}"

def generate_insights_summary(analysis_result: Dict[str, Any]) -> str:
    """
    Generate a comprehensive insights summary from analysis results
    
    Args:
        analysis_result: Complete analysis result dictionary
    
    Returns:
        Formatted insights summary
    """
    insights = []
    
    # Extract overview insights
    overview = analysis_result.get("overview", {})
    if overview:
        rows = overview.get('rows', 0)
        columns = overview.get('columns', 0)
        insights.append(f"Dataset contains {rows:,} records with {columns} features")
    
    # Extract data quality insights
    quality = analysis_result.get("data_quality", {})
    if quality:
        score = quality.get("quality_score", 100)
        if score >= 90:
            insights.append("Excellent data quality - ready for analysis")
        elif score >= 70:
            insights.append("Good data quality - minor issues detected")
        else:
            insights.append("Data quality needs improvement - consider cleaning")
        
        issues = quality.get("issues", [])
        if issues:
            insights.append(f"{len(issues)} data quality issues to address")
    
    # Extract missing values insights
    missing = quality.get("missing_percentages", {})
    if missing:
        high_missing = [col for col, perc in missing.items() if perc > 20]
        if high_missing:
            insights.append(f"{len(high_missing)} columns have >20% missing values")
    
    # Extract statistical insights
    stats = analysis_result.get("statistical_summary", {})
    if stats:
        num_cols = len(stats.get("numerical_columns", []))
        cat_cols = len(stats.get("categorical_columns", []))
        if num_cols > 0:
            insights.append(f"{num_cols} numerical columns for statistical analysis")
        if cat_cols > 0:
            insights.append(f"{cat_cols} categorical columns for grouping")
    
    # Extract pattern insights
    patterns = analysis_result.get("patterns", [])
    if patterns and len(patterns) > 0:
        insights.append(f"{len(patterns)} significant patterns detected")
    
    # Extract correlation insights
    correlations = analysis_result.get("correlation_insights", [])
    if correlations and len(correlations) > 0:
        strong_corrs = [c for c in correlations if abs(c.get("correlation", 0)) > 0.7]
        if strong_corrs:
            insights.append(f"{len(strong_corrs)} strong correlations found")
    
    # Add key insights from analysis
    key_insights = analysis_result.get("key_insights", [])
    if key_insights:
        insights.extend(key_insights[:3])  # Add top 3 key insights
    
    # If no insights generated, add generic ones
    if not insights:
        insights = [
            "Basic analysis complete - explore detailed statistics",
            "Try different analysis types for deeper insights",
            "Consider data cleaning if you notice any issues"
        ]
    
    return "\n".join(f"• {insight}" for insight in insights)

def explain_data_cleaning_action(action_type: str, details: Dict) -> str:
    """
    Explain data cleaning actions in plain English
    
    Args:
        action_type: Type of cleaning action
        details: Details about what will be changed
    
    Returns:
        Plain English explanation
    """
    def explain_handle_missing(d):
        method = d.get('method', 'Fill with median/mean/mode')
        columns_count = len(d.get('columns', []))
        total_missing = d.get('total_missing', 0)
        rows_affected = d.get('rows_affected', 0)
        
        return f"""
Missing Value Treatment:

Action: {method}
Columns affected: {columns_count}
Total missing values: {total_missing}

Impact:
• {rows_affected} rows will be modified
• Dataset completeness will improve
• Statistical properties may change slightly

Reason: Missing values can cause errors in analysis and modeling. This action ensures your data is complete.
"""
    
    def explain_remove_duplicates(d):
        duplicate_count = d.get('duplicate_count', 0)
        rows_before = d.get('rows_before', 0)
        rows_after = d.get('rows_after', 0)
        
        return f"""
Duplicate Removal:

Action: Remove {duplicate_count} duplicate rows
Rows before: {rows_before}
Rows after: {rows_after}

Impact:
• {duplicate_count} duplicate records will be removed
• Data quality will improve
• Analysis results will be more accurate

Reason: Duplicates can skew statistical results and mislead analysis.
"""
    
    def explain_fix_datatypes(d):
        columns = d.get('columns', [])
        current_types = d.get('current_types', 'Unknown')
        new_types = d.get('new_types', 'Appropriate types')
        
        return f"""
Data Type Correction:

Columns to convert: {', '.join(columns)}
Current types: {current_types}
New types: {new_types}

Impact:
• Better memory usage
• Correct statistical calculations
• Proper visualization

Reason: Incorrect data types can cause calculation errors and inefficient processing.
"""
    
    def explain_remove_outliers(d):
        method = d.get('method', 'IQR or Z-score method')
        outlier_count = d.get('outlier_count', 0)
        columns = d.get('columns', [])
        rows_affected = d.get('rows_affected', 0)
        
        return f"""
Outlier Treatment:

Method: {method}
Outliers detected: {outlier_count}
Columns affected: {', '.join(columns)}

Impact:
• {rows_affected} rows may be modified/removed
• Statistical properties will normalize
• Model performance may improve

Reason: Extreme outliers can distort analysis and modeling results.
"""
    
    def explain_standardize_format(d):
        issue = d.get('issue', 'Inconsistent formatting')
        columns_count = len(d.get('columns', []))
        examples = d.get('examples', 'Various formats')
        
        return f"""
Format Standardization:

Issue: {issue}
Columns affected: {columns_count}
Examples: {examples}

Impact:
• Consistent data presentation
• Easier analysis and filtering
• Better user experience

Reason: Inconsistent formats make data difficult to analyze and compare.
"""
    
    explanations = {
        "handle_missing": explain_handle_missing,
        "remove_duplicates": explain_remove_duplicates,
        "fix_datatypes": explain_fix_datatypes,
        "remove_outliers": explain_remove_outliers,
        "standardize_format": explain_standardize_format
    }
    
    if action_type in explanations:
        return explanations[action_type](details)
    else:
        return f"Cleaning Action: {action_type}\nDetails: {details}"

def suggest_next_steps(analysis_type: str, results: Dict) -> List[str]:
    """
    Suggest next steps based on analysis results
    
    Args:
        analysis_type: Type of analysis performed
        results: Analysis results
    
    Returns:
        List of suggested next steps
    """
    steps = []
    
    if analysis_type == "initial":
        steps.append("1. Review data quality issues and consider cleaning")
        steps.append("2. Explore detailed column statistics")
        steps.append("3. Check for correlations between variables")
        steps.append("4. Look for patterns and trends")
        steps.append("5. Consider visualization for better understanding")
    
    elif analysis_type == "statistical":
        steps.append("1. Validate statistical assumptions if applicable")
        steps.append("2. Consider hypothesis testing if patterns are found")
        steps.append("3. Check for normality in numerical columns")
        steps.append("4. Explore relationships with visualizations")
    
    elif analysis_type == "correlation":
        strong_corrs = results.get("strong_correlations", [])
        if strong_corrs:
            steps.append("1. Investigate strongly correlated variables further")
            steps.append("2. Consider if correlation implies causation (usually it doesn't)")
            steps.append("3. Remove highly correlated features if doing machine learning")
        steps.append("4. Create scatter plots for strong correlations")
    
    elif analysis_type == "quality":
        issues = results.get("issues", [])
        if issues:
            steps.append("1. Address critical data quality issues first")
            steps.append("2. Create a data cleaning plan")
            steps.append("3. Document all cleaning actions")
        steps.append("4. Re-analyze after cleaning to verify improvements")
    
    # Always add these general steps
    steps.append("5. Save your analysis results for future reference")
    steps.append("6. Consider the business context of your findings")
    steps.append("7. Prepare visualizations to communicate insights")
    
    return steps