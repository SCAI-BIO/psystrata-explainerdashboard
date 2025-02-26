from dash import html
from ollama import Client
import json
import pandas as pd
import re
import numpy as np

client = Client(host='http://localhost:11434')


def chat(message) -> str:
    """Send a chat message to the LLaMA model and return the raw response."""
    try:
        response = client.chat(
            model='llama3.2:3b',
            messages=[{'role': 'user', 'content': message}],
            stream=False
        )

        if response and 'message' in response:
            return response['message']['content']
        else:
            return "Unexpected response: " + json.dumps(response, indent=2)

    except Exception as e:
        return f"Error during chat request: {e}"


class SHAPExplanation:

    def __init__(self, shap_df, feature_df):
        self.shap_df = shap_df
        self.feature_df = feature_df
        self.shap_statistical_summary = self.create_statistical_shap_summary()
        self.prompt = f"""
        I have a dataset with SHAP values generated from a machine learning model. SHAP values explain how much each 
        factor influenced a prediction. In our case we are predicting treatment resistance, therefore positive SHAP 
        values are indicative of an increased risk of treatment resistance and negativeSHAP values indicate a decreased 
        risk of treatment resistance.

        Imagine you're explaining these SHAP values to a clinician with no background in data science. Focus on:
        - How SHAP values relate to patient risk scores or outcomes.
        - Using clinical language, like "This factor increased the risk of X" instead of technical terms.
        - Avoiding jargon and complex statistics.

        Here is a summary of the number of and median value for positive and negative shap values for the 10 most 
        important features:
        ```
        {self.shap_statistical_summary.to_string()}
        ```

        It is important how the feature values (summarized in median_positive_feature_value and median_negative_feature_value)
        relate to the SHAP values. For example, a high median_positive_feature_value for a feature with a high median_positive_shap
        value indicates that high values of that feature are associated with an increased risk of treatment resistance.

        Similarly, a low median_negative_feature_value for a feature with a high median_negative_shap value indicates 
        that low values of that feature are associated with an decreased risk of treatment resistance.

        A particular low count of positive or negative shap values for a feature may indicate the presents of outliers.

        Please explain these SHAP values in a way a clinician can understand, emphasizing the most influential factors 
        and their clinical relevance. 
        
        As you do not have any label information for categorical one-hot-encoded features, do under no cirumstances 
        imply causation in this cases and do not hallicinate any labels. Do for these cases only state their general 
        SHAP impact (e.g. some patients with high values of this feature belonging to group 1 have a higher risk of 
        treatment resistance).
        """
        self.raw_llm_response = self.get_llm_response()

    def create_statistical_shap_summary(self, limit: int=10) -> pd.DataFrame:
        """
        Create a summary DataFrame with SHAP value statistics and corresponding feature values.
        """
        shap_feature_df = self.shap_df
        feature_df = self.feature_df
        summary = pd.DataFrame({
            'mean_absolute_shap': shap_feature_df.abs().mean(),
            'median_positive_shap': shap_feature_df.apply(lambda x: x[x > 0].median(), axis=0),
            'count_positive_shap': shap_feature_df.apply(lambda x: (x > 0).sum(), axis=0),
            'median_negative_shap': shap_feature_df.apply(lambda x: x[x < 0].median(), axis=0),
            'count_negative_shap': shap_feature_df.apply(lambda x: (x < 0).sum(), axis=0),
            'median_positive_feature_value': shap_feature_df.apply(
                lambda x: feature_df.loc[x.index[x > 0], x.name].median()
                if x.name in feature_df.columns else np.nan,
                axis=0
            ),
            'median_negative_feature_value': shap_feature_df.apply(
                lambda x: feature_df.loc[x.index[x < 0], x.name].median()
                if x.name in feature_df.columns else np.nan,
                axis=0
            )
        })

        # limit summary to the top 10 features in terms if absolute mean SHAP value
        summary = summary.sort_values('mean_absolute_shap', ascending=False).head(limit)

        return summary

    def get_llm_response(self):
        raw_response = chat(self.prompt)
        self.raw_llm_response = raw_response
        return raw_response

    def to_list(self):
        """Convert the LLaMA response into a list of paragraphs"""
        paragraphs = self.raw_llm_response.split("\n\n")
        paragraphs = [para.strip() for para in paragraphs if para.strip()]
        return paragraphs

    def to_dash_component(self):
        """Convert the LLaMA response into a list of Dash HTML components"""
        paragraphs = self.to_list()
        return html.Div([html.P(para) for para in paragraphs])

