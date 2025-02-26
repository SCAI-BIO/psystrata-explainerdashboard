import base64
import io

import shap
from dash import html, dcc
from matplotlib import pyplot as plt


class SHAPSummaryPlot:

    def __init__(self, shap_values, X, max_display=10, max_feature_name_length=20):
        self.shap_values = shap_values
        self.X = X
        self.max_display = max_display
        self.max_feature_name_length = max_feature_name_length
        self.create_summary_plot()

    def create_summary_plot(self):
        # Truncate feature names to a maximum of 20 characters
        self.max_feature_name_length = 20
        truncated_feature_names = [name[:self.max_feature_name_length] for name in self.X.columns]
        plt.figure(figsize=(15, 8))  # Increase the figure size
        shap.summary_plot(self.shap_values, self.X, feature_names=truncated_feature_names, max_display=self.max_display,
                          show=False)
        plt.tight_layout()

    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def to_dash_component():
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()  # Close the plot to avoid displaying it elsewhere
        buf.seek(0)
        # Convert the plot to a base64 string
        shap_plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return html.Div([html.Img(src=f"data:image/png;base64,{shap_plot_base64}", className="img-fluid")])