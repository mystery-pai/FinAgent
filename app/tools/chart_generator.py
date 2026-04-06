"""
Interactive chart generator using Plotly.
交互式图表生成器
"""
import plotly.graph_objects as go
from typing import Literal
import logging

from app.schemas.models import ChartDataSchema

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generate interactive charts using Plotly"""

    # Professional color palette inspired by D3.js Category10
    COLORS = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
    ]

    def generate(
        self,
        data: ChartDataSchema,
        chart_type: Literal["line", "bar", "grouped_bar", "pie", "auto"] = "auto"
    ) -> go.Figure:
        """
        Generate chart from structured data

        Args:
            data: Structured chart data with series and labels
            chart_type: Type of chart to generate. "auto" will auto-detect.

        Returns:
            Plotly Figure object
        """
        logger.info(f"Generating chart: {data.title}, type={chart_type}")

        # Auto-detect chart type if needed
        if chart_type == "auto":
            chart_type = self._detect_chart_type(data)
            logger.info(f"Auto-detected chart type: {chart_type}")

        # Route to specific generator
        if chart_type == "line":
            fig = self._generate_line_chart(data)
        elif chart_type == "bar":
            fig = self._generate_bar_chart(data)
        elif chart_type == "grouped_bar":
            fig = self._generate_grouped_bar_chart(data)
        elif chart_type == "pie":
            fig = self._generate_pie_chart(data)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        # Apply common layout settings
        self._apply_common_layout(fig, data)

        return fig

    def _detect_chart_type(self, data: ChartDataSchema) -> str:
        """
        Auto-detect appropriate chart type based on data structure

        Args:
            data: Chart data schema

        Returns:
            Detected chart type: "line", "bar", "grouped_bar", or "pie"
        """
        # Use LLM hint if available
        if data.chart_type_hint:
            hint = data.chart_type_hint.lower()
            if hint in ["line", "bar", "grouped_bar", "pie"]:
                return hint

        num_series = len(data.series)
        num_x_values = len(data.x_values)

        # Single x-value with multiple series → bar chart
        if num_x_values == 1 and num_series > 1:
            return "bar"

        # Check if x-values look like time series
        is_time_series = self._is_time_series(data.x_values)

        # Multiple series + categorical → grouped bar
        if num_series > 1 and not is_time_series:
            return "grouped_bar"

        # Single series + multiple x-values → line for time, bar for categorical
        if num_series == 1:
            return "line" if is_time_series else "bar"

        # Multiple series + time-based → line chart
        if num_series > 1 and is_time_series:
            return "line"

        return "line"

    def _is_time_series(self, x_values: list) -> bool:
        """Check if x-values represent time series data"""
        if not x_values:
            return False

        # Check if all values are integers that look like years (1900-2100)
        try:
            if all(isinstance(x, int) or (isinstance(x, str) and x.isdigit()) for x in x_values):
                years = [int(x) for x in x_values]
                if all(1900 <= y <= 2100 for y in years):
                    return True
        except (ValueError, TypeError):
            pass

        # Check if values contain date/quarter keywords
        str_values = [str(x).lower() for x in x_values]
        time_keywords = ['q1', 'q2', 'q3', 'q4', 'quarter', 'jan', 'feb', 'mar',
                         'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        if any(any(kw in val for kw in time_keywords) for val in str_values):
            return True

        return False

    def _generate_line_chart(self, data: ChartDataSchema) -> go.Figure:
        """Generate line chart for trend visualization"""
        fig = go.Figure()

        for idx, series in enumerate(data.series):
            color = self.COLORS[idx % len(self.COLORS)]

            hover_template = f"<b>{series.name}</b><br>"
            hover_template += f"{data.x_label}: %{{x}}<br>"
            hover_template += f"{data.y_label}: %{{y:,.2f}}"
            if series.unit:
                hover_template += f" {series.unit}"
            hover_template += "<extra></extra>"

            fig.add_trace(go.Scatter(
                x=data.x_values,
                y=series.values,
                mode='lines+markers',
                name=series.name,
                line=dict(color=color, width=3),
                marker=dict(size=8, color=color),
                hovertemplate=hover_template
            ))

        return fig

    def _generate_bar_chart(self, data: ChartDataSchema) -> go.Figure:
        """Generate bar chart for single-series comparisons"""
        fig = go.Figure()

        for idx, series in enumerate(data.series):
            color = self.COLORS[idx % len(self.COLORS)]

            hover_template = f"<b>{series.name}</b><br>"
            hover_template += f"{data.x_label}: %{{x}}<br>"
            hover_template += f"{data.y_label}: %{{y:,.2f}}"
            if series.unit:
                hover_template += f" {series.unit}"
            hover_template += "<extra></extra>"

            fig.add_trace(go.Bar(
                x=data.x_values,
                y=series.values,
                name=series.name,
                marker=dict(
                    color=color,
                    line=dict(color='white', width=1)
                ),
                hovertemplate=hover_template
            ))

        return fig

    def _generate_grouped_bar_chart(self, data: ChartDataSchema) -> go.Figure:
        """Generate grouped bar chart for multi-series comparisons"""
        fig = go.Figure()

        for idx, series in enumerate(data.series):
            color = self.COLORS[idx % len(self.COLORS)]

            hover_template = f"<b>{series.name}</b><br>"
            hover_template += f"{data.x_label}: %{{x}}<br>"
            hover_template += f"{data.y_label}: %{{y:,.2f}}"
            if series.unit:
                hover_template += f" {series.unit}"
            hover_template += "<extra></extra>"

            fig.add_trace(go.Bar(
                x=data.x_values,
                y=series.values,
                name=series.name,
                marker=dict(
                    color=color,
                    line=dict(color='white', width=1)
                ),
                hovertemplate=hover_template
            ))

        fig.update_layout(barmode='group')
        return fig

    def _generate_pie_chart(self, data: ChartDataSchema) -> go.Figure:
        """Generate pie chart for composition/proportion visualization"""
        fig = go.Figure()

        if len(data.series) == 1:
            labels = [str(x) for x in data.x_values]
            values = data.series[0].values
            series_name = data.series[0].name
        else:
            labels = [s.name for s in data.series]
            values = [sum(s.values) for s in data.series]
            series_name = data.y_label

        hover_template = "<b>%{label}</b><br>"
        hover_template += f"{series_name}: %{{value:,.2f}}<br>"
        hover_template += "Percentage: %{percent}<extra></extra>"

        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            marker=dict(
                colors=self.COLORS[:len(labels)],
                line=dict(color='white', width=2)
            ),
            textposition='inside',
            textinfo='label+percent',
            hovertemplate=hover_template
        ))

        return fig

    def _apply_common_layout(self, fig: go.Figure, data: ChartDataSchema) -> None:
        """Apply common layout settings to all chart types"""
        is_pie = any(isinstance(trace, go.Pie) for trace in fig.data)

        layout_config = {
            'title': {
                'text': data.title,
                'font': {'size': 20, 'color': '#2c3e50'},
                'x': 0.5,
                'xanchor': 'center'
            },
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'plot_bgcolor': 'rgba(250, 250, 250, 1)',
            'paper_bgcolor': 'white',
            'hovermode': 'closest',
            'showlegend': True,
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.2,
                'xanchor': 'center',
                'x': 0.5,
                'bgcolor': 'rgba(255, 255, 255, 0.8)',
                'bordercolor': '#E0E0E0',
                'borderwidth': 1
            },
            'margin': {'l': 80, 'r': 80, 't': 100, 'b': 120}
        }

        if not is_pie:
            layout_config.update({
                'xaxis': {
                    'title': {'text': data.x_label, 'font': {'size': 14, 'color': '#34495e'}},
                    'showgrid': True,
                    'gridcolor': 'rgba(200, 200, 200, 0.3)',
                    'zeroline': False
                },
                'yaxis': {
                    'title': {'text': data.y_label, 'font': {'size': 14, 'color': '#34495e'}},
                    'showgrid': True,
                    'gridcolor': 'rgba(200, 200, 200, 0.3)',
                    'zeroline': True,
                    'zerolinecolor': 'rgba(150, 150, 150, 0.5)'}
            })

        fig.update_layout(**layout_config)

        if data.data_source:
            fig.add_annotation(
                text=f"Source: {data.data_source}",
                xref="paper",
                yref="paper",
                x=1.0,
                y=-0.15 if is_pie else -0.25,
                showarrow=False,
                font={'size': 10, 'color': '#7f8c8d'},
                xanchor='right',
                yanchor='top'
            )
