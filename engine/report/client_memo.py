"""
Client memo generation for portfolio optimization results.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import structlog

logger = structlog.get_logger(__name__)


class ClientMemoGenerator:
    """Generate client memos for portfolio optimization results."""
    
    def __init__(self, output_dir: Union[str, Path] = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # PDF styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self) -> None:
        """Setup custom paragraph styles."""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        ))
        
        # Subheading style
        self.styles.add(ParagraphStyle(
            name='CustomSubHeading',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            textColor=colors.darkgreen
        ))
        
        # Body style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6
        ))
    
    def generate_memo(
        self,
        backtest_results: Dict[str, Any],
        policy_config: Dict[str, Any],
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Generate client memo PDF.
        
        Args:
            backtest_results: Backtest results dictionary
            policy_config: Client policy configuration
            output_filename: Output filename (optional)
        
        Returns:
            Path to generated PDF file
        """
        
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"client_memo_{timestamp}.pdf"
        
        output_path = self.output_dir / output_filename
        
        logger.info("Generating client memo", output_path=str(output_path))
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build content
        story = []
        
        # Title
        story.append(Paragraph("Portfolio Optimization Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 12))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['CustomHeading']))
        story.append(self._generate_executive_summary(backtest_results, policy_config))
        story.append(Spacer(1, 12))
        
        # Performance Metrics
        story.append(Paragraph("Performance Metrics", self.styles['CustomHeading']))
        story.append(self._generate_performance_table(backtest_results))
        story.append(Spacer(1, 12))
        
        # Risk Analysis
        story.append(Paragraph("Risk Analysis", self.styles['CustomHeading']))
        story.append(self._generate_risk_analysis(backtest_results))
        story.append(Spacer(1, 12))
        
        # Portfolio Composition
        story.append(Paragraph("Portfolio Composition", self.styles['CustomHeading']))
        story.append(self._generate_portfolio_composition(backtest_results))
        story.append(Spacer(1, 12))
        
        # Constraints Analysis
        story.append(Paragraph("Constraints Analysis", self.styles['CustomHeading']))
        story.append(self._generate_constraints_analysis(policy_config))
        story.append(Spacer(1, 12))
        
        # Recommendations
        story.append(Paragraph("Recommendations", self.styles['CustomHeading']))
        story.append(self._generate_recommendations(backtest_results, policy_config))
        story.append(Spacer(1, 12))
        
        # Disclaimer
        story.append(Paragraph("Disclaimer", self.styles['CustomHeading']))
        story.append(self._generate_disclaimer())
        
        # Build PDF
        doc.build(story)
        
        logger.info("Client memo generated successfully", output_path=str(output_path))
        return output_path
    
    def _generate_executive_summary(
        self,
        backtest_results: Dict[str, Any],
        policy_config: Dict[str, Any]
    ) -> Paragraph:
        """Generate executive summary paragraph."""
        
        # Extract key metrics
        total_return = backtest_results.get("total_return", 0)
        annualized_return = backtest_results.get("annualized_return", 0)
        sharpe_ratio = backtest_results.get("sharpe_ratio", 0)
        max_drawdown = backtest_results.get("max_drawdown", 0)
        
        # Generate summary text
        summary_text = f"""
        This report presents the results of a comprehensive portfolio optimization analysis 
        conducted using the Omega Portfolio Engine. The optimization was performed using a 
        {policy_config.get('optimization', {}).get('primary_objective', 'GMV')} objective 
        with regime-aware parameter estimation.
        
        <b>Key Results:</b><br/>
        • Total Return: {total_return:.2%}<br/>
        • Annualized Return: {annualized_return:.2%}<br/>
        • Sharpe Ratio: {sharpe_ratio:.2f}<br/>
        • Maximum Drawdown: {max_drawdown:.2%}<br/>
        
        The portfolio demonstrates {'strong' if sharpe_ratio > 1.0 else 'moderate' if sharpe_ratio > 0.5 else 'weak'} 
        risk-adjusted performance with a Sharpe ratio of {sharpe_ratio:.2f}. 
        {'The maximum drawdown of ' + f'{max_drawdown:.2%}' + ' is within acceptable limits.' if abs(max_drawdown) < 0.20 else 'The maximum drawdown of ' + f'{max_drawdown:.2%}' + ' exceeds typical risk tolerance levels.'}
        """
        
        return Paragraph(summary_text, self.styles['CustomBody'])
    
    def _generate_performance_table(self, backtest_results: Dict[str, Any]) -> Table:
        """Generate performance metrics table."""
        
        # Extract metrics
        metrics = [
            ["Metric", "Value"],
            ["Total Return", f"{backtest_results.get('total_return', 0):.2%}"],
            ["Annualized Return", f"{backtest_results.get('annualized_return', 0):.2%}"],
            ["Annualized Volatility", f"{backtest_results.get('annualized_volatility', 0):.2%}"],
            ["Sharpe Ratio", f"{backtest_results.get('sharpe_ratio', 0):.2f}"],
            ["Sortino Ratio", f"{backtest_results.get('sortino_ratio', 0):.2f}"],
            ["Calmar Ratio", f"{backtest_results.get('calmar_ratio', 0):.2f}"],
            ["Maximum Drawdown", f"{backtest_results.get('max_drawdown', 0):.2%}"],
            ["VaR (95%)", f"{backtest_results.get('var_95', 0):.2%}"],
            ["CVaR (95%)", f"{backtest_results.get('cvar_95', 0):.2%}"],
        ]
        
        # Create table
        table = Table(metrics, colWidths=[2*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _generate_risk_analysis(self, backtest_results: Dict[str, Any]) -> Paragraph:
        """Generate risk analysis paragraph."""
        
        # Extract risk metrics
        max_drawdown = backtest_results.get("max_drawdown", 0)
        var_95 = backtest_results.get("var_95", 0)
        cvar_95 = backtest_results.get("cvar_95", 0)
        volatility = backtest_results.get("annualized_volatility", 0)
        
        # Generate risk analysis text
        risk_text = f"""
        <b>Risk Assessment:</b><br/>
        
        The portfolio exhibits a maximum drawdown of {max_drawdown:.2%}, which represents 
        the largest peak-to-trough decline during the analysis period. The Value at Risk 
        (VaR) at the 95% confidence level is {var_95:.2%}, meaning there is a 5% probability 
        of experiencing losses greater than this amount on any given day. The Conditional 
        Value at Risk (CVaR) is {cvar_95:.2%}, representing the expected loss given that 
        the VaR threshold is exceeded.
        
        The annualized volatility of {volatility:.2%} indicates the portfolio's sensitivity 
        to market fluctuations. {'This level of volatility is considered ' + ('low' if volatility < 0.10 else 'moderate' if volatility < 0.20 else 'high') + ' for a diversified portfolio.'}
        """
        
        return Paragraph(risk_text, self.styles['CustomBody'])
    
    def _generate_portfolio_composition(self, backtest_results: Dict[str, Any]) -> Paragraph:
        """Generate portfolio composition analysis."""
        
        # Extract portfolio weights if available
        weights = backtest_results.get("final_weights", {})
        
        if weights:
            # Create weights table
            weight_data = [["Asset", "Weight"]]
            for asset, weight in weights.items():
                weight_data.append([asset, f"{weight:.2%}"])
            
            # Add table to story
            weight_table = Table(weight_data, colWidths=[2*inch, 1*inch])
            weight_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            return weight_table
        else:
            return Paragraph("Portfolio composition data not available.", self.styles['CustomBody'])
    
    def _generate_constraints_analysis(self, policy_config: Dict[str, Any]) -> Paragraph:
        """Generate constraints analysis paragraph."""
        
        # Extract constraint information
        asset_constraints = policy_config.get("asset_constraints", {})
        risk_constraints = policy_config.get("risk_constraints", {})
        esg_constraints = policy_config.get("esg_constraints", {})
        
        # Generate constraints text
        constraints_text = f"""
        <b>Constraints Applied:</b><br/>
        
        The portfolio optimization was subject to the following constraints based on your 
        investment policy:
        
        <b>Asset Allocation Constraints:</b><br/>
        • Fixed Income: {asset_constraints.get('min_fixed_income', 0):.0%} - {asset_constraints.get('max_fixed_income', 100):.0%}<br/>
        • Equity: {asset_constraints.get('min_equity', 0):.0%} - {asset_constraints.get('max_equity', 100):.0%}<br/>
        • Commodities: {asset_constraints.get('min_commodities', 0):.0%} - {asset_constraints.get('max_commodities', 100):.0%}<br/>
        
        <b>Risk Constraints:</b><br/>
        • Maximum Drawdown: {risk_constraints.get('max_drawdown', 0):.0%}<br/>
        • Maximum Volatility: {risk_constraints.get('max_volatility', 0):.0%}<br/>
        
        <b>ESG Constraints:</b><br/>
        • ESG Screening: {'Enabled' if esg_constraints.get('enabled', False) else 'Disabled'}<br/>
        • Minimum ESG Score: {esg_constraints.get('min_esg_score', 0):.0f}<br/>
        """
        
        return Paragraph(constraints_text, self.styles['CustomBody'])
    
    def _generate_recommendations(
        self,
        backtest_results: Dict[str, Any],
        policy_config: Dict[str, Any]
    ) -> Paragraph:
        """Generate recommendations paragraph."""
        
        # Extract key metrics for recommendations
        sharpe_ratio = backtest_results.get("sharpe_ratio", 0)
        max_drawdown = backtest_results.get("max_drawdown", 0)
        turnover = backtest_results.get("annual_turnover", 0)
        
        # Generate recommendations based on performance
        recommendations = []
        
        if sharpe_ratio > 1.0:
            recommendations.append("The portfolio demonstrates strong risk-adjusted performance.")
        elif sharpe_ratio > 0.5:
            recommendations.append("The portfolio shows moderate risk-adjusted performance.")
        else:
            recommendations.append("Consider reviewing the portfolio strategy to improve risk-adjusted returns.")
        
        if abs(max_drawdown) > 0.20:
            recommendations.append("The maximum drawdown exceeds typical risk tolerance levels.")
        
        if turnover > 0.50:
            recommendations.append("High portfolio turnover may increase transaction costs.")
        
        # Generate recommendations text
        recommendations_text = f"""
        <b>Recommendations:</b><br/>
        
        Based on the analysis results, the following recommendations are provided:
        
        • {recommendations[0] if recommendations else "The portfolio meets the specified investment objectives."}
        • {'• '.join(recommendations[1:]) if len(recommendations) > 1 else ""}
        
        <b>Next Steps:</b><br/>
        
        1. Review the portfolio composition and risk characteristics<br/>
        2. Consider rebalancing if significant deviations from target weights occur<br/>
        3. Monitor market conditions and adjust strategy as needed<br/>
        4. Regular performance review and optimization updates<br/>
        """
        
        return Paragraph(recommendations_text, self.styles['CustomBody'])
    
    def _generate_disclaimer(self) -> Paragraph:
        """Generate disclaimer paragraph."""
        
        disclaimer_text = """
        <b>Disclaimer:</b><br/>
        
        This report is for informational purposes only and does not constitute investment advice. 
        Past performance is not indicative of future results. The information contained herein 
        is based on historical data and assumptions that may not hold in the future. 
        
        Investors should carefully consider their investment objectives, risk tolerance, and 
        financial situation before making any investment decisions. The Omega Portfolio Engine 
        is a research tool and should not be used as the sole basis for investment decisions.
        
        This report is confidential and intended solely for the use of the client. 
        Distribution to third parties is prohibited without prior written consent.
        """
        
        return Paragraph(disclaimer_text, self.styles['CustomBody'])
    
    def generate_summary_report(
        self,
        backtest_results: Dict[str, Any],
        policy_config: Dict[str, Any],
        output_filename: Optional[str] = None
    ) -> Path:
        """Generate a summary report (shorter version)."""
        
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"summary_report_{timestamp}.pdf"
        
        output_path = self.output_dir / output_filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build content
        story = []
        
        # Title
        story.append(Paragraph("Portfolio Optimization Summary", self.styles['CustomTitle']))
        story.append(Spacer(1, 12))
        
        # Performance Metrics
        story.append(Paragraph("Performance Metrics", self.styles['CustomHeading']))
        story.append(self._generate_performance_table(backtest_results))
        story.append(Spacer(1, 12))
        
        # Key Recommendations
        story.append(Paragraph("Key Recommendations", self.styles['CustomHeading']))
        story.append(self._generate_recommendations(backtest_results, policy_config))
        
        # Build PDF
        doc.build(story)
        
        logger.info("Summary report generated", output_path=str(output_path))
        return output_path
