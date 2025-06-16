import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Initialize sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

class ReligiousEthicsAnalyzer:
    def __init__(self):
        # Define core ethical principles for major religions
        self.religious_principles = {
            'Christianity': {
                'keywords': ['love', 'compassion', 'forgiveness', 'charity', 'service', 'sacrifice', 
                           'mercy', 'justice', 'truth', 'humility', 'peace', 'kindness'],
                'negative_keywords': ['hatred', 'violence', 'greed', 'pride', 'revenge', 'dishonesty',
                                    'cruelty', 'selfishness', 'harm', 'exploitation'],
                'weight_compassion': 0.4,
                'weight_justice': 0.3,
                'weight_service': 0.3
            },
            'Islam': {
                'keywords': ['justice', 'compassion', 'charity', 'honesty', 'respect', 'moderation',
                           'peace', 'mercy', 'brotherhood', 'responsibility', 'wisdom', 'patience'],
                'negative_keywords': ['injustice', 'oppression', 'dishonesty', 'excess', 'hatred',
                                    'corruption', 'harm', 'exploitation', 'violence', 'greed'],
                'weight_justice': 0.4,
                'weight_compassion': 0.3,
                'weight_community': 0.3
            },
            'Judaism': {
                'keywords': ['justice', 'righteousness', 'repair', 'responsibility', 'learning',
                           'community', 'ethics', 'compassion', 'wisdom', 'truth', 'peace', 'charity'],
                'negative_keywords': ['injustice', 'ignorance', 'selfishness', 'hatred', 'destruction',
                                    'dishonesty', 'harm', 'oppression', 'violence', 'cruelty'],
                'weight_justice': 0.4,
                'weight_learning': 0.3,
                'weight_community': 0.3
            },
            'Hinduism': {
                'keywords': ['dharma', 'karma', 'ahimsa', 'truth', 'duty', 'righteousness',
                           'non-violence', 'compassion', 'service', 'wisdom', 'balance', 'harmony'],
                'negative_keywords': ['adharma', 'violence', 'dishonesty', 'selfishness', 'harm',
                                    'injustice', 'cruelty', 'ignorance', 'hatred', 'imbalance'],
                'weight_dharma': 0.4,
                'weight_ahimsa': 0.3,
                'weight_karma': 0.3
            },
            'Buddhism': {
                'keywords': ['compassion', 'wisdom', 'mindfulness', 'non-harm', 'moderation',
                           'peace', 'understanding', 'loving-kindness', 'liberation', 'balance',
                           'awareness', 'enlightenment'],
                'negative_keywords': ['suffering', 'attachment', 'ignorance', 'hatred', 'greed',
                                    'violence', 'harm', 'delusion', 'cruelty', 'selfishness'],
                'weight_compassion': 0.4,
                'weight_wisdom': 0.3,
                'weight_non_harm': 0.3
            },
            'Sikhism': {
                'keywords': ['equality', 'service', 'truth', 'justice', 'compassion', 'humility',
                           'sharing', 'devotion', 'courage', 'righteousness', 'unity', 'honesty'],
                'negative_keywords': ['inequality', 'selfishness', 'dishonesty', 'injustice', 'pride',
                                    'hatred', 'violence', 'discrimination', 'greed', 'harm'],
                'weight_equality': 0.4,
                'weight_service': 0.3,
                'weight_truth': 0.3
            }
        }
        
        # Ethical scenarios for contextual analysis
        self.ethical_contexts = {
            'harm_reduction': ['harm', 'safety', 'protection', 'welfare', 'wellbeing'],
            'fairness': ['fair', 'equal', 'just', 'equitable', 'impartial'],
            'autonomy': ['choice', 'freedom', 'consent', 'independence', 'self-determination'],
            'beneficence': ['benefit', 'good', 'help', 'support', 'improvement'],
            'transparency': ['transparent', 'open', 'honest', 'clear', 'accountable']
        }

    def preprocess_text(self, text):
        """Clean and preprocess text for analysis"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def calculate_religious_alignment(self, text, religion):
        """Calculate how well text aligns with religious principles"""
        processed_text = self.preprocess_text(text)
        principles = self.religious_principles[religion]
        
        # Count positive and negative keyword matches
        positive_matches = sum(1 for keyword in principles['keywords'] 
                             if keyword in processed_text)
        negative_matches = sum(1 for keyword in principles['negative_keywords'] 
                             if keyword in processed_text)
        
        # Calculate base alignment score
        total_keywords = len(principles['keywords']) + len(principles['negative_keywords'])
        positive_ratio = positive_matches / len(principles['keywords']) if principles['keywords'] else 0
        negative_ratio = negative_matches / len(principles['negative_keywords']) if principles['negative_keywords'] else 0
        
        # Base score (0-1 scale)
        base_score = max(0, positive_ratio - negative_ratio)
        
        # Apply sentiment analysis
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity
        vader_scores = vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        # Combine sentiment scores
        combined_sentiment = (textblob_sentiment + vader_compound) / 2
        
        # Final score calculation (weighted combination)
        final_score = (base_score * 0.6 + (combined_sentiment + 1) / 2 * 0.4)
        
        return min(max(final_score, 0), 1)  # Ensure score is between 0 and 1

    def analyze_ethical_context(self, text):
        """Analyze the ethical context of the scenario"""
        processed_text = self.preprocess_text(text)
        context_scores = {}
        
        for context, keywords in self.ethical_contexts.items():
            matches = sum(1 for keyword in keywords if keyword in processed_text)
            context_scores[context] = matches / len(keywords) if keywords else 0
        
        return context_scores

    def generate_ethics_report(self, scenario):
        """Generate comprehensive ethics assessment"""
        # Calculate alignment scores for each religion
        religious_scores = {}
        for religion in self.religious_principles.keys():
            score = self.calculate_religious_alignment(scenario, religion)
            religious_scores[religion] = score

        # Analyze ethical contexts
        context_analysis = self.analyze_ethical_context(scenario)
        
        # Calculate overall ethics score
        overall_score = np.mean(list(religious_scores.values()))
        
        # Generate insights
        insights = self.generate_insights(religious_scores, context_analysis, scenario)
        
        return {
            'religious_scores': religious_scores,
            'context_analysis': context_analysis,
            'overall_score': overall_score,
            'insights': insights
        }

    def generate_insights(self, religious_scores, context_analysis, scenario):
        """Generate actionable insights from the analysis"""
        insights = []
        
        # Religious perspective insights
        highest_score_religion = max(religious_scores, key=religious_scores.get)
        lowest_score_religion = min(religious_scores, key=religious_scores.get)
        
        insights.append(f"**Religious Alignment**: {highest_score_religion} tradition shows highest alignment "
                       f"({religious_scores[highest_score_religion]:.2f}), while {lowest_score_religion} "
                       f"shows lowest alignment ({religious_scores[lowest_score_religion]:.2f}).")
        
        # Context analysis insights
        dominant_context = max(context_analysis, key=context_analysis.get)
        if context_analysis[dominant_context] > 0.3:
            insights.append(f"**Primary Ethical Context**: The scenario primarily involves "
                           f"considerations of {dominant_context.replace('_', ' ')}.")
        
        # Overall assessment
        overall_avg = np.mean(list(religious_scores.values()))
        if overall_avg > 0.7:
            insights.append("**Overall Assessment**: The scenario shows strong ethical alignment across traditions.")
        elif overall_avg > 0.4:
            insights.append("**Overall Assessment**: The scenario shows moderate ethical considerations with room for improvement.")
        else:
            insights.append("**Overall Assessment**: The scenario raises significant ethical concerns across multiple traditions.")
        
        return insights

# Initialize the analyzer with caching
@st.cache_resource
def load_analyzer():
    return ReligiousEthicsAnalyzer()

def create_visualizations(results):
    """Create interactive visualizations for the results"""
    
    # Religious Scores Radar Chart
    religions = list(results['religious_scores'].keys())
    scores = list(results['religious_scores'].values())
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=scores,
        theta=religions,
        fill='toself',
        name='Ethical Alignment',
        line_color='rgb(46, 204, 113)',
        fillcolor='rgba(46, 204, 113, 0.3)'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Religious Ethics Alignment Scores",
        font=dict(size=12)
    )
    
    # Context Analysis Bar Chart
    contexts = list(results['context_analysis'].keys())
    context_scores = list(results['context_analysis'].values())
    
    fig_bar = px.bar(
        x=[ctx.replace('_', ' ').title() for ctx in contexts],
        y=context_scores,
        title="Ethical Context Analysis",
        labels={'x': 'Ethical Contexts', 'y': 'Relevance Score'},
        color=context_scores,
        color_continuous_scale='Viridis'
    )
    fig_bar.update_layout(showlegend=False)
    
    return fig_radar, fig_bar

def main():
    st.set_page_config(
        page_title="Multi-Religious Ethics Assessment System",
        page_icon="⚖️",
        layout="wide"
    )
    
    # Header
    st.title("⚖️ Multi-Religious Ethics Assessment System")
    st.markdown("*An AI-powered tool for ethical analysis through diverse religious perspectives*")
    
    # Sidebar information
    with st.sidebar:
        st.header("About This System")
        st.markdown("""
        This system analyzes ethical scenarios using:
        - **6 Major Religious Traditions**: Christianity, Islam, Judaism, Hinduism, Buddhism, Sikhism
        - **Advanced NLP**: TextBlob & VADER sentiment analysis
        - **Contextual Analysis**: Multiple ethical frameworks
        - **Machine Learning**: Weighted scoring algorithms
        """)
        
        st.header("How It Works")
        st.markdown("""
        1. **Text Processing**: Cleans and analyzes input text
        2. **Religious Alignment**: Scores alignment with each tradition
        3. **Context Analysis**: Identifies primary ethical contexts
        4. **Insight Generation**: Provides actionable recommendations
        """)
    
    # Main interface
    analyzer = load_analyzer()
    
    # Input section
    st.header("📝 Ethical Scenario Analysis")
    
    # Provide example scenarios
    example_scenarios = {
        "Select an example...": "",
        "AI Healthcare Decision": "An AI system must decide whether to prioritize treating a young patient with higher survival chances or an elderly patient who arrived first at the emergency room.",
        "Data Privacy vs Public Safety": "A government wants to use personal data from smartphones to track disease spread during a pandemic, potentially saving lives but violating privacy.",
        "Autonomous Vehicle Dilemma": "A self-driving car must choose between hitting one person to avoid hitting five people, or continuing straight and potentially harming more people.",
        "Employment AI Bias": "An AI hiring system shows better performance but demonstrates bias against certain ethnic groups in recruitment decisions."
    }
    
    selected_example = st.selectbox("Choose an example scenario or write your own:", list(example_scenarios.keys()))
    
    if selected_example != "Select an example...":
        default_text = example_scenarios[selected_example]
    else:
        default_text = ""
    
    scenario = st.text_area(
        "Enter your ethical scenario for analysis:",
        value=default_text,
        height=150,
        placeholder="Describe an ethical dilemma or decision scenario you'd like to analyze..."
    )
    
    if st.button("🔍 Analyze Ethics", type="primary") and scenario.strip():
        with st.spinner("Analyzing ethical implications across religious traditions..."):
            results = analyzer.generate_ethics_report(scenario)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📊 Analysis Results")
            
            # Overall Score
            overall_score = results['overall_score']
            score_color = "green" if overall_score > 0.7 else "orange" if overall_score > 0.4 else "red"
            
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin: 10px 0;">
                <h3 style="color: {score_color};">Overall Ethics Score: {overall_score:.2f}/1.00</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualizations
            fig_radar, fig_bar = create_visualizations(results)
            
            st.plotly_chart(fig_radar, use_container_width=True)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.header("🎯 Key Insights")
            for insight in results['insights']:
                st.markdown(f"• {insight}")
            
            st.header("📋 Detailed Scores")
            
            # Religious scores table
            religious_df = pd.DataFrame(
                list(results['religious_scores'].items()),
                columns=['Religion', 'Alignment Score']
            )
            religious_df['Alignment Score'] = religious_df['Alignment Score'].round(3)
            religious_df = religious_df.sort_values('Alignment Score', ascending=False)
            
            st.dataframe(religious_df, use_container_width=True)
            
            # Context analysis
            st.subheader("Ethical Context Relevance")
            for context, score in results['context_analysis'].items():
                context_name = context.replace('_', ' ').title()
                st.progress(score, text=f"{context_name}: {score:.2f}")
    
    elif st.button("🔍 Analyze Ethics", type="primary"):
        st.warning("Please enter a scenario to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Multi-Religious Ethics Assessment System | Promoting Inclusive AI Governance</p>
        <p><em>This tool provides perspective-based analysis and should complement, not replace, human ethical judgment.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()