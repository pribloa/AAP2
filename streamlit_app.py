# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="Productivity Prediction & AI Report",
    page_icon="🤖",
    layout="wide"
)

st.title("🏭 Factory Productivity Report Generator")
st.write("""
Upload a CSV file with worker data to get a productivity analysis.
The app will predict productivity scores and generate AI-powered reports.
""")

# --- Model and Preprocessor Loading ---
@st.cache_resource
def load_model_and_preprocessor():
    MODEL_PATH = 'productivity_prediction_model.joblib'
    PREPROCESSOR_PATH = 'fitted_preprocessor.joblib'
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        st.success("✅ Machine Learning Model and Preprocessor loaded successfully.")
        return model, preprocessor
    except FileNotFoundError as e:
        st.error(f"❌ Error: {e}. Please ensure '{MODEL_PATH}' and '{PREPROCESSOR_PATH}' are uploaded.")
        return None, None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during model loading: {e}")
        return None, None

model, preprocessor = load_model_and_preprocessor()

# --- OpenRouter API Utility Function ---
@st.cache_data(show_spinner="⏳ Generating AI Reports... This may take a moment.")
def call_openrouter_api(prompt_content, model_name="deepseek/deepseek-chat-v3-0324"):
    # Debug Statement 1: Confirm the API key is retrieved
    OPENROUTER_KEY_RAW = st.secrets.get("OPENROUTER_KEY")
    st.info(f"DEBUG: Retrieved key from st.secrets. Is it None? {'Yes' if OPENROUTER_KEY_RAW is None else 'No'}")

    if not OPENROUTER_KEY_RAW:
        st.error("❌ ERROR: OPENROUTER_KEY not configured. Please add it as a secret.")
        return "ERROR: OPENROUTER_KEY not configured. Cannot generate AI reports."

    # Remove any potential leading/trailing whitespace
    OPENROUTER_KEY = OPENROUTER_KEY_RAW.strip()

    # Debug Statement 2: Check the length of the retrieved key
    st.info(f"DEBUG: Length of the API Key (after strip): {len(OPENROUTER_KEY)}")

    # Construct the Authorization header
    auth_header = f"Bearer {OPENROUTER_KEY}"
    
    # Debug Statement 3: Show the exact header being sent
    st.info("DEBUG: Full Authorization Header being sent to OpenRouter:")
    st.code(auth_header)

    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json",
    }
    
    # Debug Statement 4: Confirm the URL and full headers
    st.info(f"DEBUG: Requesting URL: {OPENROUTER_API_URL}")
    st.write("DEBUG: Request Headers:")
    st.json(headers)

    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt_content}
        ]
    }
    
    # Debug Statement 5: Show the JSON payload
    st.write("DEBUG: Request Body (JSON payload):")
    st.json(data)
    
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status() # This is the line that raises the 401
        
        # Debug Statement 6: Success!
        st.success("DEBUG: API call successful!")
        
        report_response = response.json()
        return report_response['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error communicating with OpenRouter API: {e}")
        st.error(f"API Error: {e}")
        return f"API Error: {e}"
    except KeyError:
        st.error("❌ Error: Could not parse response from OpenRouter API.")
        return "API Error: Malformed response."

# Define expected features outside of the function for reusability
expected_features_for_prediction = [
    'location_type', 'industry_sector', 'age', 'experience_years',
    'average_daily_work_hours', 'break_frequency_per_day',
    'task_completion_rate', 'tool_usage_frequency',
    'automated_task_count', 'AI_assisted_planning',
    'real_time_feedback_score'
]

# --- Grading Logic for Leaderboard ---
def assign_grade(score, cutoffs):
    if score >= cutoffs['S']:
        return 'S'
    elif score >= cutoffs['A']:
        return 'A'
    elif score >= cutoffs['B']:
        return 'B'
    elif score >= cutoffs['C']:
        return 'C'
    elif score >= cutoffs['D']:
        return 'D'
    elif score >= cutoffs['E']:
        return 'E'
    else:
        return 'F'

# --- Main Prediction and Reporting Logic ---
def run_analysis(uploaded_file):
    if model is None or preprocessor is None:
        return

    try:
        # Load and sample data
        report_df = pd.read_csv(uploaded_file)
        if len(report_df) > 50:
            report_df_sampled = report_df.sample(n=50, random_state=42).copy()
        else:
            report_df_sampled = report_df.copy()

        if 'worker_id' not in report_df_sampled.columns:
            st.error("❌ Uploaded CSV must contain a 'worker_id' column.")
            return

        # Handle missing columns
        for col in expected_features_for_prediction:
            if col not in report_df_sampled.columns:
                report_df_sampled[col] = np.nan
                st.warning(f"⚠️ Warning: Column '{col}' was missing from the uploaded file and has been filled with NaN.")

        # Select features for prediction
        X_report = report_df_sampled[expected_features_for_prediction]

        # Data preparation and prediction
        X_report_processed = preprocessor.transform(X_report)
        report_df_sampled['predicted_productivity_score'] = model.predict(X_report_processed)

        # --- Assign Grades for Leaderboard ---
        quantiles = report_df_sampled['predicted_productivity_score'].quantile([0.05, 0.20, 0.40, 0.60, 0.80, 0.95])
        grade_cutoffs = {
            'S': quantiles[0.95],
            'A': quantiles[0.80],
            'B': quantiles[0.60],
            'C': quantiles[0.40],
            'D': quantiles[0.20],
            'E': quantiles[0.05],
            'F': -np.inf # A value lower than any possible score
        }
        report_df_sampled['Grade'] = report_df_sampled['predicted_productivity_score'].apply(lambda x: assign_grade(x, grade_cutoffs))

        # --- Display Predicted Scores ---
        st.subheader("Predicted Productivity Scores")
        st.dataframe(report_df_sampled[['worker_id', 'predicted_productivity_score']].round(2), use_container_width=True)
        
        # Download button for results
        csv_output = report_df_sampled[['worker_id', 'predicted_productivity_score']].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv_output,
            file_name="productivity_predictions.csv",
            mime="text/csv",
        )

        st.divider()
        st.subheader("AI-Powered Productivity Reports")
        
        # Use st.tabs to organize the reports
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overall Report", "👨‍🏫 Personalized Advice", "🔍 Single Worker Analysis", "🏆 Leaderboard"])

        with tab1:
            st.header("Overall Managerial Report")
            # Logic for overall report
            sorted_workers = report_df_sampled.sort_values(by='predicted_productivity_score', ascending=False)
            high_performers_for_report = sorted_workers.head(2).copy()
            low_performers_for_report = sorted_workers.tail(2).copy()
            
            standout_workers_str = "\n**Standout Workers Analysis:**\n"
            if not high_performers_for_report.empty:
                standout_workers_str += "\n**Top 2 Predicted High Performers:**\n"
                for _, row in high_performers_for_report.iterrows():
                    standout_workers_str += f"- Worker ID: {row['worker_id']}, Predicted Score: {row['predicted_productivity_score']:.2f}\n"
            else:
                standout_workers_str += "No clear high performers identified in this sample.\n"

            if not low_performers_for_report.empty:
                standout_workers_str += "\n**Bottom 2 Predicted Low Performers:**\n"
                for _, row in low_performers_for_report.iterrows():
                    standout_workers_str += f"- Worker ID: {row['worker_id']}, Predicted Score: {row['predicted_productivity_score']:.2f}\n"
            else:
                standout_workers_str += "No clear low performers identified in this sample.\n"

            overall_report_prompt = f"""
            You are an experienced factory manufacturing manager reviewing a concise productivity report.
            This report is based on sampled worker data and their predicted productivity scores from a machine learning model.
            
            Your report should be structured as follows:
            
            ## Overall Productivity Snapshot
            - Provide a general assessment of the predicted productivity scores across the sampled workers (e.g., average, range, common trends).
            
            ## Standout Workers
            {standout_workers_str}
            - Based on the data, what are the characteristics or activity patterns that distinguish these high and low performers? (e.g., high/low task completion, time on platform, etc.).
            - What specific metrics seem to be strongly associated with their predicted productivity?
            
            ## Recommendations for Next Steps
            - Based on the overall trends and the analysis of standout workers, provide 3-5 high-level, actionable recommendations for improving factory floor productivity.
            - Focus on practical, implementable actions relevant to a manufacturing environment (e.g., training, process optimization, resource allocation, feedback mechanisms).
            
            Here is the sampled worker data with original (if available) and predicted productivity scores for your reference:
            
            {report_df_sampled.to_markdown(index=False)}
            
            Ensure the report is concise, directly actionable, and written from the perspective of a factory manager to their leadership.
            """
            overall_report_content = call_openrouter_api(overall_report_prompt)
            st.markdown(overall_report_content)

        with tab2:
            st.header("Personalized Worker Advice")
            # Logic for personalized advice
            workers_for_coaching = pd.DataFrame()
            if not high_performers_for_report.empty:
                workers_for_coaching = pd.concat([workers_for_coaching, high_performers_for_report.sample(n=1, random_state=42)]).drop_duplicates(subset=['worker_id'])
            if not low_performers_for_report.empty and len(workers_for_coaching) < 2:
                unique_low_performers = low_performers_for_report[~low_performers_for_report['worker_id'].isin(workers_for_coaching['worker_id'])]
                if not unique_low_performers.empty:
                    workers_for_coaching = pd.concat([workers_for_coaching, unique_low_performers.sample(n=1, random_state=42)]).drop_duplicates(subset=['worker_id'])

            if len(workers_for_coaching) < 2:
                remaining_workers = report_df_sampled[~report_df_sampled['worker_id'].isin(workers_for_coaching['worker_id'])]
                if len(remaining_workers) >= (2 - len(workers_for_coaching)):
                    workers_for_coaching = pd.concat([workers_for_coaching, remaining_workers.sample(n=(2 - len(workers_for_coaching)), random_state=42)]).drop_duplicates(subset=['worker_id'])
                elif not remaining_workers.empty:
                    workers_for_coaching = pd.concat([workers_for_coaching, remaining_workers]).drop_duplicates(subset=['worker_id'])

            personalized_advice_content = ""
            if workers_for_coaching.empty:
                personalized_advice_content = "Not enough unique workers in the sample to generate personalized advice."
            else:
                personalized_coaching_prompt_parts = []
                for _, worker_row in workers_for_coaching.iterrows():
                    worker_id = worker_row['worker_id']
                    worker_data_for_prompt = {k: v for k, v in worker_row.to_dict().items() if k in expected_features_for_prediction or k == 'predicted_productivity_score'}
                    
                    personalized_coaching_prompt_parts.append(f"""
### Advice for Worker ID: {worker_id} (Predicted Score: {worker_row['predicted_productivity_score']:.2f})
Worker's Data:
{json.dumps(worker_data_for_prompt, indent=2)}

-   **Acknowledgement**: Briefly acknowledge their current predicted productivity level.
-   **Strengths/Areas for Improvement**: Based on their specific metrics (e.g., 'task_completion_rate', 'daily_work_minutes', etc.), identify potential strengths or areas where improvement could lead to higher productivity. Reference specific metrics constructively.
-   **Specific Actionable Tips**: Provide 1-2 concrete, practical tips relevant to a factory worker, directly linking to their data.
-   **Overall Encouragement**: End with a motivational statement.
""")

                final_personalized_coach_prompt = f"""
You are an AI-powered "Work Style & Optimization Coach" providing personalized advice to **two selected factory workers**.
Your goal is to provide constructive feedback and actionable steps to help each worker improve their productivity within a manufacturing setting, based on their individual performance metrics and predicted productivity scores.

Please generate a separate, clear coaching message for each of the two workers below. Ensure each message is distinct and directly addresses the specific data provided for that worker.

{''.join(personalized_coaching_prompt_parts)}

Keep the tone supportive, direct, and practical for factory workers.
"""
                personalized_advice_content = call_openrouter_api(final_personalized_coach_prompt)
            
            st.markdown(personalized_advice_content)

        with tab3:
            st.header("Single Worker Analysis")
            
            # Create a select box to choose a worker
            worker_ids = sorted(report_df_sampled['worker_id'].unique().tolist())
            selected_worker_id = st.selectbox(
                "Select a Worker ID to get a detailed analysis:",
                options=[''] + worker_ids # Add an empty option
            )

            if selected_worker_id:
                # Find the data for the selected worker
                worker_data = report_df_sampled[report_df_sampled['worker_id'] == selected_worker_id].iloc[0]
                
                # Create the prompt for a single worker
                single_worker_prompt = f"""
                You are a "Productivity Analyst" with expertise in manufacturing. Your task is to provide a detailed analysis of a single worker's performance based on their data.
                
                Analyze the following data for Worker ID: {selected_worker_id}.
                
                **Worker Data:**
                {json.dumps({k: v for k, v in worker_data.to_dict().items() if k in expected_features_for_prediction or k == 'predicted_productivity_score'}, indent=2)}
                
                Your analysis should be structured as follows:
                
                ### Performance Snapshot
                - Briefly summarize the worker's predicted productivity score and key metrics.
                
                ### Strengths & Weaknesses
                - Identify specific metrics that stand out as strengths (e.g., high task completion rate, high tool usage).
                - Identify areas for potential improvement (e.g., low average daily work hours, low real-time feedback score).
                
                ### Actionable Recommendations
                - Provide 2-3 specific, actionable recommendations for this worker to improve their productivity.
                - Link these recommendations directly to the data provided. For example, "Consider a short training session on new tools to increase their tool_usage_frequency."
                
                Ensure the tone is professional, constructive, and encouraging.
                """

                # Call the API and display the result
                single_worker_report = call_openrouter_api(single_worker_prompt)
                st.markdown(single_worker_report)

        with tab4:
            st.header("Productivity Leaderboard")
            st.markdown("""
            This leaderboard ranks all workers by their predicted productivity score and assigns a relative grade.
            <div style="font-size: 1.5em; color: gold; font-weight: bold;">S: Top 5%</div>
            <div style="color: green;">A: 80% to 95%</div>
            <div style="color: #6699FF;">B: 60% to 80%</div>
            <div style="color: gray;">C: 40% to 60%</div>
            <div style="color: #FF9933;">D: 20% to 40%</div>
            <div style="color: #FF6600;">E: 5% to 20%</div>
            <div style="color: red;">F: Bottom 5%</div>
            """, unsafe_allow_html=True)

            # Sort the DataFrame for the leaderboard
            leaderboard_df = report_df_sampled.sort_values(
                by='predicted_productivity_score',
                ascending=False
            ).reset_index(drop=True)

            # Function to apply color styling to the 'Grade' column
            def color_grades(val):
                if val == 'S':
                    return 'background-color: gold; color: white; font-weight: bold;'
                elif val == 'A':
                    return 'color: green;'
                elif val == 'B':
                    return 'color: #6699FF;'
                elif val == 'C':
                    return 'color: gray;'
                elif val == 'D':
                    return 'color: #FF9933;'
                elif val == 'E':
                    return 'color: #FF6600;'
                elif val == 'F':
                    return 'color: red;'
                return ''

            # Display the styled DataFrame
            st.dataframe(
                leaderboard_df[['worker_id', 'predicted_productivity_score', 'Grade']].style.applymap(color_grades, subset=['Grade']),
                use_container_width=True
            )


    except Exception as e:
        st.error(f"❌ An error occurred during processing: {e}")
        st.stop()


# --- Streamlit UI Flow ---
if __name__ == "__main__":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        run_analysis(uploaded_file)