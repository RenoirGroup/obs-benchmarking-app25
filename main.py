import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sqlalchemy import create_engine, text
import os
import io
import base64
from openai import OpenAI
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="OBS Benchmarking",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATABASE AND BENCHMARK CONFIGURATION ---
DB_HOST = "dpg-d2n7dmvdiees73c9e8t0-a.singapore-postgres.render.com"
DB_NAME = "obs_bench_0925"
DB_USER = "obs_bench_0925_user"
DB_PASSWORD = "ZVvS4VbSo8COhNPl7pGSgwZPSXOnsJqn"
DB_PORT = 5432

# --- MAPPINGS AND DEFINITIONS ---
theme_mapping_benchmark = {
    'A': 'Targets', 'B': 'Results', 'C': 'Costs', 'D': 'Routine', 'E': 'Quality',
    'F': 'System Focus', 'G': 'System Utility', 'H': 'Problem Solving', 'I': 'Meetings',
    'J': 'Changes', 'K': 'Leadership', 'L': 'Communication', 'M': 'Synergy'
}
theme_order = [
    'Targets', 'Results', 'Costs', 'Routine', 'Quality', 'System Focus',
    'System Utility', 'Problem Solving', 'Meetings', 'Changes', 'Leadership',
    'Communication', 'Synergy'
]
category_mapping = {
    'Targets': 'Result', 'Results': 'Result', 'Costs': 'Result',
    'Routine': 'Client', 'Quality': 'Client',
    'System Focus': 'Process', 'System Utility': 'Process', 'Problem Solving': 'Process', 'Meetings': 'Process',
    'Changes': 'People', 'Leadership': 'People', 'Communication': 'People', 'Synergy': 'People'
}
absentee_mapping = {
    'Targets': 'Unclear', 'Results': 'Consequence', 'Costs': 'Consequence',
    'Routine': 'Executors', 'Quality': 'Transfer', 'System Focus': 'Operational',
    'System Utility': 'Bureaucracy', 'Problem Solving': 'Empirical', 'Meetings': 'Bureaucratic',
    'Changes': 'Resistance', 'Leadership': 'Imposition', 'Communication': 'Unidirectional',
    'Synergy': 'Fragmented'
}
optimal_mapping = {
    'Targets': 'Clear', 'Results': 'Focused', 'Costs': 'Responsibility',
    'Routine': 'Managers', 'Quality': 'Commitment', 'System Focus': 'Results',
    'System Utility': 'Tools', 'Problem Solving': 'Methodical', 'Meetings': 'Effective',
    'Changes': 'Openness', 'Leadership': 'Condition', 'Communication': 'Two-Way',
    'Synergy': 'Integrated'
}
level_mapping = {
    'A': 'EVP VP Director',
    'B': 'Package Function Manager',
    'C': 'Team Leader',
    'D': 'Team Member'
}

# --- CORE FUNCTIONS ---

@st.cache_resource
def get_db_engine():
    try:
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"Error connecting to the database. Please check credentials in app.py. Error: {e}")
        return None

@st.cache_data(ttl=3600)
def load_benchmark_data():
    engine = get_db_engine()
    if not engine:
        return pd.DataFrame()
    try:
        query = text("""
            WITH RespondentScores AS (
                SELECT
                    sr.client_id,
                    sr.level_id,
                    sr.respondent_id,
                    ss.theme,
                    AVG((sr.response = ss.correct_answer)::INT) AS score
                FROM SurveyResponses sr
                JOIN SurveySummary ss ON sr.question_id = ss.question_id
                GROUP BY sr.client_id, sr.level_id, sr.respondent_id, ss.theme
            )
            SELECT
                c.client_name,
                sl.level_name,
                rs.respondent_id,
                rs.theme,
                rs.score
            FROM RespondentScores rs
            JOIN Clients c ON rs.client_id = c.client_id
            JOIN SurveyLevels sl ON rs.level_id = sl.level_id;
        """)
        
        df = pd.read_sql_query(query, engine)
        
        df['theme'] = df['theme'].map(theme_mapping_benchmark)
        df['category'] = df['theme'].map(category_mapping)
        return df

    except Exception as e:
        st.error(f"Failed to load benchmark data. Error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_database_stats():
    engine = get_db_engine()
    if not engine: return None
    with engine.connect() as connection:
        total_clients = connection.execute(text("SELECT COUNT(DISTINCT client_id) FROM Clients")).scalar_one()
        level_query = text("SELECT sl.level_name, COUNT(DISTINCT sr.respondent_id) as count FROM surveyresponses sr JOIN surveylevels sl ON sr.level_id = sl.level_id GROUP BY sl.level_name")
        level_counts = pd.read_sql(level_query, connection).set_index('level_name')['count'].to_dict()
        total_respondents = sum(level_counts.values())
    return {"total_clients": total_clients, "total_respondents": total_respondents, "level_counts": level_counts}

def append_to_database(new_client_df, client_name):
    engine = get_db_engine()
    if not engine:
        st.error("Database connection not available. Cannot append.")
        return
    connection = None
    try:
        connection = engine.connect()
        transaction = connection.begin()
        client_query = text("SELECT client_id FROM Clients WHERE client_name = :name")
        result = connection.execute(client_query, {'name': client_name}).fetchone()
        if result: client_id = result[0]
        else:
            st.info(f"Client '{client_name}' not found. Adding to database.")
            insert_query = text("INSERT INTO Clients (client_name) VALUES (:name) RETURNING client_id;")
            client_id = connection.execute(insert_query, {'name': client_name}).fetchone()[0]
        
        st.info("Checking for and removing any old data for this client to ensure a clean upload...")
        delete_query = text("DELETE FROM surveyresponses WHERE client_id = :id")
        connection.execute(delete_query, {'id': client_id})

        levels_map = pd.read_sql("SELECT level_id, level_name FROM SurveyLevels", connection).set_index('level_name')['level_id'].to_dict()
        questions_map = pd.read_sql("SELECT question_id, theme, question_number FROM SurveySummary", connection)
        questions_map['theme'] = questions_map['theme'].map(theme_mapping_benchmark)
        df_to_append = new_client_df.copy()
        df_to_append['client_id'] = client_id
        df_to_append['level_id'] = df_to_append['level_name'].map(levels_map)
        df_to_append = pd.merge(df_to_append, questions_map, on=['theme', 'question_number'], how='left')
        
        if df_to_append['question_id'].isnull().any() or df_to_append['level_id'].isnull().any():
            raise ValueError("Data integrity issue: Could not map all rows to a valid Question ID or Level ID. Please check if the Excel file data matches the database structure.")

        df_for_sql = df_to_append[['client_id', 'level_id', 'question_id', 'respondent_id', 'response']]
        st.info(f"Inserting {len(df_for_sql)} new responses...")
        df_for_sql.to_sql('surveyresponses', connection, if_exists='append', index=False, method='multi', chunksize=500)
        
        transaction.commit()
        
        st.success(f"Successfully appended {len(df_for_sql)} responses for client '{client_name}' to the database!")
        st.cache_data.clear()
        get_database_stats.clear() 
        st.rerun()
    except Exception as e:
        if 'transaction' in locals() and transaction.is_active: transaction.rollback()
        st.error(f"An error occurred during the database append operation: {e}")
    finally:
        if connection is not None: connection.close()

def get_ai_analysis(image_bytes, client_metrics, benchmark_metrics, client_name, benchmark_level):
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please set it as an Environment Variable in Render.")
            return None
        client = OpenAI(api_key=api_key)
        base64_image = base64.b64encode(image_bytes.read()).decode('utf-8')
        data_summary = "Client Data:\n" + client_metrics[['theme', 'avg_score']].to_string(index=False)
        data_summary += "\n\nBenchmark Data:\n" + benchmark_metrics[['theme', 'avg_score']].to_string(index=False)
        prompt = f"""
        You are an expert management consultant providing an executive summary...
        """ # Omitted long prompt for brevity
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}], max_tokens=800)
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred during AI analysis: {e}")
        return None

def find_column_names(df_columns):
    normalized_cols = {str(col).strip().lower(): col for col in df_columns}
    aliases = {'theme_code': ['habil x attit', 'temas'], 'correct_answer': ['gabarito', 'correct answer', 'answer key'], 'question_number': ['questões', 'média pts', 'media pts', 'question number', 'question_no']}
    found_columns = {}
    for standard_name, alias_list in aliases.items():
        for alias in alias_list:
            if alias in normalized_cols:
                found_columns[standard_name] = normalized_cols[alias]
                break
    return found_columns

def process_new_client_data(uploaded_file, client_name):
    uploaded_file.seek(0)
    st.session_state.uploaded_file_bytes = uploaded_file.read()
    try:
        xls = pd.ExcelFile(io.BytesIO(st.session_state.uploaded_file_bytes))
        valid_sheets = [sheet for sheet in xls.sheet_names if sheet in ['A', 'B', 'C', 'D']]
        if not valid_sheets:
            st.error("Uploaded file must contain worksheets named 'A', 'B', 'C', or 'D'.")
            return None
        all_sheets_data = []
        for sheet_name in valid_sheets:
            df_sheet = pd.read_excel(xls, sheet_name=sheet_name, header=None, engine='openpyxl')
            if len(df_sheet) < 3: continue
            new_headers = df_sheet.iloc[2]
            df_sheet.columns = new_headers
            df_sheet = df_sheet.drop([0, 1, 2]).reset_index(drop=True)
            df_sheet.dropna(how='all', inplace=True)
            found_cols = find_column_names(df_sheet.columns)
            required = ['theme_code', 'correct_answer', 'question_number']
            if not all(k in found_cols for k in required):
                st.warning(f"Skipping sheet {sheet_name}: Could not find required columns.")
                continue
            rename_mapping = {v: k for k, v in found_cols.items()}
            df_sheet.rename(columns=rename_mapping, inplace=True)
            id_vars = [col for col in required if col in df_sheet.columns]
            respondent_cols = [col for col in df_sheet.columns if isinstance(col, (int, float)) or str(col).isnumeric()]
            df_melted = df_sheet.melt(id_vars=id_vars, value_vars=respondent_cols, var_name='respondent_id', value_name='response')
            df_melted.dropna(subset=['response'], inplace=True)
            if df_melted.empty: continue
            df_melted['client_name'] = client_name
            df_melted['level_name'] = level_mapping.get(sheet_name)
            all_sheets_data.append(df_melted)
        if not all_sheets_data:
            st.error("No valid data could be processed from the worksheets.")
            return None
        new_client_df = pd.concat(all_sheets_data, ignore_index=True)
        new_client_df['theme'] = new_client_df['theme_code'].astype(str).str.strip().str.upper().map(theme_mapping_benchmark)
        response_map = {'A': 1, 'D': 0}
        new_client_df['response_val'] = new_client_df['response'].map(response_map)
        new_client_df['correct_answer_val'] = new_client_df['correct_answer'].map(response_map)
        new_client_df['score'] = np.where(new_client_df['response_val'] == new_client_df['correct_answer_val'], 1, 0)
        return new_client_df.dropna(subset=['theme', 'score'])
    except Exception as e:
        st.error(f"An error occurred while processing the Excel file: {e}")
        return None

def calculate_metrics(data):
    respondent_scores = data.groupby(['respondent_id', 'theme'])['score'].mean().reset_index()
    metrics = []
    for theme in theme_order:
        theme_data = respondent_scores[respondent_scores['theme'] == theme]
        if theme_data.empty: continue
        avg_score = theme_data['score'].mean()
        std_dev = theme_data['score'].std()
        metrics.append({'theme': theme, 'category': category_mapping.get(theme, 'Unknown'), 'avg_score': avg_score, 'std_dev': std_dev if pd.notna(std_dev) else 0, 'num_respondents': theme_data['respondent_id'].nunique()})
    metrics_df = pd.DataFrame(metrics)
    metrics_df['theme'] = pd.Categorical(metrics_df['theme'], categories=theme_order, ordered=True)
    return metrics_df.sort_values('theme').reset_index(drop=True)

def create_benchmark_plot(benchmark_metrics, client_metrics=None, title="", client_name=None):
    fig, ax = plt.subplots(figsize=(14, 9))
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    
    # Define the nodes and colors for the custom gradient based on user request
    # The node points (0.0, 0.1, etc.) correspond to the inverted score on the x-axis
    # where x=0 is a 100% score and x=1 is a 0% score.
    nodes = [0.0, 0.1, 0.2, 0.35, 0.65, 1.0]
    colors = ["darkgreen", "#59a14f", "yellow", "#fd9e4a", "#e15759", "darkred"] # Greens -> Yellow -> Orange -> Reds
    
    # Create the custom colormap by creating a list of (node, color) tuples
    color_tuples = list(zip(nodes, colors))
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_gradient', color_tuples)

    # Apply it to the plot
    ax.imshow(gradient, aspect='auto', cmap=custom_cmap, extent=[0, 1, -0.5, len(theme_order)-0.5])
    
    y_pos = range(len(benchmark_metrics))
    x_benchmark = 1 - benchmark_metrics['avg_score']
    ax.plot(x_benchmark, y_pos, marker='s', color='black', label='Benchmark', markersize=8)
    ax.errorbar(x_benchmark, y_pos, xerr=benchmark_metrics['std_dev'], fmt='none', ecolor='black', alpha=0.4, capsize=5, label='±1 SD')
    if client_metrics is not None and not client_metrics.empty:
        merged = pd.merge(benchmark_metrics[['theme']], client_metrics, on='theme', how='left')
        x_client = 1 - merged['avg_score']
        client_label = f"New Client ({client_name})"
        ax.plot(x_client, merged.index, marker='o', color='white', linestyle='solid', label=client_label, markersize=8, markeredgecolor='black')
    ax.invert_yaxis()
    ax.set_yticks(range(len(benchmark_metrics)))
    ax.set_yticklabels(benchmark_metrics['theme'], fontsize=10)
    ax.tick_params(axis='y', which='major', pad=80)
    for i, theme in enumerate(benchmark_metrics['theme']):
        ax.text(1.01, i, absentee_mapping.get(theme, ''), color='red', fontsize=9, ha='left', va='center')
        ax.text(-0.01, i, optimal_mapping.get(theme, ''), color='green', fontsize=9, ha='right', va='center')
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels([f"{int(s*100)}%" for s in np.round(1 - np.arange(0, 1.1, 0.1), 1)])
    ax.set_xlabel('Average Score', fontsize=12, labelpad=10)
    ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.7, color='white')
    ax.grid(False, which='major', axis='y')
    if not benchmark_metrics.empty:
        separator_indices = [2.5, 4.5, 8.5]
        for sep_idx in separator_indices:
             ax.axhline(y=sep_idx, color='white', linestyle='-', linewidth=2)
        category_indices = benchmark_metrics.reset_index().groupby('category')['index'].agg(['min', 'max'])
        category_positions = category_indices['min'] + (category_indices['max'] - category_indices['min']) / 2
        for category, pos in category_positions.items():
            ax.text(-0.4, pos, category.upper(), ha='center', va='center', fontweight='bold', fontsize=11, rotation=90)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    handles, labels = ax.get_legend_handles_labels()
    order = ['Benchmark', '±1 SD']
    if client_name: order.append(f"New Client ({client_name})")
    legend_elements = {label: handle for label, handle in zip(labels, handles)}
    ordered_handles = [legend_elements[label] for label in order if label in legend_elements]
    ordered_labels = [label for label in order if label in legend_elements]
    ax.legend(ordered_handles, ordered_labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=10)
    plt.subplots_adjust(left=0.30, right=0.85, top=0.9, bottom=0.15)
    return fig

def create_pareto_chart(client_metrics):
    if client_metrics.empty: return None
    pareto_data = client_metrics.sort_values(by='avg_score', ascending=False).copy()
    overall_avg_score = pareto_data['avg_score'].mean()
    pareto_data['color'] = np.where(pareto_data['avg_score'] >= overall_avg_score, '#2ca02c', '#d62728')
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(pareto_data['theme'], pareto_data['avg_score'], color=pareto_data['color'], label='Average Score (by Theme)')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.0%}', ha='center', va='bottom', fontsize=10)
    ax.axhline(y=overall_avg_score, color='black', linestyle='--', label=f'Average Score (Survey - {overall_avg_score:.0%})')
    ax.axhline(y=0.85, color='blue', linestyle='-', label='Best Practices (85%)')
    ax.set_title('Organizational Behavior Survey (OBS) - Score Pareto (by Theme)', fontsize=16, pad=20)
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    return fig

def get_demographics_from_excel(_uploaded_file_bytes):
    if not _uploaded_file_bytes:
        return {}
    try:
        file_io = io.BytesIO(_uploaded_file_bytes)
        xls = pd.ExcelFile(file_io)
        counts = {}
        for sheet_name in ['A', 'B', 'C', 'D']:
            if sheet_name in xls.sheet_names:
                df_head = pd.read_excel(xls, sheet_name=sheet_name, header=None, usecols='A:H', nrows=2)
                if df_head.shape[0] > 1 and df_head.shape[1] > 7:
                    count_val = df_head.iat[1, 7]
                    numeric_count = pd.to_numeric(count_val, errors='coerce')
                    if pd.notna(numeric_count):
                        level_name = level_mapping.get(sheet_name)
                        counts[level_name] = int(numeric_count)
        return counts
    except Exception as e:
        st.warning(f"An error occurred while reading the demographics from the Excel file. Error: {e}")
        return {}

def render_benchmark_page(df_benchmark, selected_company, selected_level, client_name_input):
    st.title("📊 Organisational Behaviour Survey (OBS) Benchmark")
    
    filtered_benchmark_df = df_benchmark.copy()
    if selected_company != 'All':
        filtered_benchmark_df = filtered_benchmark_df[filtered_benchmark_df['client_name'] == selected_company]
    if selected_level != 'All':
        filtered_benchmark_df = filtered_benchmark_df[filtered_benchmark_df['level_name'] == selected_level]
    
    if filtered_benchmark_df.empty:
        st.warning("No benchmark data available for the selected filters.")
        return

    benchmark_metrics = calculate_metrics(filtered_benchmark_df)
    client_metrics, client_name_for_plot = None, None
    
    if 'client_df' in st.session_state and st.session_state.client_df is not None:
        full_client_df = st.session_state.client_df
        filtered_client_df = full_client_df.copy()
        if selected_level != 'All':
            filtered_client_df = filtered_client_df[filtered_client_df['level_name'] == selected_level]
        if not filtered_client_df.empty:
            client_metrics = calculate_metrics(filtered_client_df)
            client_name_for_plot = st.session_state.client_name
        elif selected_level != 'All':
            st.info(f"The uploaded client data does not contain entries for the level: '{selected_level}'")
    
    title = f"Level: ({filtered_benchmark_df['respondent_id'].nunique()}) General Summary\nBenchmark: {selected_company} | Group: {selected_level}"
    if client_name_for_plot:
        title += f" | Client: {client_name_for_plot}"
    
    final_plot = create_benchmark_plot(benchmark_metrics, client_metrics, title, client_name=client_name_for_plot)
    st.pyplot(final_plot)

    if 'client_df' in st.session_state and st.session_state.client_df is not None:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🤖 AI-Powered Analysis")
            if st.button("Analyze Report with AI"):
                if client_metrics is not None and not client_metrics.empty:
                    with st.spinner("Contacting AI... This may take a moment."):
                        buf = io.BytesIO()
                        final_plot.savefig(buf, format="png")
                        buf.seek(0)
                        ai_summary = get_ai_analysis(buf, client_metrics, benchmark_metrics, client_name_for_plot, selected_level)
                        st.session_state.ai_summary = ai_summary
                else:
                    st.warning("Cannot analyze: No client data is currently displayed on the chart.")
            if "ai_summary" in st.session_state and st.session_state.get('client_name') == client_name_input :
                st.markdown(st.session_state.ai_summary)

        with col2:
            st.subheader("Append Client Data to Database")
            st.warning(f"This will add ALL processed data for **{st.session_state.client_name}** to the database.")
            if st.button(f"Confirm and Append Data for '{st.session_state.client_name}'"):
                with st.spinner("Appending data..."):
                    append_to_database(st.session_state.client_df, st.session_state.client_name)

def render_demographics_page():
    st.title("📄 Client Demographics")
    st.markdown("This page shows the breakdown of survey respondents by level for the currently uploaded client file.")

    if 'uploaded_file_bytes' not in st.session_state or st.session_state.uploaded_file_bytes is None:
        st.info("No client file has been uploaded. Please process a file on the 'Benchmark Analysis' page.")
        return

    level_counts = get_demographics_from_excel(st.session_state.uploaded_file_bytes)

    if not level_counts:
        st.warning("No demographic data could be extracted from the uploaded file. Please check that cell H2 on sheets A, B, C, and D contains a valid number.")
        return
        
    client_name = st.session_state.get('client_name', 'Current Client')
    
    df_counts = pd.DataFrame(list(level_counts.items()), columns=['Level', 'Number of Respondents'])
    total_respondents = df_counts['Number of Respondents'].sum()
    st.metric("Total Respondents in Uploaded File", f"{total_respondents}")

    fig = px.pie(
        df_counts, 
        names='Level', 
        values='Number of Respondents',
        title=f"Respondent Breakdown for: {client_name}",
        hole=0.4,
        template='plotly_white'
    )
    fig.update_traces(
        textposition='inside', 
        textinfo='percent',
        textfont_size=16,
        pull=[0.02] * len(df_counts)
    )
    fig.update_layout(
        title_x=0.5,
        title_font_size=24,
        showlegend=True,
        legend_title_text='Levels',
        legend_font_size=14,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Raw Counts")
    st.dataframe(df_counts, use_container_width=True, hide_index=True)

def render_pareto_page():
    st.title("📋 Pareto Analysis")
    st.markdown("This page shows the client's average score for each theme, sorted from highest to lowest.")
    if 'client_df' not in st.session_state or st.session_state.client_df is None:
        st.warning("Please upload a client data file on the 'Benchmark Analysis' page first.")
        return
    client_name = st.session_state.get('client_name', 'Current Client')
    st.header(f"Pareto Analysis for: {client_name}")
    client_df = st.session_state.client_df
    with st.spinner("Calculating Pareto analysis..."):
        client_metrics = calculate_metrics(client_df)
        pareto_chart = create_pareto_chart(client_metrics)
        if pareto_chart:
            st.pyplot(pareto_chart)
        else:
            st.error("Could not generate Pareto chart.")

# --- MAIN APP LAYOUT & ROUTING ---

# Initialize session state
if 'df_benchmark' not in st.session_state:
    st.session_state.df_benchmark = None

# Sidebar Layout is now conditional on data being loaded
if st.session_state.df_benchmark is not None and not st.session_state.df_benchmark.empty:
    st.sidebar.image("Renoir-Logo-1.png", use_container_width=True)
    st.sidebar.header("Filters & Controls")
    df_benchmark = st.session_state.df_benchmark

    company_list = ['All'] + sorted(df_benchmark['client_name'].unique())
    selected_company = st.sidebar.selectbox("Benchmark Company:", company_list)
    level_list = ['All'] + sorted(df_benchmark['level_name'].unique())
    selected_level = st.sidebar.selectbox("Benchmark Level:", level_list)
    
    st.sidebar.subheader("Upload New Client Data")
    client_name_input = st.sidebar.text_input("Enter New Client's Name:", st.session_state.get('client_name', ''))
    uploaded_file = st.sidebar.file_uploader("Upload Client Excel File:", type=["xlsx"])
        
    if st.sidebar.button("Process & Generate Report", type="primary"):
        if uploaded_file and client_name_input:
            with st.spinner("Processing client data..."):
                st.session_state.client_df = process_new_client_data(uploaded_file, client_name_input)
                st.session_state.client_name = client_name_input
                if 'ai_summary' in st.session_state: del st.session_state.ai_summary
        elif not client_name_input:
            st.sidebar.warning("Please enter a client name.")
        else:
            st.sidebar.warning("Please upload a client Excel file.")

    st.sidebar.markdown("---")
    page_selection = st.sidebar.radio(
        "Navigation",
        ["Benchmark Analysis", "Client Demographics", "Pareto Analysis"]
    )

    with st.sidebar.expander("Database Statistics", expanded=False):
        stats = get_database_stats()
        if stats:
            st.metric("Total Clients in DB", stats['total_clients'])
            st.metric("Total Respondents in DB", stats['total_respondents'])
            for level, count in sorted(stats['level_counts'].items()):
                st.write(f"- {level}: **{count}** respondents")
        else:
            st.write("Could not retrieve stats.")
else:
    # This block handles the initial, unloaded state
    st.sidebar.image("Renoir-Logo-1.png", use_container_width=True)
    st.sidebar.header("Filters & Controls")
    st.sidebar.info("Benchmark data not loaded. Click the button on the main page to begin.")
    selected_company, selected_level, client_name_input, page_selection = 'All', 'All', '', "Benchmark Analysis"

# Page Routing
if st.session_state.df_benchmark is None:
    st.info("Welcome to the OBS Benchmarking Platform.")
    if st.button("Click to Load Benchmark Data", type="primary"):
        with st.spinner("Loading benchmark data from the database..."):
            st.session_state.df_benchmark = load_benchmark_data()
            st.rerun() # Rerun the script to display the page
else:
    df_benchmark = st.session_state.df_benchmark
    if df_benchmark.empty:
         st.error("Application cannot start. Failed to load benchmark data from database.")
    else:
        if page_selection == "Benchmark Analysis":
            render_benchmark_page(df_benchmark, selected_company, selected_level, client_name_input)
        elif page_selection == "Client Demographics":
            render_demographics_page()
        elif page_selection == "Pareto Analysis":
            render_pareto_page()

# --- Copyright Notice ---
st.markdown("---")
st.text("""
Organisational Behaviour Survey (OBS) Benchmarking Platform
Copyright (c) August 28 2025 Keith Symondson and the Renoir Group

This software, and the data it contains, is proprietary and confidential. Unauthorized copying of this file,
via any medium, is strictly prohibited. All rights reserved. For internal use within the company only.
""")