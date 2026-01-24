import asyncio
import pandas as pd
import plotly.express as px
import torch
import streamlit as st
from datetime import date
from datetime import datetime, timedelta
from pymongo import MongoClient
from threading import Thread
from transformers import pipeline, TextIteratorStreamer




#MongoDB access
db_access = st.secrets.mongo_db_key


# -------------------------------
# DATABASE SETUP
# -------------------------------
client = MongoClient(db_access)  
db = client["Sierra-Nevada"]

signups = db["SignUps"]
period_symptoms = db["PeriodSymptoms"]


@st.cache_resource
def load_pipeline():
    # Adding torch_dtype="auto" or "float16" speeds up GPU inference
    return pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float16)
pipe = load_pipeline()




# -------------------------------
# SESSION STATE INIT
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "need_to_enter_symptoms" not in st.session_state:
    st.session_state.need_to_enter_symptoms = False

# -------------------------------
# CONFIG
# -------------------------------
DEFAULT_SYMPTOMS = [
    "Cramps", "Fatigue", "Bloating", "Mood swings",
    "Back pain", "Migraines", "Sleep problems",
    "Food cravings", "Heavy bleeding"
]

# -------------------------------
# DATABASE HELPERS
# -------------------------------
def get_user_doc(username):
    return period_symptoms.find_one({"ID": username}) or {}

def save_symptom_library(username, default_symptoms, custom_symptoms):
    period_symptoms.update_one(
        {"ID": username},
        {
            "$set": {
                "symptom_library.default": default_symptoms,
                "symptom_library.custom": custom_symptoms
            }
        },
        upsert=True
    )

def get_all_symptoms(username):
    doc = get_user_doc(username)
    lib = doc.get("symptom_library", {})
    return sorted(set(lib.get("default", []) + lib.get("custom", [])))

def save_daily_log(username, symptom_scores):
    today = date.today().isoformat()
    period_symptoms.update_one(
        {"ID": username},
        {"$set": {f"logs.{today}.symptoms": symptom_scores}},
        upsert=True
    )

def load_logs_df(username):
    doc = get_user_doc(username)
    logs = doc.get("logs", {})

    rows = []
    for log_date, entry in logs.items():
        for symptom, score in entry["symptoms"].items():
            rows.append({
                "date": log_date,
                "symptom": symptom,
                "score": score
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df

# -------------------------------
# UTILITIES
# -------------------------------
def severity_label(score):
    if score <= 3:
        return "🟢 Mild"
    elif score <= 6:
        return "🟡 Moderate"
    return "🔴 Severe"

# -------------------------------
# AUTH UI
# -------------------------------
def login_signup_page():
    st.markdown(
          """
          <h1 style="
          color: #007D8C; 
          text-align: center; 
          font-family: 'Helvetica Neue', sans-serif; 
          letter-spacing: 1px;
          font-weight: 700;
          margin-bottom: 0px;
          ">
           ZOLDAA
          </h1>
          <p style="
          text-align: center; 
          color: #555555; 
          font-size: 14px; 
          margin-top: -10px;
          margin-bottom: 25px;
          ">
          Stand Tall & Thrive ✨
         </p>
         """, 
         unsafe_allow_html=True
     )

    st.divider() 
    st.image("dope_cover.png")
    login, signup = st.tabs(["Login", "Sign Up"])
    

    with login:
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")

        if st.button("Login"):
            if signups.find_one({"Name": user, "Password": pw}):
                st.session_state.logged_in = True
                st.session_state.username = user
                st.session_state.need_to_enter_symptoms = False
                st.rerun()
            else:
                st.error("Invalid credentials")

    with signup:
        user = st.text_input("New Username")
        pw = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            if signups.find_one({"Name": user}):
                st.error("Username exists")
            else:
                signups.insert_one({"Name": user, "Password": pw})
                st.session_state.logged_in = True
                st.session_state.username = user
                st.session_state.need_to_enter_symptoms = True
                st.success("Account created")
                st.rerun()


def energy_levels(score):
    if score <= 3:
        # Using a span with font-size to enlarge the emoji
        st.markdown('<p style="text-align: center; font-size: 50px;">🪫</p>', unsafe_allow_html=True)
    elif score <= 6:
        st.markdown('<p style="text-align: center; font-size: 30px;">🟡 Moderate</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="text-align: center; font-size: 50px;">🔋</p>', unsafe_allow_html=True)


def fatigue():
    fatigue_emoji_placeholder = st.empty()
    st.markdown("Your Energy Levels At The Moment?",text_alignment="center")
    energy = st.select_slider(label="Energy",
                              options=[1,2,3,4,5,6,7,8,9,10],
                              value= 10)
    
    # Update the placeholder at the top of the page with the current selection
    fatigue_emoji_placeholder.markdown(f"## {energy}", text_alignment="center")
    energy_levels(energy)

    

def mood():
    # Create a placeholder for the emoji
    emoji_placeholder = st.empty()
    st.markdown("What's Your Mood like now?", text_alignment= "center")
    st.write(" ")
    mood = st.select_slider(
                label="Mood",
                options=["😡","😠","😖","😞","☹️","😕","😓","😔","😒","😑","😐","🙄"],
                value="😒"
                )

    # Update the placeholder at the top of the page with the current selection
    emoji_placeholder.markdown(f"## {mood}", text_alignment="center")


def migraine():
    st.markdown('<p style = "text-align: center; font-size: 30px;">🤦🏾‍♀️</p>',unsafe_allow_html=True)
    st.markdown("Head Pain Level Right Now?", text_alignment="center")
    migraine = st.select_slider(
        label="Migraine",
        options=[1,2,3,4,5,6,7,8,9,10],
        value=10
    )


def cramps():
    st.markdown('<p style = "text-align: center; font-size: 30px;">⚡</p>',unsafe_allow_html=True)
    st.markdown("How's the cramping?", text_alignment="center")
    # Return value of this widget
    cramps_val = st.select_slider(
        label="Cramps",
        options=[1,2,3,4,5,6,7,8,9,10],
        value=10,
        key = "cramps_slider"
    )
    return cramps_val


def bloating():
    st.markdown('<p style = "text-align: center; font-size: 30px;">🤢</p>',unsafe_allow_html=True)
    st.markdown("How's the bloating right now?", text_alignment="center")
    bloating = st.select_slider(
        label="Bloated",
        options=[1,2,3,4,5,6,7,8,9,10],
        value=10
    )


def sleep():
    st.markdown('<p style = "text-align: center; font-size: 30px;">💤</p>',unsafe_allow_html=True)
    st.markdown("How was your rest last night?", text_alignment="center")
    sleep = st.select_slider(
        label="Sleep",
        options=[1,2,3,4,5,6,7,8,9,10],
        value=10
    )


def food_cravings():
    st.markdown('<p style = "text-align: center; font-size: 30px;">🍔</p>',unsafe_allow_html=True)
    st.markdown("Rate your major snack cravings at the moment", text_alignment="center")
    food_cravings = st.select_slider(
        label="Food Cravings",
        options=[1,2,3,4,5,6,7,8,9,10],
        value=10
    )


def back_pain():
    st.markdown('<p style = "text-align: center; font-size: 30px;">🙇🏼‍♀️</p>',unsafe_allow_html=True)
    st.markdown("Back pain level at the moment?", text_alignment="center")
    back_pain = st.select_slider(
        label="Back pain",
        options=[1,2,3,4,5,6,7,8,9,10],
        value=10
    )


def heavy_bleeding():
    with st.container():
        st.markdown('<p style = "text-align: center; font-size: 30px;">🩸</p>',unsafe_allow_html=True)
        st.markdown("How heavy is your bleeding now?", text_alignment="center")
        heavy_bleeding = st.select_slider(
               label="Bleeds",
               options=[1,2,3,4,5,6,7,8,9,10],
               value=10
            )


# -------------------------------
# SYMPTOM SETUP
# -------------------------------
def pick_symptoms():
    st.subheader("Select your symptoms")

    selected_defaults = []
    cols = st.columns(3)

    for i, symptom in enumerate(DEFAULT_SYMPTOMS):
        with cols[i % 3]:
            if st.checkbox(symptom):
                selected_defaults.append(symptom)

    st.divider()
    custom_symptom = st.text_input("Add a custom symptom")

    if st.button("Save Symptoms"):
        custom_list = []
        if custom_symptom.strip():
            custom_list.append(custom_symptom.strip().title())

        save_symptom_library(
            st.session_state.username,
            selected_defaults,
            custom_list
        )

        st.session_state.need_to_enter_symptoms = False
        st.success("Symptoms saved")
        st.rerun()

# -------------------------------
# TABS
# -------------------------------
def today_tab():
    symptoms = get_all_symptoms(st.session_state.username)

    if not symptoms:
        st.warning("No symptoms configured")
        pick_symptoms()
        return

    st.subheader("Journal Symptoms",text_alignment="center")

    scores = {}
    for symptom in symptoms:
        if symptom == "Fatigue":
            fatigue()
        elif symptom == "Bloating":
            bloating()
        elif symptom == "Sleep problems":
            sleep()
        elif symptom == "Back pain":
            back_pain()
        elif symptom == "Food cravings":
            food_cravings()
        elif symptom == "Cramps":
            scores[symptom] = cramps()
        elif symptom == "Heavy bleeding":
            heavy_bleeding()
        elif symptom == "Mood swings":
            mood()
        elif symptom == "Migraines":
            migraine()
        else:
            score = st.slider(symptom, 0, 10, 0)
            scores[symptom] = score
            st.caption(severity_label(score))

    if st.button("Save Today"):
        #save_daily_log(st.session_state.username, scores)
        st.success("Logged successfully")



def metrics_tab():
    df = load_logs_df(st.session_state.username)

    if df.empty:
        st.info("No data yet")
        return

    symptom = st.selectbox("Select symptom", df["symptom"].unique())
    filtered = df[df["symptom"] == symptom]
    #with st.expander("Symptoms at a glance", expanded=True):
        #st.line_chart(filtered.set_index("date")["score"])
    st.bar_chart(df.groupby("symptom")["score"].mean())

def insights_tab():
    df = load_logs_df(st.session_state.username)

    if df.empty:
        st.info("No insights yet")
        return

    st.subheader("Insights")

    avg_scores = df.groupby("symptom")["score"].mean()

    for symptom, avg in avg_scores.items():
        if avg >= 7:
            st.error(f"{symptom} has been consistently severe")
        elif avg >= 4:
            st.warning(f"{symptom} shows moderate recurring intensity")
        else:
            st.success(f"{symptom} remains mild")


def filter_logs_by_date(logs, date_range):
    """Filter logs based on the selected date range."""
    today = datetime.today()
    if date_range == "This Week":
        start_date = today - timedelta(days=7)
    elif date_range == "Last 6 Months":
        start_date = today - timedelta(weeks=26)
    elif date_range == "Last 1 Year":
        start_date = today - timedelta(days=365)
    else:
        start_date = today  # Default to today if no range is selected

    # Filter the logs based on the date range
    filtered_data = []
    for log_date, symptoms in logs.items():
        log_date_obj = datetime.strptime(log_date, "%Y-%m-%d")
        if log_date_obj >= start_date:
            for symptom, rating in symptoms['symptoms'].items():
                filtered_data.append({
                    'Date': log_date,
                    'Symptom': symptom,
                    'Severity': rating
                })
    return filtered_data

def symptoms_insights():
    """Function to display symptom logs and generate insights."""
    # Access the user data
    user_data = period_symptoms.find_one({"ID": st.session_state.username})

    # Ensure data exists
    if user_data and 'logs' in user_data:
        logs = user_data['logs']
        
        # Date range filter
        date_range = st.selectbox("Select Date Range", ["This Week", "Last 6 Months", "Last 1 Year"])
        
        # Filter logs based on the selected date range
        filtered_data = filter_logs_by_date(logs, date_range)

        # Convert to DataFrame for easier manipulation
        if filtered_data:
            df = pd.DataFrame(filtered_data)

            # Check if there is data to plot
            if not df.empty:
                st.subheader(f"Symptom Ratings In {date_range}")
                
                # Aggregate symptoms by date
                symptoms_trends = df.groupby(['Symptom', 'Date']).agg({'Severity': 'mean'}).reset_index()

                # Create an interactive plot with Plotly
                fig = px.line(symptoms_trends, x="Date", y="Severity", color="Symptom",
                              #title="Symptom Severity Over Time",
                              labels={"Severity": "Severity (0-10)", "Date": "Date"},
                              markers=True)

                # Update the layout for better readability
                fig.update_layout(xaxis_title="Date", yaxis_title="Severity (0-10)",
                                  legend_title="Symptoms", template="plotly_dark")
                st.plotly_chart(fig)

            else:
                st.warning("No symptom data available for visualization.")
        else:
            st.warning("No logs found for the selected date range.")
    else:
        st.warning("No logs available for this user.")


    
# -------------------------------
# MAIN APP
# -------------------------------
def landing_page():
    st.markdown(
          """
          <h1 style="
          color: #007D8C; 
          text-align: center; 
          font-family: 'Helvetica Neue', sans-serif; 
          letter-spacing: 1px;
          font-weight: 700;
          margin-bottom: 0px;
          ">
           ZOLDAA
          </h1>
          <p style="
          text-align: center; 
          color: #555555; 
          font-size: 14px; 
          margin-top: -10px;
          margin-bottom: 25px;
          ">
          Stand Tall & Thrive ✨
         </p>
         """, 
         unsafe_allow_html=True
     )

    st.divider()  
    app = st.sidebar.selectbox("Menu",["📝 Journals","🧭 Metrics", "🧠Insights","✨ Ask Kyma", "🌏 About","❌ Log Out"])

    if app == "📝 Journals":
        today_tab()
    elif app == "🧭 Metrics":
        metrics_tab()
    elif app == "🧠Insights":
        symptoms_insights()
        insights_tab()
    elif app == "✨ Ask Kyma":
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


        # ... (History initialization and display code) ...

        if prompt := st.chat_input("✨ Ask Kyma",accept_file = True,):
            user_input = prompt.text
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
                # Optional: Handle uploaded files if any
                if prompt.files:
                    st.caption(f"📎 {len(prompt.files)} file(s) uploaded")


            with st.chat_message("assistant",avatar = "👱🏽‍♀️"):
                # Setup for streaming
                streamer = TextIteratorStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
                # Prepare arguments
                messages = st.session_state.messages # Use full history for context
                generation_kwargs = dict(
                    text_inputs=messages, 
                    streamer=streamer,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )

                # Run generation in a background thread to prevent UI blocking
                thread = Thread(target=pipe, kwargs=generation_kwargs)
                thread.start()

                # Display the stream
                full_response = st.write_stream(streamer)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

    elif app == "🌏 About":
        st.image("group_of_women.png")
        st.caption ("_❤️AI-Driven Healthcare, Accessible and Affordable for Every lady_",text_alignment="center")
        st.write("Every lady deserves access to quality healthcare, no matter where she lives. We're on a mission to make affordable, world-class care available to women & girls across Africa and beyond. It's time to prioritize women's health, globally.")
        st.image("iron_lady.png")
        st.caption ("_...built with 💜 4rm ZOLDAA_",text_alignment = "center")
        st.write("Together, we can make women's health a priority, and ensure that every woman is empowered with the knowledge and resources she needs to thrive")

        st.markdown("""
                         ---
                         📩 We would love to hear your thoughts (gilcare.com@gmail.com)
                                          """)
    elif app == "❌ Log Out":
        st.session_state.logged_in = False
        st.rerun()        

if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

def main():
    if not st.session_state.logged_in:
        login_signup_page()
    elif st.session_state.need_to_enter_symptoms:
        pick_symptoms()
    else:
        landing_page()

if __name__ == "__main__":
    main()


