import streamlit as st
import pandas as pd
import joblib

# Load models
salary_from_model = joblib.load('salary_from_model.pkl')
salary_to_model = joblib.load('salary_to_model.pkl')

# Load selected feature lists
selected_features_from = joblib.load('selected_features_from.pkl')
selected_features_to = joblib.load('selected_features_to.pkl')

# Load cleaned dataset
df = pd.read_csv("cleaned_nyc_jobs.csv")

st.title("NYC Job Salary Prediction")
st.write(
    "Predict the minimum and maximum annual salary based on job-related information."
)

st.header("Enter Job Details")

# Original categorical features
agency = st.selectbox(
    "Agency",
    sorted(df["Agency"].dropna().unique())
)

business_title = st.selectbox(
    "Business Title",
    sorted(df["Business Title"].dropna().unique())
)

civil_service_title = st.selectbox(
    "Civil Service Title",
    sorted(df["Civil Service Title"].dropna().unique())
)

title_classification = st.selectbox(
    "Title Classification",
    sorted(df["Title Classification"].dropna().unique())
)

title_code = st.selectbox(
    "Title Code No",
    sorted(df["Title Code No"].dropna().unique())
)

level = st.selectbox(
    "Level",
    sorted(df["Level"].dropna().unique())
)

job_category = st.selectbox(
    "Job Category",
    sorted(df["Job Category"].dropna().unique())
)

career_level = st.selectbox(
    "Career Level",
    sorted(df["Career Level"].dropna().unique())
)

work_location = st.selectbox(
    "Work Location",
    sorted(df["Work Location"].dropna().unique())
)

division = st.selectbox(
    "Division/Work Unit",
    sorted(df["Division/Work Unit"].dropna().unique())
)

education_level = st.selectbox(
    "Education Level",
    sorted(df["Education_Level"].dropna().unique())
)

experience_required = st.selectbox(
    "Experience Required",
    sorted(df["Experience_Required"].dropna().unique())
)

management_experience = st.selectbox(
    "Management Experience Required",
    sorted(df["Management_Experience_Required"].dropna().unique())
)

employment_type = st.selectbox(
    "Full-Time/Part-Time",
    sorted(df["Full-Time/Part-Time indicator"].dropna().unique())
)

posting_year = st.selectbox(
    "Posting Year",
    sorted(df["Posting_Year"].dropna().unique())
)

posting_month = st.selectbox(
    "Posting Month",
    sorted(df["Posting_Month"].dropna().unique())
)

num_positions = st.number_input(
    "Number of Positions",
    min_value=1,
    value=1
)

 # Create input data
input_data = pd.DataFrame({
    "Agency": [agency],
    "Business Title": [business_title],
    "Civil Service Title": [civil_service_title],
    "Title Classification": [title_classification],
    "Title Code No": [title_code],
    "Level": [level],
    "Job Category": [job_category],
    "Career Level": [career_level],
    "Work Location": [work_location],
    "Division/Work Unit": [division],
    "Education_Level": [education_level],
    "Experience_Required": [experience_required],
    "Management_Experience_Required": [management_experience],
    "Full-Time/Part-Time indicator": [employment_type],
    "Posting_Year": [posting_year],
    "Posting_Month": [posting_month],
    "# Of Positions": [num_positions]
}) 
   
    
input_data["Career Level_Executive"] = int(career_level == "Executive")

input_data["Career Level_Experienced (non-manager)"] = int(
    career_level == "Experienced (non-manager)"
)
input_data["Career Level_Manager"] = int(career_level == "Manager")

input_data["Career Level_Student"] = int(
    career_level == "Student"
)
input_data["Education_Level_Not Specified"] = int(
    education_level == "Not Specified"
        )
input_data["Education_Level_Masters"] = int(
    education_level == "Masters"
    )
input_data["Education_Level_Bachelors"] = int(
    education_level == "Bachelors"
    )
input_data["Education_Level_Doctorate"] = int(
    education_level == "Doctorate"
    )
input_data["Education_Level_High School"] = int(
    education_level == "High School"
    )
input_data["Experience_Required_Yes"] = int(
    experience_required == "Yes"
    )
input_data["Management_Experience_Required_Yes"] = int(
    management_experience == "Yes"
    )

input_data["Full-Time/Part-Time indicator_P"] = int(
    employment_type == "P"
    )
input_data["Full-Time/Part-Time indicator_Not Specified"] = int(
    employment_type == "Not Specified"
)
input_data["Title Classification_Pending Classification-2"] = int(
    title_classification == "Pending Classification-2"
    )

input_data["Title Classification_Labor-3"] = int(
    title_classification == "Labor-3"
    )

input_data["Title Classification_Non-Competitive-5"] = int(
    title_classification == "Non-Competitive-5"
    )

input_data["Title Classification_Exempt-4"] = int(
    title_classification == "Exempt-4"
    ) 

input_data["Posting_Month"] = posting_month
    
X_from_input = input_data[selected_features_from]
X_to_input = input_data[selected_features_to]

# Match the exact feature order used when training the models
from_model = salary_from_model.steps[-1][1]
to_model = salary_to_model.steps[-1][1]

X_from_input = X_from_input[from_model.feature_names_in_]
X_to_input = X_to_input[to_model.feature_names_in_]


if st.button("Predict Salary"):

    predicted_salary_from = salary_from_model.predict(X_from_input)[0]
    predicted_salary_to = salary_to_model.predict(X_to_input)[0]

    st.success("Salary Prediction")

    st.metric(
        "Predicted Minimum Salary",
        f"${predicted_salary_from:,.0f}"
    )

    st.metric(
        "Predicted Maximum Salary",
        f"${predicted_salary_to:,.0f}"
    )
    

