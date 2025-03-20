import streamlit as st
import pandas as pd
import pickle
import google.generativeai as genai
from sklearn.preprocessing import OneHotEncoder


# Configure Google Gemini API
genai.configure(api_key="AIzaSyC4rvQzFCG0gpnhhnYPpTjol8jcg_8aAY0")
mentor_model = genai.GenerativeModel("gemini-1.5-flash")
career_model = genai.GenerativeModel("gemini-1.5-flash")
# Load the saved model, scaler, one-hot encoder, and training columns from the 'model' folder

with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)


with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('onehot_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('training_columns.pkl', 'rb') as f:
    training_columns = pickle.load(f)

st.markdown("""
<style>
sidebar .sidebar-content {
    background-color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.header('Navigation')
page = st.sidebar.selectbox("Choose a page", ["Home", "Scholarship Finder", "Mentor Bot", "Upskilling Opportunities"])

# Session state to store student data and chat history
if 'student_data' not in st.session_state:
    st.session_state['student_data'] = {}
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def dropout_prediction_page():
    st.sidebar.header('Student Information')
    st.sidebar.markdown("Use the form below to input student data.")
    st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTVbRXce2XFjzGRQhGT_gxrZs6-8pR1FORYLw&s")
    st.sidebar.markdown("<a href='http://dropoutmanagementsystem.42web.io'>Click Here to Login!</a>", unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: left;' >🎓 Student Dropout Prediction</h1>  <a href='https://res.cloudinary.com/diunbb0ky/image/upload/v1733477220/ymvlpfaoplsr7iyz2hh3.jpg' target='_blank'>Click Here ! For Dashboard</a>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left;'>Predict the likelihood of student dropout using demographic and academic data</h3>", unsafe_allow_html=True)

    if "number" not in st.session_state:
        st.session_state["number"] = 0

    def update_number(value):
        st.session_state["number"] = value

    def user_input_features():
        st.header("Student Demographics")
        col1, col2 = st.columns(2)
        with col1:
            marital_status = st.selectbox('Marital Status', ['1 – Single', '2 – Married', '3 – Widower', '4 – Divorced', '5 – Facto Union', '6 – Legally Separated'])
            nationality = st.selectbox('Nationality', ['1 - Portuguese', '2 - German', '6 - Spanish', '11 - Italian', '13 - Dutch', '14 - English', '17 - Lithuanian', '21 - Angolan', '22 - Cape Verdean', '24 - Guinean', '25 - Mozambican', '26 - Santomean', '32 - Turkish', '41 - Brazilian', '62 - Romanian', '100 - Moldova (Republic of)', '101 - Mexican', '103 - Ukrainian', '105 - Russian', '108 - Cuban', '109 - Colombian'])
            gender = st.selectbox('Gender', ['1 – Male', '0 – Female'])
        with col2:
            age_at_enrollment = st.slider('Age at Enrollment', 17, 70, 18)
            displaced = st.selectbox('Displaced', ['1 – Yes', '0 – No'])
            international = st.selectbox('International', ['1 – Yes', '0 – No'])

        st.header("Family Background")
        col3, col4 = st.columns(2)
        with col3:
            mothers_qualification = st.selectbox('Mother\'s Qualification', ['1 - Secondary Education - 12th Year of Schooling or Eq.', '2 - Higher Education - Bachelor\'s Degree', '3 - Higher Education - Degree', '4 - Higher Education - Master\'s', '5 - Higher Education - Doctorate', '6 - Frequency of Higher Education', '9 - 12th Year of Schooling - Not Completed', '10 - 11th Year of Schooling - Not Completed', '11 - 7th Year (Old)', '12 - Other - 11th Year of Schooling', '14 - 10th Year of Schooling', '18 - General commerce course', '19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.', '22 - Technical-professional course', '26 - 7th year of schooling', '27 - 2nd cycle of the general high school course', '29 - 9th Year of Schooling - Not Completed', '30 - 8th year of schooling', '34 - Unknown', '35 - Can\'t read or write', '36 - Can read without having a 4th year of schooling', '37 - Basic education 1st cycle (4th/5th year) or equiv.', '38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.', '39 - Technological specialization course', '40 - Higher education - degree (1st cycle)', '41 - Specialized higher studies course', '42 - Professional higher technical course', '43 - Higher Education - Master (2nd cycle)', '44 - Higher Education - Doctorate (3rd cycle)'])
            fathers_qualification = st.selectbox('Father\'s Qualification', ['1 - Secondary Education - 12th Year of Schooling or Eq.', '2 - Higher Education - Bachelor\'s Degree', '3 - Higher Education - Degree', '4 - Higher Education - Master\'s', '5 - Higher Education - Doctorate', '6 - Frequency of Higher Education', '9 - 12th Year of Schooling - Not Completed', '10 - 11th Year of Schooling - Not Completed', '11 - 7th Year (Old)', '12 - Other - 11th Year of Schooling', '13 - 2nd year complementary high school course', '14 - 10th Year of Schooling', '18 - General commerce course', '19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.', '20 - Complementary High School Course', '22 - Technical-professional course', '25 - Complementary High School Course - not concluded', '26 - 7th year of schooling', '27 - 2nd cycle of the general high school course', '29 - 9th Year of Schooling - Not Completed', '30 - 8th year of schooling', '31 - General Course of Administration and Commerce', '33 - Supplementary Accounting and Administration', '34 - Unknown', '35 - Can\'t read or write', '36 - Can read without having a 4th year of schooling', '37 - Basic education 1st cycle (4th/5th year) or equiv.', '38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.', '39 - Technological specialization course', '40 - Higher education - degree (1st cycle)', '41 - Specialized higher studies course', '42 - Professional higher technical course', '43 - Higher Education - Master (2nd cycle)', '44 - Higher Education - Doctorate (3rd cycle)'])
        with col4:
            mothers_occupation = st.selectbox('Mother\'s Occupation', ['0 - Student', '1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers', '2 - Specialists in Intellectual and Scientific Activities', '3 - Intermediate Level Technicians and Professions', '4 - Administrative staff', '5 - Personal Services, Security and Safety Workers and Sellers', '6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry', '7 - Skilled Workers in Industry, Construction and Craftsmen', '8 - Installation and Machine Operators and Assembly Workers', '9 - Unskilled Workers', '10 - Armed Forces Professions', '90 - Other Situation', '99 - (blank)', '122 - Health professionals', '123 - Teachers', '125 - Specialists in information and communication technologies (ICT)', '131 - Intermediate level science and engineering technicians and professions', '132 - Technicians and professionals, of intermediate level of health', '134 - Intermediate level technicians from legal, social, sports, cultural and similar services', '141 - Office workers, secretaries in general and data processing operators', '143 - Data, accounting, statistical, financial services and registry-related operators', '144 - Other administrative support staff', '151 - Personal service workers', '152 - Sellers', '153 - Personal care workers and the like', '171 - Skilled construction workers and the like, except electricians', '173 - Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like', '175 - Workers in food processing, woodworking, clothing and other industries and crafts', '191 - Cleaning workers', '192 - Unskilled workers in agriculture, animal production, fisheries and forestry', '193 - Unskilled workers in extractive industry, construction, manufacturing and transport', '194 - Meal preparation assistants'])
            fathers_occupation = st.selectbox('Father\'s Occupation', ['0 - Student', '1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers', '2 - Specialists in Intellectual and Scientific Activities', '3 - Intermediate Level Technicians and Professions', '4 - Administrative staff', '5 - Personal Services, Security and Safety Workers and Sellers', '6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry', '7 - Skilled Workers in Industry, Construction and Craftsmen', '8 - Installation and Machine Operators and Assembly Workers', '9 - Unskilled Workers', '10 - Armed Forces Professions', '90 - Other Situation', '99 - (blank)', '101 - Armed Forces Officers', '102 - Armed Forces Sergeants', '103 - Other Armed Forces personnel', '112 - Directors of administrative and commercial services', '114 - Hotel, catering, trade and other services directors', '121 - Specialists in the physical sciences, mathematics, engineering and related techniques', '122 - Health professionals', '123 - Teachers', '124 - Specialists in finance, accounting, administrative organization, public and commercial relations', '131 - Intermediate level science and engineering technicians and professions', '132 - Technicians and professionals, of intermediate level of health', '134 - Intermediate level technicians from legal, social, sports, cultural and similar services', '135 - Information and communication technology technicians', '141 - Office workers, secretaries in general and data processing operators', '143 - Data, accounting, statistical, financial services and registry-related operators', '144 - Other administrative support staff', '151 - Personal service workers', '152 - Sellers', '153 - Personal care workers and the like', '154 - Protection and security services personnel', '161 - Market-oriented farmers and skilled agricultural and animal production workers', '163 - Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence', '171 - Skilled construction workers and the like, except electricians', '172 - Skilled workers in metallurgy, metalworking and similar', '174 - Skilled workers in electricity and electronics', '175 - Workers in food processing, woodworking, clothing and other industries and crafts', '181 - Fixed plant and machine operators', '182 - Assembly workers', '183 - Vehicle drivers and mobile equipment operators', '192 - Unskilled workers in agriculture, animal production, fisheries and forestry', '193 - Unskilled workers in extractive industry, construction, manufacturing and transport', '194 - Meal preparation assistants', '195 - Street vendors (except food) and street service providers'])

        st.header("Academic Background")
        col5, col6 = st.columns(2)
        with col5:
            previous_qualification = st.selectbox('Previous Qualification', ['1 - Secondary education', '2 - Higher education - bachelor\'s degree', '3 - Higher education - degree', '4 - Higher education - master\'s', '5 - Higher education - doctorate', '6 - Frequency of higher education', '9 - 12th year of schooling - not completed', '10 - 11th year of schooling - not completed', '12 - Other - 11th year of schooling', '14 - 10th Year of Schooling', '15 - 10th year of schooling - not completed', '19 - Basic education 3rd cycle (9th/10th/11th year) or equiv.', '38 - Basic education 2nd cycle (6th/7th/8th year) or equiv.', '39 - Technological specialization course', '40 - Higher education - degree (1st cycle)', '42 - Professional higher technical course', '43 - Higher education - master (2nd cycle)'])
            previous_qualification_grade = st.slider('Previous Qualification Grade', 0.0, 200.0, 150.0)
            admission_grade = st.slider('Admission Grade', 0.0, 200.0, 150.0)
        with col6:
            application_mode = st.selectbox('Application Mode', ['1 - 1st phase - general contingent', '2 - Ordinance No. 612/93', '5 - 1st phase - special contingent (Azores Island)', '7 - Holders of other higher courses', '10 - Ordinance No. 854-B/99', '15 - International student (bachelor)', '16 - 1st phase - special contingent (Madeira Island)', '17 - 2nd phase - general contingent', '18 - 3rd phase - general contingent', '26 - Ordinance No. 533-A/99, item b2) (Different Plan)', '27 - Ordinance No. 533-A/99, item b3 (Other Institution)', '39 - Over 23 years old', '42 - Transfer', '43 - Change of course', '44 - Technological specialization diploma holders', '51 - Change of institution/course', '53 - Short cycle diploma holders', '57 - Change of institution/course (International)'])
            application_order = st.slider('Application Order', 0, 9, 0)
            course = st.selectbox('Course', [
                '33 - Biofuel Production Technologies',
                '171 - Animation and Multimedia Design',
                '8014 - Social Service (evening attendance)',
                '9003 - Agronomy',
                '9070 - Communication Design',
                '9085 - Veterinary Nursing',
                '9119 - Informatics Engineering',
                '9130 - Equinculture',
                '9147 - Management',
                '9238 - Social Service',
                '9254 - Tourism',
                '9500 - Nursing',
                '9556 - Oral Hygiene',
                '9670 - Advertising and Marketing Management',
                '9773 - Journalism and Communication',
                '9853 - Basic Education',
                '9991 - Management (evening attendance)',
                '1000 - 10th',
                '1001 - Intermediate(MPC)',
                '1002 - Intermediate(BIPC)',
                '1003 - Intermediate(MEC)',
                '1004 - Intermediate(CEC)',
                '2001 - BTECH(CSE)',
                '2002 - BTECH(MECH)',
                '2003 - BTECH(EEE)',
                '2004 - BTECH(ECE)',
                '2005 - BTECH(CIVL)'
            ])

        st.header("Current Academic Performance")
        col7, col8 = st.columns(2)
        with col7:
            daytime_evening_attendance = st.selectbox('Daytime/Evening Attendance', ['1 – Daytime', '0 - Evening'])
            curricular_units_1st_sem_credited = st.slider('Curricular Units 1st Sem (Credited)', 0, 60, 30)
            curricular_units_1st_sem_enrolled = st.slider('Curricular Units 1st Sem (Enrolled)', 0, 60, 30)
            st.number_input("Enter attendance percentage(in %):", min_value=0, max_value=100, value=st.session_state["number"], key="number_input", on_change=lambda: update_number(st.session_state["number_input"]))
        with col8:
            curricular_units_1st_sem_evaluations = st.slider('Curricular Units 1st Sem (Evaluations)', 0, 60, 30)
            curricular_units_1st_sem_approved = st.slider('Curricular Units 1st Sem (Approved)', 0, 60, 30)

        st.header("Additional Information")
        col9, col10 = st.columns(2)
        with col9:
            educational_special_needs = st.selectbox('Educational Special Needs', ['1 – Yes', '0 – No'])
            debtor = st.selectbox('Debtor', ['1 – Yes', '0 – No'])
        with col10:
            tuition_fees_up_to_date = st.selectbox('Tuition Fees Up to Date', ['1 – Yes', '0 – No'])
            scholarship_holder = st.selectbox('Scholarship Holder', ['1 – Yes', '0 – No'])

        # Handle both separators for Daytime/Evening Attendance
        if ' – ' in daytime_evening_attendance:
            daytime_value = daytime_evening_attendance.split(' – ')[0]
        else:
            daytime_value = daytime_evening_attendance.split(' - ')[0]

        data = {
            'Marital_status': int(marital_status.split(' – ')[0]),
            'Application_mode': int(application_mode.split(' - ')[0]),
            'Application_order': application_order,
            'Course': int(course.split(' - ')[0]),
            'Daytime_evening_attendance': int(daytime_value),
            'Previous_qualification': int(previous_qualification.split(' - ')[0]),
            'Previous_qualification_grade': previous_qualification_grade,
            'Nacionality': int(nationality.split(' - ')[0]),
            'Mothers_qualification': int(mothers_qualification.split(' - ')[0]),
            'Fathers_qualification': int(fathers_qualification.split(' - ')[0]),
            'Mothers_occupation': int(mothers_occupation.split(' - ')[0]),
            'Fathers_occupation': int(fathers_occupation.split(' - ')[0]),
            'Admission_grade': admission_grade,
            'Displaced': int(displaced.split(' – ')[0]),
            'Educational_special_needs': int(educational_special_needs.split(' – ')[0]),
            'Debtor': int(debtor.split(' – ')[0]),
            'Tuition_fees_up_to_date': int(tuition_fees_up_to_date.split(' – ')[0]),
            'Gender': int(gender.split(' – ')[0]),
            'Scholarship_holder': int(scholarship_holder.split(' – ')[0]),
            'Age_at_enrollment': age_at_enrollment,
            'International': int(international.split(' – ')[0]),
            'Curricular_units_1st_sem_credited': curricular_units_1st_sem_credited,
            'Curricular_units_1st_sem_enrolled': curricular_units_1st_sem_enrolled,
            'Curricular_units_1st_sem_evaluations': curricular_units_1st_sem_evaluations,
            'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved
        }

        features = pd.DataFrame(data, index=[0])
        return features, course

    input_df, selected_course = user_input_features()

    if st.button("Predict Dropout"):
        # One-Hot Encoding for categorical features
        categorical_cols = ['Application_mode', 'Course', 'Marital_status', 'Nacionality', 'Mothers_qualification', 'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation']
        input_encoded = encoder.transform(input_df[categorical_cols])
        input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_cols))

        # Preserve non-categorical, non-numerical columns
        numerical_cols = ['Previous_qualification_grade', 'Admission_grade', 'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved', 'Age_at_enrollment']
        non_processed_cols = [col for col in input_df.columns if col not in categorical_cols + numerical_cols]
        non_processed_df = input_df[non_processed_cols]

        # Drop original categorical columns and concatenate
        input_df = input_df.drop(columns=categorical_cols)
        input_df = pd.concat([input_df[numerical_cols], input_encoded_df, non_processed_df], axis=1)

        # Scale the numerical features
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Align input_df with training_columns
        missing_cols = [col for col in training_columns if col not in input_df.columns]
        extra_cols = [col for col in input_df.columns if col not in training_columns]

        if missing_cols:
            for col in missing_cols:
                input_df[col] = 0
            st.warning(f"Added missing columns with default value 0: {missing_cols}")

        if extra_cols:
            input_df = input_df.drop(columns=extra_cols)
            st.warning(f"Dropped extra columns: {extra_cols}")

        # Reorder columns to match training_columns exactly
        input_df = input_df[training_columns]

        # Predict
        prediction = model.predict(input_df)
        st.subheader('Prediction')
        status_map = {0: 'Dropout', 1: 'Not Dropout'}
        try:
            dropout_status = status_map[prediction[0]]
        except KeyError:
            st.error(f"Unexpected prediction value: {prediction[0]}. Expected 0 or 1.")
            dropout_status = "Unknown"
        st.write(dropout_status)

        debtor_status = 'Yes' if input_df['Debtor'][0] == 1 else 'No'
        low_credits_threshold = 9
        credited_units = input_df['Curricular_units_1st_sem_credited'][0]
        is_low_credits = credited_units < low_credits_threshold

        # Determine academic performance based on credits approved
        approved_units = input_df['Curricular_units_1st_sem_approved'][0]
        if approved_units < 10:
            academic_performance = "low"
        elif approved_units < 20:
            academic_performance = "medium"
        else:
            academic_performance = "high"

        # Map the selected course to study field
        study_field = selected_course.split(' - ')[1] if ' - ' in selected_course else "Unknown"

        # Store student data in session state
        st.session_state['student_data'] = {
            'study_field': study_field,
            'dropout_status': dropout_status,
            'academic_performance': academic_performance,
            'has_low_credits': is_low_credits
        }

        if dropout_status == 'Dropout':
            st.write("The student is predicted to Dropout.")
            if debtor_status == 'Yes':
                st.warning("The student is in debt. Consider providing financial aid to support retention.")
                st.markdown("""<a href="https://akshayram1-scholership-finder.hf.space" target="_blank">Click here to explore the Scholarship Advisor</a>""", unsafe_allow_html=True)
            if is_low_credits:
                st.warning("The student has low academic credits. Navigate to 'Upskilling Opportunities' in the sidebar for tailored recommendations.")
        else:
            if st.session_state['number'] < 35:
                st.write("The student is predicted to be Not Dropout but has low attendance. Contact their parents.")
            else:
                st.write("The student is predicted to be Not Dropout.")
            if debtor_status == 'Yes':
                st.warning("Although the student is not predicted to dropout, consider addressing the debt issue.")
                st.markdown("""<a href="https://akshayram1-scholership-finder.hf.space" target="_blank">Click here to explore the Scholarship Advisor</a>""", unsafe_allow_html=True)
            if is_low_credits:
                st.warning("The student has low academic credits. Navigate to 'Upskilling Opportunities' in the sidebar for tailored recommendations.")

        if not is_low_credits and debtor_status == 'No':
            st.info("The student shows no signs of financial or academic issues.")

def scholarship_finder_page():
    st.title("Scholarship Finder")
    st.write("Click the link below to explore scholarship opportunities:")
    st.markdown("""<a href="https://akshayram1-scholership-finder.hf.space" target="_blank">Go to Scholarship Finder</a>""", unsafe_allow_html=True)

def mentor_bot_page():
    st.title("Student Motivation Bot")
    st.write("Welcome to the Student Motivation Bot! Share your frustrations, reasons for dropping out, or any challenges you're facing. I'm here to listen and help you find your path.")

    # Display chat history
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input for user
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Add user message to chat history
        st.session_state['chat_history'].append({"role": "user", "content": user_input})

        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_input)

        # Generate bot response
        system_content = (
            "You are a compassionate and understanding mentor, specifically designed to help students who are struggling, frustrated, or considering dropping out. "
            "Your primary goal is to listen, empathize, and motivate. "
            "Ask open-ended questions to understand the student's reasons for their struggles. "
            "Offer words of encouragement, share stories of resilience, and help them identify their strengths. "
            "Help them rediscover their passion and find alternative paths to success. "
            "Focus on building their confidence and reminding them that setbacks are a normal part of life. "
            "If the query is vague, ask clarifying questions to get to the root of the problem. "
            "Avoid judgement and focus on support."
        )
        # Include the full chat history in the prompt to maintain context
        chat_context = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state['chat_history']])
        prompt = f"System: {system_content}\n\n{chat_context}"

        try:
            response = mentor_model.generate_content(prompt)
            bot_response = response.text.strip() if hasattr(response, "text") else "Sorry, I couldn't generate a response."
        except Exception as e:
            bot_response = f"An error occurred: {e}"

        # Add bot response to chat history
        st.session_state['chat_history'].append({"role": "assistant", "content": bot_response})

        # Display bot response
        with st.chat_message("assistant"):
            st.write(bot_response)

def upskilling_opportunities_page():
    st.title("Upskilling Opportunities")
    st.write("Explore tailored upskilling opportunities based on your study field, dropout status, and academic performance.")

    if st.session_state['student_data']:
        study_field = st.session_state['student_data'].get('study_field', '')
        dropout_status = st.session_state['student_data'].get('dropout_status', 'No')
        academic_performance = st.session_state['student_data'].get('academic_performance', 'Medium')
        has_low_credits = st.session_state['student_data'].get('has_low_credits', False)
        
        if study_field:
            st.write("To generate upskilling opportunities, we use your study field, dropout status, and academic performance. Below are recommendations based on your recent prediction data:")
            st.write(f"- Study Field: {study_field}")
            st.write(f"- Dropout Status: {dropout_status}")
            st.write(f"- Academic Performance: {academic_performance.capitalize()}")

            system_content = (
                "You are a career guidance expert specializing in identifying upskilling opportunities for students. "
                "Based on the student's current course (study field), dropout status, and academic performance, provide specific and actionable upskilling opportunities. "
                "Focus on practical options such as online courses, certifications, or skills to develop that align with their field of study or improve their employability. "
                "If the student is a dropout or has career gaps, suggest alternative paths and skills to bridge gaps. "
                "Tailor recommendations to their academic performance: low performance should emphasize foundational skills, medium should balance practical and advanced skills, and high should focus on advanced certifications or specializations. "
                "Provide specific examples (e.g., course names, platforms like Coursera/Udemy, or skills like 'Python programming'). "
                "Focus only on upskilling opportunities. Do not provide motivational messages or job roles."
            )
            prompt = f"System: {system_content}\n\nStudy Field: {study_field}\nDropout Status: {dropout_status}\nAcademic Performance: {academic_performance}"
            try:
                response = career_model.generate_content(prompt)
                opportunities = response.text.strip() if hasattr(response, "text") else "Sorry, I couldn't generate upskilling opportunities."
                st.subheader("Upskilling Opportunities:")
                st.write(opportunities)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("No study field data available from the prediction. Please run a prediction on the 'Home' page first.")
    else:
        st.write("To get tailored upskilling opportunities, we need the following information:")
        st.write("- Your current or last area of study (e.g., Biofuel Production Technologies)")
        st.write("- Whether you are a dropout or have career gaps (Yes/No)")
        st.write("- Your academic performance (Low/Medium/High)")
        st.write("Please run a dropout prediction on the 'Home' page to provide this information automatically.")

if page == "Home":
    dropout_prediction_page()
elif page == "Scholarship Finder":
    scholarship_finder_page()
elif page == "Mentor Bot":
    mentor_bot_page()
elif page == "Upskilling Opportunities":
    upskilling_opportunities_page()

           
