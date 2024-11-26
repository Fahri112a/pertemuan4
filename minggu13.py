import streamlit as st
#nomor1
#st.write("Hello world")

#nomor2
#st.header('st.button')

#if st.header('st.button'):
    #st.write("Why hello there")
#else:
    #st.write('Goodbye')

#nomor3
#st.title("this is the app title")

#st.markdown("### this is the markdown")

#st.header("this is the header")

#st.subheader("this is the subheader")

#st.caption("this is the caption")

#x = 2021
#st.code(f"x = {x}", language='python')

#momor4
#if st.checkbox("yes"):
    #st.write("Checkbox selected!")

#if st.button("Click"):
    #st.write("Button clicked!")

#gender = st.radio("Pick your gender", ("Male", "Female"))
#st.write(f"You selected: {gender}")

#selected_gender = st.selectbox("Pick your gender", ["Male", "Female"])
#st.write(f"You selected: {selected_gender}")

#planet = st.selectbox("choose a planet", ["Mercury", "Venus", "Earth", "Mars", "Jupiter"])
#st.write(f"You selected: {planet}")

#mark = st.slider("Pick a mark", 0, 100, 50, format="%d", help="Bad=0, Good=50, Excellent=100")
#st.write(f"Your mark: {mark}")

#number = st.slider("Pick a number", 0, 50, 10)
#st.write(f"Number selected: {number}")

#nomor5
#number = st.number_input("Pick a number", min_value=0, max_value=100, value=1)
#st.write(f"You selected: {number}")

#email = st.text_input("Email address")
#st.write(f"Your email: {email}")

#travel_date = st.date_input("Travelling date")
#st.write(f"Travel date: {travel_date}")

#school_time = st.time_input("School time")
#st.write(f"School time: {school_time}")

#description = st.text_area("Description", placeholder="Enter a description...")
#t.write(f"Description: {description}")

#uploaded_file = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])
#if uploaded_file:
    #st.image(uploaded_file, caption="Uploaded photo", use_column_width=True)

#favorite_color = st.color_picker("Choose your favourite color", "#0000FF")
#st.write(f"Your favorite color: {favorite_color}")

#nomor 6
#import numpy as np
#import altair as alt
#import pandas as pd
#import streamlit as st

#st.header('st.write')
#st.write('Hello, World! :sunglasses:')
#st.write(1234)
#df = pd.DataFrame({
    #'first column': [1, 2, 3, 4],
    #'second column': [10, 20, 30, 40]
#})
#st.write(df)

#st.write('Below is a DataFrame:', df, 'Above is a dataframe.')

#df2 = pd.DataFrame(
    #np.random.randn(200, 3),
    #columns=['a', 'b', 'c']
#)
#c = alt.Chart(df2).mark_circle().encode(
    #x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c']
#)
#st.write(c)

#nomer7
#import streamlit as st
#import pandas as pd
#import numpy as np

#df = pd.DataFrame(
    #np.random.randn(10, 2),
    #columns=['x', 'y']
#)
#st.line_chart(df)

#nomer8
#import streamlit as st
#import pandas as pd
#import altair as alt
#import pickle

#filename = ''
#st.title("Build a Machine Learning Application")
#st.sidebar.title("Select Page")

#page = st.sidebar.selectbox("Choose an option", ["Home", "Dataset", "Visualization"])


#if page == "Home":
    #st.header("Welcome to the Home Page")
    #st.image(
        #"uang.jpg", 
        #caption="A Businessman Idea", 
        #use_column_width=True
    #)
    #st.write("This is a demo application to display an image, dataset, and visualization.")

#elif page == "Dataset":
    #st.header("Dataset")
    #data = {
        #"Loan_ID": ["LP001002", "LP001003", "LP001005", "LP001006", "LP001008"],
        #"Gender": ["Male", "Male", "Male", "Male", "Male"],
        #"Married": ["No", "Yes", "Yes", "No", "No"],
        #"Dependents": [0, 1, 0, 0, 0],
        #"Education": ["Graduate", "Graduate", "Graduate", "Not Graduate", "Not Graduate"],
        #"Self_Employed": ["No", "No", "Yes", "No", "No"]
    #}
    #df = pd.DataFrame(data)
    #st.write(df)

#elif page == "Visualization":
    #st.header("Visualization")
    #df = pd.DataFrame({
        #"ApplicantIncome": [2500, 3000, 4000, 4500, 6000, 8000, 12000],
        #"LoanAmount": [100, 120, 140, 150, 180, 250, 300]
    #})
    
    #chart = alt.Chart(df).mark_bar().encode(
        #x="ApplicantIncome",
        #y="LoanAmount"
    #)
    #st.altair_chart(chart, use_container_width=True)

#nomer11
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from wordcloud import WordCloud
import pickle

# Judul Aplikasi
st.title('Aplikasi Prediksi Harga Mobil')
st.sidebar.header("Navigasi")
menu = st.sidebar.selectbox("Pilih Halaman", ["Dataset", "Visualisasi", "Prediksi Harga"])

# Load dataset
df = pd.read_csv("CarPrice.csv")
st.sidebar.write(f"Total data: {len(df)}")

# Fungsi untuk menyimpan model
def save_model(model, filename="model_prediksi_harga_mobil.sav"):
    pickle.dump(model, open(filename, 'wb'))

# Fungsi untuk memuat model
def load_model(filename="model_prediksi_harga_mobil.sav"):
    return pickle.load(open(filename, 'rb'))

# Styling the page for a more modern look
st.markdown("""
    <style>
        body {
            background-color: #f1f1f1;
            font-family: 'Segoe UI', sans-serif;
            color: #333;
        }
        .stButton>button {
            background-color: #007BFF;
            color: white;
            font-weight: bold;
            font-size: 16px;
            padding: 12px;
            border-radius: 10px;
            width: 100%;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #0056b3;
            transform: translateY(-5px);
        }
        .stSelectbox>div>div>button {
            background-color: #007BFF;
            color: white;
            font-weight: bold;
            width: 100%;
            border-radius: 8px;
        }
        .stSelectbox>div>div>button:hover {
            background-color: #0056b3;
        }
        .stTextInput>div>div>input {
            background-color: #e9f6ff;
            border-radius: 8px;
            padding: 8px;
            font-size: 16px;
        }
        .stDataFrame {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        }
        .stSubheader {
            color: #007BFF;
            font-size: 20px;
            font-weight: bold;
        }
        .stTitle {
            font-size: 32px;
            font-weight: bold;
            color: #0056b3;
        }
        .stMarkdown {
            font-size: 16px;
            color: #333;
        }
        .stText {
            font-size: 16px;
        }
        .stAlert {
            color: #ff4757;
            font-weight: bold;
        }
        .stSuccess {
            color: #2ed573;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Jika menu adalah Dataset
if menu == "Dataset":
    st.subheader("Dataset Mobil")
    st.dataframe(df, use_container_width=True)

    # Statistik deskriptif
    st.subheader("Statistik Deskriptif")
    st.write(df.describe())

    # Data kosong
    st.subheader("Data Kosong")
    missing_data = df.isnull().sum()
    st.write(missing_data)

# Jika menu adalah Visualisasi
elif menu == "Visualisasi":
    st.subheader("Visualisasi Data Mobil")

    # Distribusi harga mobil
    st.subheader("Distribusi Harga Mobil")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["price"], kde=True, color="blue", ax=ax)
    ax.set_title("Distribusi Harga Mobil", fontsize=16, fontweight='bold')
    ax.set_xlabel("Harga", fontsize=12)
    ax.set_ylabel("Frekuensi", fontsize=12)
    st.pyplot(fig)

    # Distribusi jumlah mobil berdasarkan nama
    st.subheader("Distribusi Jumlah Mobil Berdasarkan Nama")
    car_counts = df['CarName'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    car_counts.plot(kind="bar", color="skyblue", ax=ax)
    ax.set_title("Top 10 Mobil Berdasarkan Jumlah", fontsize=16, fontweight='bold')
    ax.set_xlabel("Nama Mobil", fontsize=12)
    ax.set_ylabel("Jumlah Mobil", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # Scatter plot antara highway-mpg dan price
    st.subheader("Hubungan antara Highway MPG dan Harga")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df['highwaympg'], y=df['price'], color="purple", ax=ax)
    ax.set_title("Highway MPG vs Harga Mobil", fontsize=16, fontweight='bold')
    ax.set_xlabel("Highway MPG", fontsize=12)
    ax.set_ylabel("Harga", fontsize=12)
    st.pyplot(fig)

    # Word Cloud untuk nama mobil
    st.subheader("Word Cloud Nama Mobil")
    all_cars = " ".join(df['CarName'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_cars)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)

# Jika menu adalah Prediksi Harga
elif menu == "Prediksi Harga":
    st.subheader("Prediksi Harga Mobil")

    # Pemisahan fitur dan target
    X = df[['highwaympg', 'curbweight', 'horsepower']]
    y = df['price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Latih model
    model = LinearRegression()
    model.fit(X_train, y_train)
    save_model(model)

    # Input pengguna
    st.write("Masukkan Spesifikasi Mobil:")
    highway_mpg = st.number_input('Highway MPG', min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    curbweight = st.number_input('Curb Weight', min_value=0, max_value=5000, value=2500, step=10)
    horsepower = st.number_input('Horsepower', min_value=0, max_value=500, value=150, step=5)

    # Prediksi berdasarkan input
    if st.button("Prediksi"):
        loaded_model = load_model()
        input_data = pd.DataFrame({'highwaympg': [highway_mpg], 'curbweight': [curbweight], 'horsepower': [horsepower]})
        predicted_price = loaded_model.predict(input_data)[0]
        st.success(f"Harga mobil yang diprediksi: ${predicted_price:,.2f}")

    # Evaluasi model
    model_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, model_pred)
    mse = mean_squared_error(y_test, model_pred)
    rmse = np.sqrt(mse)

    st.write("\n### Evaluasi Model:")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

