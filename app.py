import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import json
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
import io
import base64
import os

# Initialize session state
def init_session():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'welcome'
    if 'selected_car' not in st.session_state:
        st.session_state.selected_car = None

# Database setup
def setup_database():
    conn = sqlite3.connect('car_rental.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS cars (
            id INTEGER PRIMARY KEY,
            model TEXT NOT NULL,
            year INTEGER NOT NULL,
            price REAL NOT NULL,
            category TEXT NOT NULL,
            features TEXT NOT NULL,
            image BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY,
            user_email TEXT NOT NULL,
            car_id INTEGER NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            total_price REAL NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert sample cars if none exist
    c.execute('SELECT COUNT(*) FROM cars')
    if c.fetchone()[0] == 0:
        sample_cars = [
            ('Mercedes S-Class', 2023, 499, 'Luxury', 
             'Panoramic Sunroof,Heated Seats,Apple CarPlay', 
             's_class.jpg'),
            ('BMW 7 Series', 2024, 549, 'Luxury', 
             'Massage Seats,Digital Key,5-Zone Climate'),
            ('Audi Q8', 2023, 599, 'SUV', 
             'Quattro AWD,Virtual Cockpit,Adaptive Air Suspension')
        ]
        for car in sample_cars:
            with open(f"images/{car[5]}", "rb") as f:
                image = f.read()
            c.execute('''
                INSERT INTO cars (model, year, price, category, features, image)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (car[0], car[1], car[2], car[3], car[4], image))
    
    conn.commit()
    conn.close()

# Authentication
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(full_name, email, password):
    conn = sqlite3.connect('car_rental.db')
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO users (full_name, email, password)
            VALUES (?, ?, ?)
        ''', (full_name, email, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(email, password):
    conn = sqlite3.connect('car_rental.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE email = ?', (email,))
    result = c.fetchone()
    conn.close()
    return result and result[0] == hash_password(password)

# AI Recommender
class CarRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model_matrix = None
        self.car_ids = []
    
    def train(self, cars):
        features = [f"{car[1]} {car[4]} {car[3]}" for car in cars]
        self.car_ids = [car[0] for car in cars]
        self.model_matrix = self.vectorizer.fit_transform(features)
    
    def recommend(self, car_id, n=3):
        try:
            idx = self.car_ids.index(car_id)
            sim_scores = cosine_similarity(self.model_matrix[idx], self.model_matrix)
            return [self.car_ids[i] for i in np.argsort(sim_scores[0])[-n-1:-1][::-1]]
        except:
            return []

# Image handling
def load_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image.thumbnail((800, 800))
    return image

# UI Components
def welcome_page():
    st.markdown("<h1 style='text-align: center;'>🚗 Luxury Car Rentals</h1>", unsafe_allow_html=True)
    cols = st.columns([1,2,1])
    with cols[1]:
        if st.button('Browse Cars'):
            st.session_state.current_page = 'browse'
        if st.button('Login'):
            st.session_state.current_page = 'login'
        if st.button('Sign Up'):
            st.session_state.current_page = 'signup'

def login_page():
    with st.form("Login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type='password')
        if st.form_submit_button("Login"):
            if verify_user(email, password):
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.current_page = 'browse'
                st.rerun()
            else:
                st.error("Invalid credentials")
    if st.button("Back"):
        st.session_state.current_page = 'welcome'

def signup_page():
    with st.form("Signup"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type='password')
        if st.form_submit_button("Create Account"):
            if create_user(name, email, password):
                st.success("Account created! Please login")
                st.session_state.current_page = 'login'
            else:
                st.error("Email already exists")
    if st.button("Back"):
        st.session_state.current_page = 'welcome'

def browse_page():
    st.markdown("<h1>Available Cars</h1>", unsafe_allow_html=True)
    
    conn = sqlite3.connect('car_rental.db')
    c = conn.cursor()
    c.execute('SELECT * FROM cars')
    cars = c.fetchall()
    
    # AI Recommendations
    recommender = CarRecommender()
    recommender.train(cars)
    
    cols = st.columns(3)
    for idx, car in enumerate(cars):
        with cols[idx % 3]:
            image = load_image(car[6])
            st.image(image, caption=car[1])
            st.markdown(f"""
                <div style='padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
                    <h3>{car[1]} ({car[2]})</h3>
                    <p>AED {car[3]}/day</p>
                    <p>{car[4]}</p>
                    <p>{car[5]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Book Now", key=f"book_{car[0]}"):
                st.session_state.selected_car = car
                st.session_state.current_page = 'booking'
                st.rerun()
    
    st.markdown("### AI Recommendations")
    if cars:
        rec_ids = recommender.recommend(cars[0][0])
        rec_cars = [c for c in cars if c[0] in rec_ids]
        rec_cols = st.columns(3)
        for idx, car in enumerate(rec_cars):
            with rec_cols[idx % 3]:
                image = load_image(car[6])
                st.image(image, width=200)
                st.write(f"{car[1]} ({car[2]})")

def booking_page():
    car = st.session_state.selected_car
    st.markdown(f"<h1>Book {car[1]}</h1>", unsafe_allow_html=True)
    
    with st.form("Booking"):
        start = st.date_input("Start Date", datetime.now())
        end = st.date_input("End Date", datetime.now() + timedelta(days=1))
        days = (end - start).days + 1
        total = days * car[3]
        
        st.markdown(f"""
            <div style='padding: 20px; background: #f8f9fa; border-radius: 10px;'>
                <h3>Total: AED {total}</h3>
                <p>{days} days rental</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.form_submit_button("Confirm Booking"):
            conn = sqlite3.connect('car_rental.db')
            c = conn.cursor()
            c.execute('''
                INSERT INTO bookings (user_email, car_id, start_date, end_date, total_price)
                VALUES (?, ?, ?, ?, ?)
            ''', (st.session_state.user_email, car[0], start.isoformat(), end.isoformat(), total))
            conn.commit()
            conn.close()
            st.success("Booking confirmed!")
            time.sleep(1)
            st.session_state.current_page = 'browse'
            st.rerun()
    
    if st.button("Back"):
        st.session_state.current_page = 'browse'
        st.rerun()

def main():
    init_session()
    setup_database()
    
    st.set_page_config(
        page_title="Luxury Car Rentals",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    pages = {
        'welcome': welcome_page,
        'login': login_page,
        'signup': signup_page,
        'browse': browse_page,
        'booking': booking_page
    }
    
    if st.session_state.logged_in:
        with st.sidebar:
            st.markdown(f"**Welcome {st.session_state.user_email}**")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.current_page = 'welcome'
                st.rerun()
    
    pages.get(st.session_state.current_page, welcome_page)()

if __name__ == "__main__":
    main()
