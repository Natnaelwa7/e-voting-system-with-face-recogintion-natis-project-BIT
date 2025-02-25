from flask import Flask, render_template, request, redirect, url_for, flash,Response,jsonify,json
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
import os
import uuid
import base64
import re
from scipy.spatial.distance import cosine
import cv2
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from deepface import DeepFace
from dotenv import load_dotenv
import logging
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///evoting.db'
app.config['SECRET_KEY'] = secrets.token_hex(16)  # Secure secret key
db = SQLAlchemy(app)

# Database Models
class Voter(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    facial_data = db.Column(db.Text, nullable=True)
    has_voted = db.Column(db.Boolean, default=False)

class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    votes = db.Column(db.Integer, default=0)

class ElectionOfficer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Initialize Sample Data
def init_sample_data():
    if not Candidate.query.first():
        db.session.add(Candidate(name="Candidate A"))
        db.session.add(Candidate(name="Candidate B"))
        db.session.commit()

# Routes
@app.route('/')
def home():
    return render_template('home.html')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            data = request.json
            name = data.get('name')
            email = data.get('email')
            password = data.get('password')
            image_data = data.get('image', '').split(',')[1] if data.get('image') else None
            
            if not (name and email and password and image_data):
                return jsonify({'success': False, 'message': 'All fields are required'}), 400
            
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

            if Voter.query.filter_by(email=email).first():
                return jsonify({'success': False, 'message': 'Email already registered'}), 400
            
            filename = f"temp_{uuid.uuid4()}.jpg"
            with open(filename, 'wb') as f:
                f.write(base64.b64decode(image_data))
            
            img = cv2.imread(filename)
            if img is None:
                os.remove(filename)
                return jsonify({'success': False, 'message': 'Invalid image format'}), 400

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            if len(faces) == 0:
                os.remove(filename)
                return jsonify({'success': False, 'message': 'No face detected. Please try again.'}), 400

            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                embedding_data = DeepFace.represent(face_img, model_name='Facenet')
                
                if not embedding_data:
                    os.remove(filename)
                    return jsonify({'success': False, 'message': 'Face recognition failed. Try again with a clearer image.'}), 400

                embedding = np.array(embedding_data[0]['embedding'])
                os.remove(filename)

                # Check if face is already registered using cosine similarity
                existing_voters = Voter.query.all()
                for voter in existing_voters:
                    stored_embedding = np.array(json.loads(voter.facial_data))
                    similarity = 1 - cosine(embedding, stored_embedding)
                    if similarity > 0.7:  # Threshold for considering as the same face
                        return jsonify({'success': False, 'message': 'Face already registered with a different name/email.'}), 400

                new_voter = Voter(
                    name=name,
                    email=email,
                    password=hashed_password,
                    facial_data=json.dumps(embedding.tolist())
                )
                db.session.add(new_voter)
                db.session.commit()

                return jsonify({'success': True, 'message': 'Registration successful! Please log in.'}), 201
        
        except Exception as e:
            logging.error(f"Registration error: {str(e)}")
            return jsonify({'success': False, 'message': f'An error occurred during registration: {str(e)}'}), 500
    
    return render_template('register.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')  # Return login page for GET requests

    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        image_data = data.get('image')

        # Ensure email and password exist
        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required'}), 400

        user = Voter.query.filter_by(email=email).first()
        if not user:
            return jsonify({'success': False, 'message': 'User not found'}), 400

        # Check password
        if not check_password_hash(user.password, password):
            return jsonify({'success': False, 'message': 'Incorrect password'}), 400

        # Validate image data (for face recognition)
        if image_data:
            try:
                # Extract and decode image
                image_data = image_data.split(',')[1]  # Extract base64 image content
                filename = f"temp_{uuid.uuid4()}.jpg"
                with open(filename, 'wb') as f:
                    f.write(base64.b64decode(image_data))

                # Read image with OpenCV
                img = cv2.imread(filename)
                if img is None:
                    os.remove(filename)
                    return jsonify({'success': False, 'message': 'Invalid image format'}), 400

                # Extract facial embedding
                embeddings = DeepFace.represent(img, model_name='Facenet')
                if not embeddings:
                    os.remove(filename)
                    return jsonify({'success': False, 'message': 'No face detected in the image'}), 400
                
                current_embedding = embeddings[0]["embedding"]

                # Retrieve stored embedding
                stored_embedding = eval(user.facial_data)
                distance = np.linalg.norm(np.array(current_embedding) - np.array(stored_embedding))

                os.remove(filename)  # Clean up image file

                # Verify face
                if distance < 10:  # Threshold for face recognition
                    return jsonify({
                        'success': True,
                        'redirect': url_for('vote', voter_id=user.id),  # Pass the user ID
                        'message': f'Login successful. Welcome, {user.name}!'
                    })

                return jsonify({'success': False, 'message': 'Face not recognized'}), 400

            except Exception as e:
                logging.error(f"Face recognition error: {str(e)}")
                if os.path.exists(filename):
                    os.remove(filename)
                return jsonify({'success': False, 'message': 'Face verification failed'}), 400

    except Exception as e:
        logging.error(f"Login process error: {str(e)}")
        return jsonify({'success': False, 'message': 'An internal error occurred'}), 500






@app.route('/vote/<int:voter_id>', methods=['GET', 'POST'])
def vote(voter_id):
    voter = Voter.query.get_or_404(voter_id)
    candidates = Candidate.query.all()

    if request.method == 'POST':
        selected_candidate_id = request.form.get('candidate')
        if not selected_candidate_id:
            flash('Please select a candidate!', 'warning')
            return redirect(url_for('vote', voter_id=voter_id))

        candidate = Candidate.query.get(selected_candidate_id)
        if not candidate:
            flash('Invalid candidate selected!', 'danger')
            return redirect(url_for('vote', voter_id=voter_id))

        candidate.votes += 1
        voter.has_voted = True
        db.session.commit()
        flash('Your vote has been recorded successfully!', 'success')
        return redirect(url_for('home'))

    return render_template('vote.html', voter=voter, candidates=candidates)

@app.route('/results')
def results():
    candidates = Candidate.query.all()
    return render_template('results.html', candidates=candidates)

@app.route('/admin/register', methods=['GET', 'POST'])
def admin_register():
    if request.method == 'POST':
        name = request.form['name'].strip()
        email = request.form['email'].strip()
        password = request.form['password'].strip()

        # Input validation
        if not name:
            flash('Name cannot be empty!', 'danger')
            return redirect(url_for('admin_register'))
        if not email or '@' not in email:
            flash('Please enter a valid email address!', 'danger')
            return redirect(url_for('admin_register'))
        if not password or len(password) < 6:
            flash('Password must be at least 6 characters long!', 'danger')
            return redirect(url_for('admin_register'))

        # Check if email already exists
        existing_officer = ElectionOfficer.query.filter_by(email=email).first()
        if existing_officer:
            flash('Email already registered!', 'warning')
            return redirect(url_for('admin_register'))

        # Hash the password and create a new officer
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_officer = ElectionOfficer(name=name, email=email, password=hashed_password)
        db.session.add(new_officer)
        db.session.commit()
        flash('Election Officer registered successfully!', 'success')
        return redirect(url_for('admin_login'))

    return render_template('admin_register.html')

@app.route('/admin/add_candidate', methods=['GET', 'POST'])
def add_candidate():
    if request.method == 'POST':
        name = request.form['name'].strip()
        if not name:
            flash('Candidate name cannot be empty!', 'danger')
            return redirect(url_for('add_candidate'))
        new_candidate = Candidate(name=name)
        db.session.add(new_candidate)
        db.session.commit()
        flash(f'Candidate "{name}" added successfully!', 'success')
        return redirect(url_for('admin_dashboard'))

    return render_template('add_candidate.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form['email'].strip()
        password = request.form['password'].strip()

        # Input validation
        if not email or '@' not in email:
            flash('Please enter a valid email address!', 'danger')
            return redirect(url_for('admin_login'))
        if not password:
            flash('Password cannot be empty!', 'danger')
            return redirect(url_for('admin_login'))

        # Authenticate the officer
        officer = ElectionOfficer.query.filter_by(email=email).first()
        if officer and check_password_hash(officer.password, password):
            flash('Login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid credentials!', 'danger')

    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    voters = Voter.query.all()
    candidates = Candidate.query.all()
    officers = ElectionOfficer.query.all()

    return render_template('admin_dashboard.html', voters=voters, candidates=candidates, officers=officers)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        init_sample_data()
    app.run(debug=True)