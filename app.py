from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from models import db, User, Expense, Budget
from flask_migrate import Migrate
from expense_classifier import classifier  # Import the classifier

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///expense_tracker.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
migrate = Migrate(app, db)

# Categories for the application (fallback categories)
CATEGORIES = ['Food', 'Transport', 'Entertainment', 'Groceries', 'Bills', 'Shopping', 'Other']

# Rule-based categorization (fallback)
def categorize_expense_rule_based(description):
    description = description.lower()
    
    if any(keyword in description for keyword in ['coffee', 'food', 'restaurant', 'dinner', 'lunch', 'cafe']):
        return 'Food', 0.9
    elif any(keyword in description for keyword in ['taxi', 'bus', 'fuel', 'train', 'flight', 'uber']):
        return 'Transport', 0.9
    elif any(keyword in description for keyword in ['movie', 'concert', 'game', 'ticket', 'netflix']):
        return 'Entertainment', 0.9
    elif any(keyword in description for keyword in ['groceries', 'supermarket', 'milk', 'bread', 'food']):
        return 'Groceries', 0.9
    elif any(keyword in description for keyword in ['electricity', 'water', 'internet', 'phone', 'bill']):
        return 'Bills', 0.9
    elif any(keyword in description for keyword in ['clothing', 'electronics', 'store', 'mall', 'shop']):
        return 'Shopping', 0.9
    else:
        return 'Other', 0.8

# Hybrid categorization function
def categorize_expense(description):
    """
    Hybrid categorization: Try DistilBERT model first, fall back to rule-based
    Returns: (category, confidence_score, method_used)
    """
    # Try model prediction first
    model_category, model_confidence = classifier.predict(description)
    
    if model_category is not None:
        return model_category, model_confidence, 'model'
    else:
        # Fall back to rule-based
        rule_category, rule_confidence = categorize_expense_rule_based(description)
        return rule_category, rule_confidence, 'rule_based'

def generate_pie_chart(category_totals):
    if not category_totals:
        return None
    
    labels = list(category_totals.keys())
    sizes = list(category_totals.values())
    
    plt.figure(figsize=(8, 6))
    total = np.sum(sizes)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90, shadow=True)
    
    plt.title('Spending by Category', fontsize=16, fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.axis('equal')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    
    image_png = buffer.getvalue()
    chart_image = base64.b64encode(image_png).decode('utf-8')
    buffer.close()
    plt.close()
    
    return chart_image

def login_required(f):
    def wrap(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email_username = request.form['email_username']
        password = request.form['password']
        user = User.query.filter((User.email == email_username) | (User.username == email_username)).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid credentials.', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    
    if User.query.filter_by(username=username).first():
        flash('Username already exists.', 'danger')
        return redirect(url_for('login'))
    if User.query.filter_by(email=email).first():
        flash('Email already registered.', 'danger')
        return redirect(url_for('login'))
    
    hashed_pw = generate_password_hash(password)
    new_user = User(username=username, email=email, password_hash=hashed_pw, monthly_income=0.0)
    db.session.add(new_user)
    db.session.commit()
    flash('Account created successfully! Please set up your monthly income in your profile.', 'success')
    return redirect(url_for('login'))

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    user = User.query.get(user_id)
    
    expenses = Expense.query.filter_by(user_id=user_id).all()
    if not expenses:
        total_spending = 0
        recent_expenses = []
        chart_data = None  # Changed from chart_image
        categories_count = 0
        remaining_budget = user.monthly_income if user.monthly_income else 0
        category_budgets = {}
    else:
        expense_data = []
        for e in expenses:
            expense_data.append({
                'date': e.date,
                'amount': e.amount,
                'category': e.category
            })
        
        df = pd.DataFrame(expense_data)
        total_spending = np.sum(df['amount'].values)
        recent_expenses = expenses[-5:]
        
        if not df.empty and 'category' in df.columns:
            category_sum = df.groupby('category')['amount'].sum().to_dict()
        else:
            category_sum = {}
        
        # Generate Chart.js compatible data instead of PNG
        chart_data = generate_chartjs_data(category_sum)  # Changed this line
        categories_count = len(category_sum)
        
        # Calculate remaining budget based on income minus total spending
        remaining_budget = user.monthly_income - total_spending if user.monthly_income else -total_spending
        
        # Calculate category budgets
        budgets = Budget.query.filter_by(user_id=user_id).all()
        category_budgets = {}
        for budget in budgets:
            spending = category_sum.get(budget.category, 0)
            category_budgets[budget.category] = {
                'limit': budget.monthly_limit,
                'spending': spending
            }
    
    return render_template('dashboard.html', 
                         total_spending=total_spending,
                         recent_expenses=recent_expenses,
                         chart_data=chart_data,  # Changed from chart_image
                         categories_count=categories_count,
                         monthly_income=user.monthly_income,
                         remaining_budget=remaining_budget,
                         category_budgets=category_budgets)

# Add this new function to generate Chart.js data
def generate_chartjs_data(category_totals):
    if not category_totals:
        return None
    
    # Define a color palette for the chart
    colors = [
        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', 
        '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
    ]
    
    chart_data = {
        'labels': list(category_totals.keys()),
        'data': list(category_totals.values()),
        'colors': colors[:len(category_totals)]
    }
    
    return chart_data

@app.route('/expense_entry', methods=['GET', 'POST'])
@login_required
def expense_entry():
    if request.method == 'POST':
        amount = float(request.form['amount'])
        description = request.form['description']
        date = datetime.strptime(request.form['date'], '%Y-%m-%d')
        
        # Use hybrid categorization
        category, confidence, method = categorize_expense(description)
        
        new_expense = Expense(
            user_id=session['user_id'],
            amount=amount,
            description=description,
            category=category,
            confidence_score=confidence,
            date=date
        )
        
        db.session.add(new_expense)
        db.session.commit()
        
        method_text = "AI model" if method == 'model' else "rule-based system"
        flash(f'Expense of R {amount} added to {category} category! (Classified by {method_text} with {confidence:.2f} confidence)', 'success')
        return redirect(url_for('expense_entry'))
    
    # Get ALL expenses for display, ordered by date (newest first)
    all_expenses = Expense.query.filter_by(user_id=session['user_id']).order_by(Expense.date.desc()).all()
    
    return render_template('expense_entry.html', all_expenses=all_expenses)

@app.route('/delete_expense/<int:expense_id>', methods=['POST'])
@login_required
def delete_expense(expense_id):
    expense = Expense.query.filter_by(id=expense_id, user_id=session['user_id']).first()
    
    if expense:
        db.session.delete(expense)
        db.session.commit()
        flash('Expense deleted successfully!', 'success')
    else:
        flash('Expense not found or you do not have permission to delete it.', 'danger')
    
    return redirect(url_for('expense_entry'))

@app.route('/budget_settings', methods=['GET', 'POST'])
@login_required
def budget_settings():
    user_id = session['user_id']
    
    # Get current spending by category
    expenses = Expense.query.filter_by(user_id=user_id).all()
    current_spending = {}
    for expense in expenses:
        if expense.category in current_spending:
            current_spending[expense.category] += expense.amount
        else:
            current_spending[expense.category] = expense.amount
    
    # Get current budgets
    budgets = {}
    budget_objects = Budget.query.filter_by(user_id=user_id).all()
    for budget in budget_objects:
        budgets[budget.category] = budget.monthly_limit
    
    if request.method == 'POST':
        # Update budgets
        for category in CATEGORIES:
            limit_key = f'limit_{category}'
            if limit_key in request.form:
                monthly_limit = float(request.form[limit_key]) if request.form[limit_key] else 0
                
                budget = Budget.query.filter_by(user_id=user_id, category=category).first()
                if budget:
                    budget.monthly_limit = monthly_limit
                else:
                    budget = Budget(user_id=user_id, category=category, monthly_limit=monthly_limit)
                    db.session.add(budget)
        
        db.session.commit()
        flash('Budgets updated successfully!', 'success')
        return redirect(url_for('budget_settings'))
    
    # Check if any category is over budget
    over_budget = False
    for category, spending in current_spending.items():
        if category in budgets and budgets[category] > 0 and spending > budgets[category]:
            over_budget = True
            break
    
    return render_template('budget_settings.html', 
                         categories=CATEGORIES,
                         budgets=budgets,
                         current_spending=current_spending,
                         over_budget=over_budget)

@app.route('/update_budgets', methods=['POST'])
@login_required
def update_budgets():
    user_id = session['user_id']
    
    for category in CATEGORIES:
        limit_key = f'limit_{category}'
        if limit_key in request.form:
            monthly_limit = float(request.form[limit_key]) if request.form[limit_key] else 0
            
            budget = Budget.query.filter_by(user_id=user_id, category=category).first()
            if budget:
                budget.monthly_limit = monthly_limit
            else:
                budget = Budget(user_id=user_id, category=category, monthly_limit=monthly_limit)
                db.session.add(budget)
    
    db.session.commit()
    flash('Budgets updated successfully!', 'success')
    return redirect(url_for('budget_settings'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user_id = session['user_id']
    user = User.query.get(user_id)
    
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        
        # Verify current password
        if not check_password_hash(user.password_hash, current_password):
            flash('Current password is incorrect.', 'danger')
            return redirect(url_for('profile'))
        
        # Check if username is already taken by another user
        if username != user.username and User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return redirect(url_for('profile'))
        
        # Check if email is already taken by another user
        if email != user.email and User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return redirect(url_for('profile'))
        
        # Update user details
        user.username = username
        user.email = email
        
        # Update password if provided
        if new_password:
            user.password_hash = generate_password_hash(new_password)
        
        db.session.commit()
        
        # Update session username if changed
        session['username'] = username
        
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))
    
    return render_template('profile.html', user=user)

@app.route('/update_income', methods=['POST'])
@login_required
def update_income():
    user_id = session['user_id']
    user = User.query.get(user_id)
    
    user.monthly_income = float(request.form['monthly_income'])
    db.session.commit()
    
    flash('Monthly income updated successfully!', 'success')
    return redirect(url_for('profile'))

@app.route('/model_status')
@login_required
def model_status():
    """Endpoint to check model status"""
    status = {
        'model_loaded': classifier.model is not None,
        'tokenizer_loaded': classifier.tokenizer is not None,
        'device': str(classifier.device),
        'num_categories': classifier.model.config.num_labels if classifier.model else 0
    }
    return jsonify(status)

@app.route('/budget_forecast')
@login_required
def budget_forecast():
    """Budget and forecast page - placeholder for future implementation"""
    return render_template('budget_forecast.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)