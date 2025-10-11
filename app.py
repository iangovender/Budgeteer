from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from models import db, User, Expense, Budget  # Import models and db from models.py

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random string for security
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///expense_tracker.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)  # Initialize db with app

# Rule-based function to categorize expenses (replacing FinBERT)
def categorize_expense(description):
    description = description.lower()
    categories = ['Food', 'Transport', 'Entertainment', 'Groceries', 'Bills', 'Shopping', 'Other']
    
    # Simple keyword-based rules for categorization
    if any(keyword in description for keyword in ['coffee', 'food', 'restaurant', 'dinner', 'lunch']):
        return 'Food', 0.9
    elif any(keyword in description for keyword in ['taxi', 'bus', 'fuel', 'train', 'flight']):
        return 'Transport', 0.9
    elif any(keyword in description for keyword in ['movie', 'concert', 'game', 'ticket']):
        return 'Entertainment', 0.9
    elif any(keyword in description for keyword in ['grocery', 'supermarket', 'milk', 'bread']):
        return 'Groceries', 0.9
    elif any(keyword in description for keyword in ['electricity', 'water', 'internet', 'phone']):
        return 'Bills', 0.9
    elif any(keyword in description for keyword in ['clothing', 'electronics', 'store', 'mall']):
        return 'Shopping', 0.9
    else:
        return 'Other', 0.8  # Default category with slightly lower confidence

# Helper to generate pie chart image
def generate_pie_chart(category_totals):
    if not category_totals:
        # Return None if no data
        return None
    
    # Prepare data for pie chart
    labels = list(category_totals.keys())
    sizes = list(category_totals.values())
    
    # Create pie chart
    plt.figure(figsize=(8, 6))
    
    # Use numpy for calculations
    total = np.sum(sizes)
    
    # Create pie chart with customization
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90, shadow=True)
    
    # Style the chart
    plt.title('Spending by Category', fontsize=16, fontweight='bold')
    
    # Make the percentages inside the pie chart more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    
    # Save chart to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    
    # Encode the image to base64
    image_png = buffer.getvalue()
    chart_image = base64.b64encode(image_png).decode('utf-8')
    buffer.close()
    plt.close()  # Close the figure to free memory
    
    return chart_image

# Helper to check if user is logged in
def login_required(f):
    def wrap(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email_username = request.form['email_username']
        password = request.form['password']
        user = User.query.filter((User.email == email_username) | (User.username == email_username)).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        flash('Invalid credentials.')
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
        flash('Account already exists.')
        return redirect(url_for('login'))
    hashed_pw = generate_password_hash(password)
    new_user = User(username=username, email=email, password_hash=hashed_pw)
    db.session.add(new_user)
    db.session.commit()
    flash('Account created! Please log in.')
    return redirect(url_for('login'))

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    # Use Pandas/NumPy for analytics
    expenses = Expense.query.filter_by(user_id=user_id).all()
    if not expenses:
        total_spending = 0
        recent_expenses = []
        chart_image = None
        categories_count = 0
        remaining_budget = 0
    else:
        # Create expense data safely
        expense_data = []
        for e in expenses:
            expense_data.append({
                'date': e.date,
                'amount': e.amount,
                'category': e.category
            })
        
        df = pd.DataFrame(expense_data)
        total_spending = np.sum(df['amount'].values)
        recent_expenses = expenses[-5:]  # Last 5
        
        # Calculate category totals for pie chart - with error handling
        if not df.empty and 'category' in df.columns:
            category_sum = df.groupby('category')['amount'].sum().to_dict()
        else:
            category_sum = {}
        
        chart_image = generate_pie_chart(category_sum)
        
        categories_count = len(category_sum)
        
        # Calculate remaining budget (simple implementation)
        budgets = Budget.query.filter_by(user_id=user_id).all()
        total_budget = sum(budget.monthly_limit for budget in budgets)
        remaining_budget = max(0, total_budget - total_spending)
    
    return render_template('dashboard.html', 
                         total_spending=total_spending, 
                         recent_expenses=recent_expenses, 
                         chart_image=chart_image,
                         categories_count=categories_count,
                         remaining_budget=remaining_budget)

@app.route('/expense_entry', methods=['GET'])
@login_required
def expense_entry():
    recent_expenses = Expense.query.filter_by(user_id=session['user_id']).order_by(Expense.created_at.desc()).limit(5).all()
    return render_template('expense_entry.html', recent_expenses=recent_expenses)

@app.route('/add_expense', methods=['POST'])
@login_required
def add_expense():
    date = datetime.strptime(request.form['date'], '%Y-%m-%d').date()
    amount = float(request.form['amount'])
    description = request.form['description']
    category, confidence = categorize_expense(description)
    new_expense = Expense(user_id=session['user_id'], date=date, amount=amount, description=description, category=category, confidence_score=confidence)
    db.session.add(new_expense)
    db.session.commit()
    # Check budget alert (simple example)
    budget = Budget.query.filter_by(user_id=session['user_id'], category=category).first()
    if budget and amount > budget.monthly_limit:
        flash('Warning: This expense exceeds your budget for ' + category)
    return render_template('expense_entry.html', category=category, amount=amount, description=description, confidence_score=confidence, recent_expenses=Expense.query.filter_by(user_id=session['user_id']).order_by(Expense.created_at.desc()).limit(5).all())

@app.route('/budget_settings', methods=['GET'])
@login_required
def budget_settings():
    categories = ['Food', 'Transport', 'Entertainment', 'Groceries', 'Bills', 'Shopping', 'Other']
    budgets = {b.category: b.monthly_limit for b in Budget.query.filter_by(user_id=session['user_id']).all()}
    
    # Calculate current spending without pandas to avoid KeyError
    expenses = Expense.query.filter_by(user_id=session['user_id']).all()
    current_spending = {category: 0 for category in categories}  # Initialize all categories to 0
    
    for expense in expenses:
        if expense.category in current_spending:
            current_spending[expense.category] += expense.amount
        else:
            # Handle unexpected categories by adding them
            current_spending[expense.category] = expense.amount
    
    over_budget = any(current_spending.get(c, 0) > budgets.get(c, 0) for c in categories)
    return render_template('budget_settings.html', categories=categories, budgets=budgets, current_spending=current_spending, over_budget=over_budget)

@app.route('/update_budgets', methods=['POST'])
@login_required
def update_budgets():
    categories = ['Food', 'Transport', 'Entertainment', 'Groceries', 'Bills', 'Shopping', 'Other']
    for cat in categories:
        limit = float(request.form.get(f'limit_{cat}', 0))
        existing = Budget.query.filter_by(user_id=session['user_id'], category=cat).first()
        if existing:
            existing.monthly_limit = limit
        else:
            new_budget = Budget(user_id=session['user_id'], category=cat, monthly_limit=limit)
            db.session.add(new_budget)
    db.session.commit()
    flash('Budgets updated!')
    return redirect(url_for('budget_settings'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Creates the database tables if they don't exist
    app.run(debug=True)