from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from forms import LoginForm, SignupForm, ExpenseForm, IncomeForm
from expense_classifier import classifier
from datetime import datetime, date
import numpy as np
import pandas as pd
import os
from functools import wraps
from collections import defaultdict

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///budgeteer.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['WTF_CSRF_ENABLED'] = True

db = SQLAlchemy(app)
migrate = Migrate(app, db)

CATEGORIES = ['Food', 'Transport', 'Entertainment', 'Groceries', 'Utilities', 'Shopping', 'Restaurants', 'Gas & Fuel', 'Other']

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    monthly_income = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expenses = db.relationship('Expense', backref='user', lazy=True)
    budgets = db.relationship('Budget', backref='user', lazy=True)

class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    confidence_score = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Budget(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    monthly_limit = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

def categorize_expense_rule_based(description):
    description = description.lower()
    rules = {
        'Food': ['coffee', 'food', 'restaurant', 'dinner', 'lunch', 'cafe'],
        'Transport': ['taxi', 'bus', 'fuel', 'train', 'flight', 'uber'],
        'Entertainment': ['movie', 'concert', 'game', 'ticket', 'netflix'],
        'Groceries': ['supermarket', 'milk', 'bread'],
        'Utilities': ['electricity', 'water', 'internet', 'phone', 'bill'],
        'Shopping': ['clothing', 'electronics', 'store', 'mall', 'shop']
    }
    for category, keywords in rules.items():
        if any(keyword in description for keyword in keywords):
            return category, 0.9
    return 'Other', 0.8

def categorize_expense(description):
    try:
        model_category, model_confidence = classifier.predict(description)
        if model_category and model_confidence >= 0.7:
            return model_category, model_confidence, 'AI Model'
    except:
        pass
    rule_category, rule_confidence = categorize_expense_rule_based(description)
    return rule_category, rule_confidence, 'Rule-based'

def check_budget_alert(user_id, category, amount):
    budget = Budget.query.filter_by(user_id=user_id, category=category).first()
    if budget and budget.monthly_limit > 0:
        current_spending = db.session.query(db.func.sum(Expense.amount)).filter(
            Expense.user_id == user_id,
            Expense.category == category
        ).scalar() or 0
        new_total = current_spending + amount
        if new_total > budget.monthly_limit:
            return f"Warning: This expense will exceed your {category} budget by R{new_total - budget.monthly_limit:.2f}!"
    return None

def generate_smart_budget_recommendations(user_id, monthly_income):

    if not monthly_income or monthly_income <= 0:
        return {}
    
    # Calculate current month spending ONLY
    today = date.today()
    first_day_current = today.replace(day=1)
    
    current_month_expenses = Expense.query.filter(
        Expense.user_id == user_id,
        Expense.date >= first_day_current
    ).all()
    
    # Get current budgets
    budgets = Budget.query.filter_by(user_id=user_id).all()
    current_budgets = {b.category: b.monthly_limit for b in budgets}
    
    # Current month spending by category
    current_spending = {}
    for e in current_month_expenses:
        current_spending[e.category] = current_spending.get(e.category, 0) + e.amount
    
    # Total current month spending and remaining income
    total_current_spending = sum(current_spending.values())
    remaining_income = monthly_income - total_current_spending
    
    # If remaining income is negative, MUST work within what's available
    available_for_budgets = max(remaining_income, monthly_income * 0.10)  # At least 10% of income as buffer
    
    # rule-based category system with priorities 
    category_rules = {
    'Groceries': {
        'priority': 1,
        'base_percent': 0.20,
        'min_percent': 0.18,
        'max_percent': 0.28,
        'essential': True,
        'description': 'Home food supplies'
    },
    'Utilities': {
        'priority': 1,
        'base_percent': 0.15,
        'min_percent': 0.12,
        'max_percent': 0.20,
        'essential': True,
        'description': 'Electricity, water, internet, phone bills'
    },
    'Transport': {
        'priority': 1,  
        'base_percent': 0.12,
        'min_percent': 0.08,
        'max_percent': 0.18,
        'essential': True,
        'description': 'Public transport, ride-sharing'
    },
    'Gas & Fuel': {
        'priority': 1,  
        'base_percent': 0.10,
        'min_percent': 0.05,
        'max_percent': 0.15,
        'essential': True,
        'description': 'Vehicle fuel costs'
    },
    'Shopping': {  
        'priority': 1,  
        'base_percent': 0.12,  
        'min_percent': 0.08,
        'max_percent': 0.18,
        'essential': True,
        'description': 'Essential shopping, clothing, household items'
    },
    'Restaurants': {
        'priority': 3,
        'base_percent': 0.08,
        'min_percent': 0.03,
        'max_percent': 0.12,
        'essential': False,
        'description': 'Dining out, restaurants'
    },
    'Food': {
        'priority': 4,
        'base_percent': 0.07,
        'min_percent': 0.03,
        'max_percent': 0.12,
        'essential': False,
        'description': 'Dining out, takeaways, snacks'
    },
    'Entertainment': {
        'priority': 4,
        'base_percent': 0.07,
        'min_percent': 0.03,
        'max_percent': 0.12,
        'essential': False,
        'description': 'Movies, games, subscriptions'
    },
    'Other': {
        'priority': 5,
        'base_percent': 0.08,
        'min_percent': 0.03,
        'max_percent': 0.10,
        'essential': False,
        'description': 'Miscellaneous expenses'
    }
    }
    
    recommendations = {}
    
    # 1: Identify over-budget categories and calculate minimum requirements
    over_budget_categories = {}
    essential_requirements = 0
    total_minimum_required = 0
    
    for category in CATEGORIES:
        current_spent = current_spending.get(category, 0)
        rules = category_rules.get(category, {
            'priority': 5,
            'min_percent': 0.03,
            'essential': False
        })
        
        min_budget = monthly_income * rules['min_percent']
        
        # If category is already over budget, MUST accommodate at least the current spending
        if current_spent > current_budgets.get(category, 0):
            over_budget_categories[category] = {
                'current_spent': current_spent,
                'min_required': max(current_spent * 1.05, min_budget),  # At least 5% above current spending
                'essential': rules.get('essential', False),
                'priority': rules.get('priority', 5)
            }
        
        # Calculate total minimum requirements
        if rules.get('essential', False):
            essential_requirements += max(current_spent, min_budget)
        total_minimum_required += max(current_spent, min_budget)
    
    # 2: Calculate base recommendations
    for category in CATEGORIES:
        current_spent = current_spending.get(category, 0)
        rules = category_rules.get(category, {
            'priority': 5,
            'base_percent': 0.05,
            'min_percent': 0.03,
            'max_percent': 0.10,
            'essential': False
        })
        
        base_budget = monthly_income * rules['base_percent']
        min_budget = monthly_income * rules['min_percent']
        max_budget = monthly_income * rules['max_percent']
        
        # If category is over budget, prioritize accommodating current spending
        if category in over_budget_categories:
            recommendations[category] = over_budget_categories[category]['min_required']
        else:
            # Normal recommendation logic
            if current_spent > 0:
                # Project future spending based on current patterns
                day_of_month = today.day
                month_progress = day_of_month / 30
                
                if month_progress > 0.1:
                    projected = current_spent / month_progress
                else:
                    projected = current_spent + base_budget * 0.8
                
                # Ensure projection is reasonable
                if projected > max_budget:
                    recommended = min(projected, max_budget)
                elif projected < min_budget:
                    recommended = max(projected, min_budget)
                else:
                    recommended = projected
                
                recommendations[category] = max(recommended, current_spent * 1.1)
            else:
                # No spending yet
                if rules.get('essential', False):
                    recommendations[category] = base_budget
                else:
                    recommendations[category] = min_budget
    
    # 3: REDISTRIBUTION LOGIC - Handle budget constraints
    total_recommended = sum(recommendations.values())
    
    if total_recommended > available_for_budgets:
        # need to redistribute - prioritize essentials and over-budget categories
        
        # Sort categories by priority (essentials and over-budget first)
        sorted_categories = sorted(CATEGORIES, key=lambda cat: (
            -category_rules.get(cat, {}).get('priority', 5),  # Higher priority first
            cat in over_budget_categories,  # Over-budget categories first
            -recommendations.get(cat, 0)  # Higher amounts first
        ))
        
        # First pass: Ensure minimums for essentials and over-budget categories
        redistributed = {}
        remaining_funds = available_for_budgets
        
        # Allocate to highest priority categories first
        for category in sorted_categories:
            current_rec = recommendations[category]
            rules = category_rules.get(category, {})
            min_budget = monthly_income * rules.get('min_percent', 0.03)
            current_spent = current_spending.get(category, 0)
            
            # Calculate minimum required for this category
            if category in over_budget_categories:
                min_required = over_budget_categories[category]['min_required']
            elif rules.get('essential', False):
                min_required = max(current_spent, min_budget)
            else:
                min_required = max(current_spent * 0.8, min_budget * 0.5)  # Reduced minimum for non-essentials
            
            # Allocate what we can
            if remaining_funds >= min_required:
                allocated = min(current_rec, min_required * 1.2)  # Don't exceed 20% above minimum
                redistributed[category] = min(allocated, remaining_funds)
                remaining_funds -= redistributed[category]
            else:
                # Can't meet minimum - allocate proportionally based on priority
                if remaining_funds > 0:
                    priority_weight = 10 if rules.get('essential', False) else 1
                    allocated = remaining_funds * priority_weight / 10  # Essentials get full share
                    redistributed[category] = allocated
                    remaining_funds -= allocated
                else:
                    redistributed[category] = 0
        
        # Second pass: Distribute any remaining funds proportionally
        if remaining_funds > 0:
            total_allocated = sum(redistributed.values())
            for category in sorted_categories:
                if category in redistributed:
                    proportion = redistributed[category] / total_allocated if total_allocated > 0 else 1/len(redistributed)
                    extra = remaining_funds * proportion
                    redistributed[category] += extra
            
            remaining_funds = 0
        
        recommendations = redistributed
    
    # 4: Final validation and rounding
    total_recommended = sum(recommendations.values())
    
    # Final hard cap if needed 
    if total_recommended > available_for_budgets:
        final_scale = available_for_budgets / total_recommended
        for category in CATEGORIES:
            recommendations[category] = recommendations.get(category, 0) * final_scale
    
    # Ensure all categories have reasonable values and round
    for category in CATEGORIES:
        current_spent = current_spending.get(category, 0)
        final_value = recommendations.get(category, 0)
        
        # Never recommend less than current spending for over-budget categories
        if category in over_budget_categories and final_value < current_spent:
            final_value = current_spent * 1.05
        
        # Apply reasonable bounds
        rules = category_rules.get(category, {})
        max_budget = monthly_income * rules.get('max_percent', 0.10)
        final_value = min(final_value, max_budget)
        
        recommendations[category] = round(max(final_value, 0), 2)
    
    # Final check
    final_total = sum(recommendations.values())
    if final_total > available_for_budgets * 1.05:  # Allow 5% tolerance
        print(f"WARNING: Final recommendations ({final_total}) exceed available ({available_for_budgets})")
        # Emergency scaling
        emergency_scale = available_for_budgets / final_total
        for category in CATEGORIES:
            recommendations[category] = round(recommendations[category] * emergency_scale, 2)
    
    return recommendations

def generate_chartjs_data(category_totals):
    if not category_totals:
        return None
    colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#C9CBCF']
    return {
        'labels': list(category_totals.keys()),
        'data': list(category_totals.values()),
        'colors': colors[:len(category_totals)]
    }

def generate_insights(user_id):
    expenses = Expense.query.filter_by(user_id=user_id).all()
    if not expenses:
        return {'top_categories': {}, 'trends': 'No data yet - add some expenses!', 'tips': 'Start tracking your spending.'}
    
    df = pd.DataFrame([{'category': e.category, 'amount': e.amount} for e in expenses])
    category_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False)
    
    # Convert to dictionary properly
    top_categories_dict = category_totals.head(3).to_dict()
    
    total_spending = df['amount'].sum()
    
    if len(top_categories_dict) > 0:
        top_category = list(top_categories_dict.keys())[0]
        top_amount = list(top_categories_dict.values())[0]
        trends = f"You've spent R{total_spending:.2f} total. Top category: {top_category} (R{top_amount:.2f})"
    else:
        trends = f"You've spent R{total_spending:.2f} total."
    
    # Generate tips based on spending patterns
    if total_spending == 0:
        tips = "Start by adding your first expense to get personalized insights!"
    elif len(category_totals) == 1:
        tips = "Consider diversifying your spending across more categories for better budget management."
    else:
        tips = "Review your spending patterns regularly to stay on track with your budget goals."
    
    return {'top_categories': top_categories_dict, 'trends': trends, 'tips': tips}


def login_required(f):
    @wraps(f)
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
    login_form = LoginForm()
    signup_form = SignupForm()
    
    if request.method == 'POST':
        if 'action' in request.form and request.form['action'] == 'login' and login_form.validate_on_submit():
            email_username = login_form.email_username.data
            password = login_form.password.data
            user = User.query.filter((User.email == email_username) | (User.username == email_username)).first()
            if user and check_password_hash(user.password_hash, password):
                session['user_id'] = user.id
                session['username'] = user.username
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            flash('Invalid credentials.', 'danger')
        elif 'action' in request.form and request.form['action'] == 'signup' and signup_form.validate_on_submit():
            username = signup_form.username.data
            email = signup_form.email.data
            password = signup_form.password.data
            if User.query.filter_by(username=username).first():
                flash('Username already exists.', 'danger')
            elif User.query.filter_by(email=email).first():
                flash('Email already registered.', 'danger')
            else:
                hashed_pw = generate_password_hash(password)
                new_user = User(username=username, email=email, password_hash=hashed_pw, monthly_income=0.0)
                db.session.add(new_user)
                db.session.commit()
                flash('Account created successfully! Please log in.', 'success')
                return redirect(url_for('login'))
    return render_template('login.html', login_form=login_form, signup_form=signup_form)

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    user = db.session.get(User, user_id)
    expenses = Expense.query.filter_by(user_id=user_id).all()
    
    # Add current month info
    today = date.today()
    current_month = today.strftime('%B %Y')
    
    if not expenses:
        total_spending = 0
        recent_expenses = []
        chart_data = None
        categories_count = 0
        category_sum = {}
    else:
        expense_data = [{'date': e.date, 'amount': e.amount, 'category': e.category} for e in expenses]
        df = pd.DataFrame(expense_data)
        total_spending = df['amount'].sum()
        category_sum = df.groupby('category')['amount'].sum().to_dict()
        recent_expenses = expenses[-5:]
        chart_data = generate_chartjs_data(category_sum)
        categories_count = len(category_sum)
    
    remaining_budget = user.monthly_income - total_spending if user.monthly_income else -total_spending
    budgets = Budget.query.filter_by(user_id=user_id).all()
    category_budgets = {}
    for budget in budgets:
        spending = category_sum.get(budget.category, 0)
        category_budgets[budget.category] = {
            'limit': budget.monthly_limit,
            'spending': spending
        }
    insights = generate_insights(user_id)
    
    return render_template('dashboard.html', 
                         total_spending=total_spending, recent_expenses=recent_expenses,
                         chart_data=chart_data, categories_count=categories_count,
                         monthly_income=user.monthly_income, remaining_budget=remaining_budget,
                         category_budgets=category_budgets, insights=insights,
                         current_month=current_month)

@app.route('/expense_entry', methods=['GET', 'POST'])
@login_required
def expense_entry():
    form = ExpenseForm()
    if form.validate_on_submit():
        amount = form.amount.data
        description = form.description.data
        date = form.date.data
        
        category, confidence, method = categorize_expense(description)
        
        alert = check_budget_alert(session['user_id'], category, amount)
        
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
        
        method_text = "AI model" if method == 'AI Model' else "rule-based system"
        flash_msg = f'Expense of R {amount} added to {category} category! (Classified by {method_text} with {confidence:.2f} confidence)'
        if alert:
            flash_msg += f' | {alert}'
            flash(flash_msg, 'warning')
        else:
            flash(flash_msg, 'success')
        return redirect(url_for('expense_entry'))
    
    all_expenses = Expense.query.filter_by(user_id=session['user_id']).order_by(Expense.date.desc()).all()
    return render_template('expense_entry.html', form=form, all_expenses=all_expenses)

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
    user = db.session.get(User, user_id)
    
    # Get CURRENT MONTH expenses only (not all-time)
    today = date.today()
    first_day_current = today.replace(day=1)
    
    current_month_expenses = Expense.query.filter(
        Expense.user_id == user_id,
        Expense.date >= first_day_current
    ).all()
    
    # Calculate current month spending by category
    current_spending = {}
    for expense in current_month_expenses:
        current_spending[expense.category] = current_spending.get(expense.category, 0) + expense.amount
    
    budgets = Budget.query.filter_by(user_id=user_id).all()
    budget_dict = {b.category: b.monthly_limit for b in budgets}
    
    recommendations = generate_smart_budget_recommendations(user_id, user.monthly_income)
    
    if request.method == 'POST':
        # Normal budget update
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
    
    over_budget = any(current_spending.get(cat, 0) > budget_dict.get(cat, 0) for cat in CATEGORIES if cat in budget_dict and budget_dict[cat] > 0)
    
    return render_template('budget_settings.html', 
                         categories=CATEGORIES,
                         budgets=budget_dict,
                         current_spending=current_spending,
                         over_budget=over_budget,
                         recommendations=recommendations,
                         monthly_income=user.monthly_income)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user_id = session['user_id']
    user = db.session.get(User, user_id)
    
    # Add current month info
    today = date.today()
    current_month = today.strftime('%B %Y')
    
    form = IncomeForm()
    
    # Handle account information update
    if request.method == 'POST':
        # Check if this is an account update (not income form)
        if 'username' in request.form or 'email' in request.form or 'new_password' in request.form:
            current_password = request.form.get('current_password', '')
            new_username = request.form.get('username', '').strip()
            new_email = request.form.get('email', '').strip()
            new_password = request.form.get('new_password', '').strip()
            confirm_password = request.form.get('confirm_new_password', '').strip()
            
            # Verify current password
            if not check_password_hash(user.password_hash, current_password):
                flash('Current password is incorrect. Please try again.', 'danger')
                return redirect(url_for('profile'))
            
            changes_made = False
            
            # Update username if changed
            if new_username and new_username != user.username:
                # Check if username is already taken
                existing_user = User.query.filter(User.username == new_username, User.id != user_id).first()
                if existing_user:
                    flash('Username already taken. Please choose a different one.', 'danger')
                else:
                    user.username = new_username
                    session['username'] = new_username
                    changes_made = True
            
            # Update email if changed
            if new_email and new_email != user.email:
                # Check if email is already registered
                existing_user = User.query.filter(User.email == new_email, User.id != user_id).first()
                if existing_user:
                    flash('Email already registered. Please use a different email.', 'danger')
                else:
                    user.email = new_email
                    changes_made = True
            
            # Update password if provided
            if new_password:
                if new_password != confirm_password:
                    flash('New passwords do not match. Please try again.', 'danger')
                elif len(new_password) < 6:
                    flash('Password must be at least 6 characters long.', 'danger')
                else:
                    user.password_hash = generate_password_hash(new_password)
                    changes_made = True
            
            # Commit changes if any were made
            if changes_made:
                db.session.commit()
                flash('Account information updated successfully!', 'success')
            
            return redirect(url_for('profile'))
    
    # Handle income form submission
    if form.validate_on_submit():
        user.monthly_income = form.monthly_income.data
        db.session.commit()
        flash('Monthly income updated successfully!', 'success')
        return redirect(url_for('profile'))
    
    return render_template('profile.html', user=user, form=form, current_month=current_month)

@app.route('/budget_forecast')
@login_required
def budget_forecast():
    user_id = session['user_id']
    user = db.session.get(User, user_id)
    
    # Get current month expenses
    today = date.today()
    first_day_current = today.replace(day=1)
    
    current_expenses = Expense.query.filter(
        Expense.user_id == user_id,
        Expense.date >= first_day_current
    ).all()
    
    # Get historical data for prediction (last 3 months)
    three_months_ago = today.replace(month=today.month-3) if today.month > 3 else today.replace(year=today.year-1, month=today.month+9)
    
    historical_expenses = Expense.query.filter(
        Expense.user_id == user_id,
        Expense.date >= three_months_ago
    ).all()
    
    # Calculate current spending by category
    current_spending = {}
    for expense in current_expenses:
        current_spending[expense.category] = current_spending.get(expense.category, 0) + expense.amount
    
    # Calculate predicted spending (simple moving average + trend)
    predicted_spending = {}
    
    # Group historical expenses by month
    monthly_data = defaultdict(lambda: defaultdict(float))
    
    for expense in historical_expenses:
        month_key = expense.date.strftime('%Y-%m')
        monthly_data[month_key][expense.category] += expense.amount
        monthly_data[month_key]['total'] += expense.amount
    
    # Calculate predictions
    for category in CATEGORIES:
        category_totals = []
        for month_data in monthly_data.values():
            if category in month_data:
                category_totals.append(month_data[category])
        
        if category_totals:
            # Use weighted average (more recent months have higher weight)
            weights = [i+1 for i in range(len(category_totals))]  # Linear weights
            weighted_avg = sum(x * w for x, w in zip(category_totals, weights)) / sum(weights)
            
            # Apply seasonal adjustment and growth trend
            current = current_spending.get(category, 0)
            if len(category_totals) >= 2:
                growth_rate = (category_totals[-1] - category_totals[0]) / max(category_totals[0], 1)
                predicted = weighted_avg * (1 + growth_rate * 0.3)  # Dampened growth
            else:
                predicted = weighted_avg
            
            predicted_spending[category] = max(predicted, current * 0.8)  # At least 80% of current
        else:
            predicted_spending[category] = current_spending.get(category, 0)
    
    # Get budget limits
    budgets = Budget.query.filter_by(user_id=user_id).all()
    budget_dict = {b.category: b.monthly_limit for b in budgets}
    
    # Calculate totals
    predicted_total = sum(predicted_spending.values())
    current_total = sum(current_spending.values())
    predicted_savings = (user.monthly_income or 0) - predicted_total
    
    # Determine trend
    if predicted_total > current_total * 1.1:
        trend_indicator = "Increasing trend"
    elif predicted_total < current_total * 0.9:
        trend_indicator = "Decreasing trend"
    else:
        trend_indicator = "Stable spending"
    
    # Budget health assessment
    if predicted_savings > user.monthly_income * 0.2:
        budget_health = "Excellent"
    elif predicted_savings > 0:
        budget_health = "Good"
    elif predicted_savings > -user.monthly_income * 0.1:
        budget_health = "Needs Attention"
    else:
        budget_health = "At Risk"
    
    # Identify risk areas
    risk_areas = []
    for category in CATEGORIES:
        predicted = predicted_spending.get(category, 0)
        budget = budget_dict.get(category, 0)
        if budget > 0 and predicted > budget:
            risk_areas.append(category)
    
    # Generate AI recommendations
    recommendations = []
    
    if predicted_savings < 0:
        recommendations.append({
            'type': 'danger',
            'icon': 'exclamation-triangle',
            'title': 'Budget Deficit Predicted',
            'message': f'You may overspend by R {abs(predicted_savings):.2f} next month.',
            'suggestion': 'Consider reducing discretionary spending or increasing income.'
        })
    
    if risk_areas:
        recommendations.append({
            'type': 'warning',
            'icon': 'bullhorn',
            'title': 'Budget Risks Detected',
            'message': f'{len(risk_areas)} categories may exceed budgets.',
            'suggestion': 'Review spending in: ' + ', '.join(risk_areas)
        })
    
    if predicted_total < current_total * 0.8:
        recommendations.append({
            'type': 'success',
            'icon': 'trophy',
            'title': 'Positive Trend',
            'message': 'Your spending is predicted to decrease next month!',
            'suggestion': 'Keep up the good financial habits!'
        })
    
    # Add general recommendation if no specific issues
    if not recommendations and predicted_savings > 0:
        recommendations.append({
            'type': 'info',
            'icon': 'lightbulb',
            'title': 'On Track',
            'message': 'Your budget looks healthy for next month.',
            'suggestion': 'Consider allocating extra savings towards financial goals.'
        })
    
    # Seasonal trends (simplified)
    seasonal_trends = []
    high_season_categories = ['Shopping', 'Entertainment']  # Typically increase during holidays
    for category in high_season_categories:
        if category in predicted_spending and predicted_spending[category] > current_spending.get(category, 0):
            seasonal_trends.append({
                'category': category,
                'increase': True
            })
    
    # Prepare chart data
    forecast_data = {
        'labels': CATEGORIES,
        'current': [current_spending.get(cat, 0) for cat in CATEGORIES],
        'predicted': [predicted_spending.get(cat, 0) for cat in CATEGORIES],
        'budgets': [budget_dict.get(cat, 0) for cat in CATEGORIES]
    }
    
    return render_template('budget_forecast.html',
                         predicted_total=predicted_total,
                         predicted_savings=predicted_savings,
                         trend_indicator=trend_indicator,
                         budget_health=budget_health,
                         risk_areas=risk_areas,
                         recommendations=recommendations,
                         seasonal_trends=seasonal_trends,
                         forecast_data=forecast_data,
                         current_spending=current_spending,
                         predicted_spending=predicted_spending,
                         budgets=budget_dict,
                         categories=CATEGORIES,
                         monthly_income=user.monthly_income)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)