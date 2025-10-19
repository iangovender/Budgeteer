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

CATEGORIES = ['Food', 'Transport', 'Entertainment', 'Groceries', 'Bills', 'Shopping', 'Other']
# Define category priorities (lower number = higher priority)
CATEGORY_PRIORITIES = {
    'Bills': 1,      # Highest priority - essential
    'Groceries': 2,  # Essential living expenses
    'Transport': 3,  # Essential for work/life
    'Food': 4,       # Eating out - moderate priority
    'Shopping': 5,   # Discretionary spending
    'Entertainment': 6,  # Leisure - lower priority
    'Other': 7       # Miscellaneous - lowest priority
}

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
        'Groceries': ['groceries', 'supermarket', 'milk', 'bread'],
        'Bills': ['electricity', 'water', 'internet', 'phone', 'bill'],
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
    expenses = Expense.query.filter_by(user_id=user_id).all()
    historical = {}
    for e in expenses:
        historical[e.category] = historical.get(e.category, 0) + e.amount
    recommendations = {}
    for category in CATEGORIES:
        hist = historical.get(category, 0)
        if category in ['Food', 'Groceries', 'Bills', 'Transport']:
            base = monthly_income * 0.5 / 4
        elif category in ['Entertainment', 'Shopping']:
            base = monthly_income * 0.3 / 2
        else:
            base = monthly_income * 0.2 / 1
        recommended = max(base, hist * 1.1)  # 10% buffer
        recommendations[category] = round(recommended, 2)
    return recommendations

def redistribute_excess_budgets(user_id, monthly_income, current_budgets, current_spending):
    """
    Smart budget redistribution based on spending patterns and remaining salary
    """
    if monthly_income <= 0:
        return current_budgets
    
    # Calculate total spending and remaining budget
    total_spending = sum(current_spending.values())
    total_allocated = sum(current_budgets.get(cat, 0) for cat in CATEGORIES)
    remaining_budget = monthly_income - total_spending
    
    # If no remaining budget or negative, return current budgets
    if remaining_budget <= 0:
        return current_budgets
    
    # Create new budgets starting with current spending as baseline
    new_budgets = {}
    
    # First pass: Set minimum budgets based on actual spending with buffer
    for category in CATEGORIES:
        spending = current_spending.get(category, 0)
        current_budget = current_budgets.get(category, 0)
        
        # Essential categories get at least their spending + 15% buffer
        if CATEGORY_PRIORITIES.get(category, 7) <= 3:  # Bills, Groceries, Transport
            new_budgets[category] = max(spending * 1.15, current_budget)
        else:
            # Discretionary categories get at least their spending
            new_budgets[category] = max(spending, current_budget)
    
    # Calculate total after minimum allocation
    total_after_min = sum(new_budgets.values())
    
    # If we're within budget, distribute remaining funds
    if total_after_min <= monthly_income:
        remaining_after_min = monthly_income - total_after_min
        if remaining_after_min > 0:
            # Distribute remaining budget based on priority and need
            distribution_weights = {}
            total_weight = 0
            
            for category in CATEGORIES:
                # Higher priority categories get more weight
                priority_weight = 8 - CATEGORY_PRIORITIES.get(category, 7)  # Invert so higher priority = higher weight
                spending_ratio = current_spending.get(category, 0) / max(total_spending, 1)
                weight = priority_weight * (1 + spending_ratio)
                distribution_weights[category] = weight
                total_weight += weight
            
            # Distribute remaining budget proportionally
            for category, weight in distribution_weights.items():
                share = (weight / total_weight) * remaining_after_min
                new_budgets[category] += share
            
            # Round to 2 decimal places
            for category in CATEGORIES:
                new_budgets[category] = round(new_budgets[category], 2)
                
        return new_budgets
    
    # If we're over budget, we need to reduce allocations
    else:
        overshoot = total_after_min - monthly_income
        
        # Reduce from discretionary categories first (lowest priority)
        discretionary_categories = [cat for cat in CATEGORIES if CATEGORY_PRIORITIES.get(cat, 7) >= 5]
        discretionary_categories.sort(key=lambda x: CATEGORY_PRIORITIES.get(x, 7), reverse=True)
        
        for category in discretionary_categories:
            if overshoot <= 0:
                break
                
            current_allocation = new_budgets[category]
            spending = current_spending.get(category, 0)
            
            # Don't reduce below actual spending for discretionary categories
            min_allocation = spending
            
            if current_allocation > min_allocation:
                reducible = current_allocation - min_allocation
                reduction = min(reducible, overshoot)
                
                new_budgets[category] -= reduction
                overshoot -= reduction
        
        # If still over budget, reduce from moderate priority categories
        if overshoot > 0:
            moderate_categories = [cat for cat in CATEGORIES if CATEGORY_PRIORITIES.get(cat, 7) == 4]  # Food
            for category in moderate_categories:
                if overshoot <= 0:
                    break
                    
                current_allocation = new_budgets[category]
                spending = current_spending.get(category, 0)
                
                # Allow some reduction below spending for moderate categories
                min_allocation = spending * 0.9  # Can go 10% below spending
                
                if current_allocation > min_allocation:
                    reducible = current_allocation - min_allocation
                    reduction = min(reducible, overshoot)
                    
                    new_budgets[category] -= reduction
                    overshoot -= reduction
        
        # Final check - if still over budget, apply proportional reduction to all non-essential
        if overshoot > 0:
            non_essential_categories = [cat for cat in CATEGORIES if CATEGORY_PRIORITIES.get(cat, 7) >= 4]
            total_non_essential = sum(new_budgets[cat] for cat in non_essential_categories)
            
            if total_non_essential > 0:
                for category in non_essential_categories:
                    proportion = new_budgets[category] / total_non_essential
                    reduction = overshoot * proportion
                    new_budgets[category] = max(new_budgets[category] - reduction, current_spending.get(category, 0) * 0.8)
    
    # Ensure final total doesn't exceed monthly income and round values
    final_total = sum(new_budgets.values())
    if final_total > monthly_income:
        adjustment_factor = monthly_income / final_total
        for category in CATEGORIES:
            new_budgets[category] = round(new_budgets[category] * adjustment_factor, 2)
    else:
        for category in CATEGORIES:
            new_budgets[category] = round(new_budgets[category], 2)
    
    return new_budgets

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

def reset_monthly_budgets(user_id):
    """
    Reset monthly spending by creating a psychological fresh start
    """
    try:
        today = date.today()
        current_month = today.strftime('%B %Y')
        
        # Get current month expenses count for reporting
        first_day_current = today.replace(day=1)
        current_month_expenses = Expense.query.filter(
            Expense.user_id == user_id,
            Expense.date >= first_day_current
        ).count()
        
        # The reset is mainly psychological - we don't delete data
        # but we inform the user they're starting fresh
        if current_month_expenses > 0:
            return True, f"Budget reset! {current_month_expenses} expenses archived. Ready for {current_month}!"
        else:
            return True, f"Fresh start! Ready for {current_month}!"
            
    except Exception as e:
        return False, f"Error resetting budgets: {str(e)}"

def get_monthly_reset_status(user_id):
    """
    Check if budgets have been reset for current month
    """
    today = date.today()
    first_day_current = today.replace(day=1)
    
    # Check if there are any expenses for current month
    current_month_expenses = Expense.query.filter(
        Expense.user_id == user_id,
        Expense.date >= first_day_current
    ).count()
    
    return {
        'current_month': today.strftime('%B %Y'),
        'has_current_month_expenses': current_month_expenses > 0,
        'expense_count': current_month_expenses,
        'needs_reset': current_month_expenses == 0  # Simplified logic
    }

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
    
    expenses = Expense.query.filter_by(user_id=user_id).all()
    current_spending = {}
    for expense in expenses:
        current_spending[expense.category] = current_spending.get(expense.category, 0) + expense.amount
    
    budgets = Budget.query.filter_by(user_id=user_id).all()
    budget_dict = {b.category: b.monthly_limit for b in budgets}
    
    recommendations = generate_smart_budget_recommendations(user_id, user.monthly_income)
    
    # Check for smart redistribution
    if request.method == 'POST':
        # Check if user wants smart redistribution
        if 'smart_redistribute' in request.form and user.monthly_income > 0:
            # Apply smart redistribution
            new_budgets = redistribute_excess_budgets(user_id, user.monthly_income, budget_dict, current_spending)
            
            # Update budgets with redistributed amounts
            for category in CATEGORIES:
                new_limit = new_budgets.get(category, 0)
                budget = Budget.query.filter_by(user_id=user_id, category=category).first()
                if budget:
                    budget.monthly_limit = new_limit
                else:
                    budget = Budget(user_id=user_id, category=category, monthly_limit=new_limit)
                    db.session.add(budget)
            
            db.session.commit()
            flash('Budgets smartly redistributed based on your spending patterns!', 'success')
            return redirect(url_for('budget_settings'))
        
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
    
    over_budget = any(current_spending.get(cat, 0) > budget_dict.get(cat, 0) for cat in CATEGORIES if cat in budget_dict)
    
    # Calculate redistribution suggestions
    redistribution_suggestions = None
    if user.monthly_income > 0 and over_budget:
        redistribution_suggestions = redistribute_excess_budgets(user_id, user.monthly_income, budget_dict, current_spending)
    
    return render_template('budget_settings.html', 
                         categories=CATEGORIES,
                         budgets=budget_dict,
                         current_spending=current_spending,
                         over_budget=over_budget,
                         recommendations=recommendations,
                         redistribution_suggestions=redistribution_suggestions,
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
    if form.validate_on_submit():
        user.monthly_income = form.monthly_income.data
        recommendations = generate_smart_budget_recommendations(user_id, user.monthly_income)
        
        for category, limit in recommendations.items():
            budget = Budget.query.filter_by(user_id=user_id, category=category).first()
            if not budget:
                budget = Budget(user_id=user_id, category=category, monthly_limit=limit)
                db.session.add(budget)
        
        db.session.commit()
        flash('Monthly income updated! Smart budgets suggested!', 'success')
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

@app.route('/reset_budgets', methods=['POST'])
@login_required
def reset_budgets():
    """Reset budgets for new month"""
    user_id = session['user_id']
    
    success, message = reset_monthly_budgets(user_id)
    
    if success:
        flash(message, 'success')
    else:
        flash(message, 'danger')
    
    return redirect(url_for('dashboard'))

@app.route('/get_reset_status')
@login_required
def get_reset_status():
    """Get monthly reset status for AJAX calls"""
    user_id = session['user_id']
    status = get_monthly_reset_status(user_id)
    return jsonify(status)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)