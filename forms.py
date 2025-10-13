from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, EmailField, FloatField, DateField, TextAreaField
from wtforms.validators import DataRequired, Email, Length, NumberRange

class LoginForm(FlaskForm):
    email_username = StringField('Email or Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])

class SignupForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=50)])
    email = EmailField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    monthly_income = FloatField('Monthly Income (R)', validators=[NumberRange(min=0)])

class ExpenseForm(FlaskForm):
    date = DateField('Date', validators=[DataRequired()])
    amount = FloatField('Amount (R)', validators=[DataRequired(), NumberRange(min=0.01)])
    description = StringField('Description', validators=[DataRequired(), Length(max=200)])

class BudgetForm(FlaskForm):
    food_limit = FloatField('Food', validators=[NumberRange(min=0)], default=0)
    transport_limit = FloatField('Transport', validators=[NumberRange(min=0)], default=0)
    entertainment_limit = FloatField('Entertainment', validators=[NumberRange(min=0)], default=0)
    groceries_limit = FloatField('Groceries', validators=[NumberRange(min=0)], default=0)
    bills_limit = FloatField('Bills', validators=[NumberRange(min=0)], default=0)
    shopping_limit = FloatField('Shopping', validators=[NumberRange(min=0)], default=0)
    other_limit = FloatField('Other', validators=[NumberRange(min=0)], default=0)

class IncomeForm(FlaskForm):
    monthly_income = FloatField('Monthly Income (R)', validators=[NumberRange(min=0)])