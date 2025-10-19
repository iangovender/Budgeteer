from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, FloatField, DateField, SubmitField
from wtforms.validators import DataRequired, Length, NumberRange, EqualTo, ValidationError
from datetime import date
import re

class EmailValidator:
    def __init__(self, message=None):
        if not message:
            message = 'Please enter a valid email address.'
        self.message = message

    def __call__(self, form, field):
        email = field.data
        if email:
            # Simple email regex pattern
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(pattern, email):
                raise ValidationError(self.message)

class LoginForm(FlaskForm):
    email_username = StringField('Email or Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Sign In')

class SignupForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=50)])
    email = StringField('Email', validators=[DataRequired(), EmailValidator()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Create Account')

class ExpenseForm(FlaskForm):
    date = DateField('Date', validators=[DataRequired()], default=date.today)
    amount = FloatField('Amount (R)', validators=[DataRequired(), NumberRange(min=0.01)])
    description = StringField('Description', validators=[DataRequired(), Length(max=200)])
    submit = SubmitField('Add Expense')

class IncomeForm(FlaskForm):
    monthly_income = FloatField('Monthly Income (R)', validators=[DataRequired(), NumberRange(min=0)])
    submit = SubmitField('Update Income')