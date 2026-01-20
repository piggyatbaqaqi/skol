"""Forms for contact and feedback."""
from django import forms


class ContactForm(forms.Form):
    """Form for general contact inquiries."""
    email = forms.EmailField(
        required=False,
        label='Your Email (optional)',
        help_text='Provide your email if you would like a reply.',
        widget=forms.EmailInput(attrs={
            'class': 'form-input',
            'placeholder': 'your.email@example.com'
        })
    )
    subject = forms.CharField(
        max_length=200,
        label='Subject',
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'What is this regarding?'
        })
    )
    message = forms.CharField(
        label='Message',
        widget=forms.Textarea(attrs={
            'class': 'form-input',
            'rows': 6,
            'placeholder': 'Enter your message here...'
        })
    )


class FeedbackForm(forms.Form):
    """Form for website feedback/bug reports."""
    email = forms.EmailField(
        required=False,
        label='Your Email (optional)',
        help_text='Provide your email if you would like a reply.',
        widget=forms.EmailInput(attrs={
            'class': 'form-input',
            'placeholder': 'your.email@example.com'
        })
    )
    page_url = forms.CharField(
        required=False,
        label='Page URL',
        help_text='The page you were on when you encountered the issue.',
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'readonly': 'readonly'
        })
    )
    feedback_type = forms.ChoiceField(
        choices=[
            ('bug', 'Bug Report'),
            ('feature', 'Feature Request'),
            ('usability', 'Usability Issue'),
            ('other', 'Other'),
        ],
        label='Feedback Type',
        widget=forms.Select(attrs={'class': 'form-input'})
    )
    message = forms.CharField(
        label='Description',
        widget=forms.Textarea(attrs={
            'class': 'form-input',
            'rows': 6,
            'placeholder': 'Please describe the issue or suggestion...'
        })
    )
