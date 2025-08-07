import streamlit_authenticator as stauth
print(stauth.Hasher().hash('admin123'))
print(stauth.Hasher().hash('user123'))