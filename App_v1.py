# Library import
import streamlit as st
import joblib
import pandas as pd
import requests
from urllib.parse import urlparse, parse_qs
from io import BytesIO

# Model retrieval.
def load_model(url):
        response = requests.get(url)
        response.raise_for_status() 
        model_file = BytesIO(response.content)
        model = joblib.load(model_file)
        return model

model_url = 'https://github.com/A-NeuGlia/Phishing_Detection_Eng/blob/master/Algorithm%20V2%20(Original)/phishing_model.pkl'
model = load_model(model_url)
if model:
    print("Model successfully loaded.")
else:
    print("Error while loading the model.")

# Feature extraction 
def extract_features(url):
    parsed_url = urlparse(url)
    domain_parts = parsed_url.netloc.split('.')
    subdomain_parts = domain_parts[:-2] if len(domain_parts) > 2 else []
    path = parsed_url.path if parsed_url.path else "/"
    query_components = parse_qs(parsed_url.query)

    features = {
        'NumDots': url.count('.'),
        'SubdomainLevel': len(subdomain_parts),
        'PathLevel': path.count('/') - 1,
        'UrlLength': len(url),
        'NumDash': url.count('-'),
        'NumDashInHostname': parsed_url.netloc.count('-'),
        'AtSymbol': '@' in url,
        'TildeSymbol': '~' in url,
        'NumUnderscore': url.count('_'),
        'NumPercent': url.count('%'),
        'NumQueryComponents': len(query_components),
        'NumAmpersand': url.count('&'),
        'NumHash': url.count('#'),
        'NumNumericChars': sum(c.isdigit() for c in url),
        'NoHttps': parsed_url.scheme != 'https',
        'RandomString': any(c.isalnum() for c in url),
        'IpAddress': parsed_url.netloc.replace('.', '').isdigit(),
        'DomainInSubdomains': any(part in subdomain_parts for part in domain_parts),
        'DomainInPaths': any(part in path for part in domain_parts),
        'HttpsInHostname': 'https' in parsed_url.netloc,
        'HostnameLength': len(parsed_url.netloc),
        'PathLength': len(path),
        'QueryLength': len(parsed_url.query),
        'DoubleSlashInPath': '//' in parsed_url.path,
        'NumSensitiveWords': sum(1 for word in ['login', 'secure', 'bank'] if word in url.lower()),
        'EmbeddedBrandName': any(brand in url.lower() for brand in ['microsoft', 'google']),
    }
    return features

#Model application 
def check_url(url):
    features = extract_features(url)
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)
    return "corrupted" if prediction[0] == 1 else "legitimate"


#User interface
st.title("Phishing Attempt Analyser")
url = st.text_input("Enter the suspicious url : ")

if st.button("Control of the legitimacy... : "):
    result = check_url(url)
    st.write(f"The url is  {result}.")
