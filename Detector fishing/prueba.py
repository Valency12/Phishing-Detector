import pandas as pd
import urllib.request
from scipy.io import arff
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import re
from urllib.parse import urlparse
import tldextract
from bs4 import BeautifulSoup
import requests

# 1. Carga y preparación de datos
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
local_filename, _ = urllib.request.urlretrieve(url)
data, meta = arff.loadarff(local_filename)
df = pd.DataFrame(data)

# Convertir bytes a strings
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.decode('utf-8')

# Mapear resultados
df['Result'] = df['Result'].map({'-1': 0, '1': 1})

# 2. División de datos
X = df.drop('Result', axis=1)
y = df['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Entrenamiento del modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Evaluación
y_pred = model.predict(X_test)
print("\nMetricas de Evaluación:")
print(classification_report(y_test, y_pred))

# 5. Optimización (opcional)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 6. Guardar modelo
joblib.dump(best_model, 'modelo_phishing.pkl')
modelo = joblib.load('modelo_phishing.pkl')

# 7. Función mejorada de extracción de características
def extraer_caracteristicas_url(url):
    try:
        ext = tldextract.extract(url)
        dominio = f"{ext.domain}.{ext.suffix}"
        subdominio = ext.subdomain
        parsed = urlparse(url)
        
        # Características básicas de URL
        features = [
            1 if re.match(r'\d+\.\d+\.\d+\.\d+', dominio) else -1,  # having_IP_Address
            len(url),  # URL_Length
            1 if any(s in url for s in ['bit.ly', 'goo.gl']) else -1,  # Shortining_Service
            1 if '@' in url and not url.split('@')[1].endswith(dominio) else -1,  # having_At_Symbol
            1 if '//' in parsed.path else -1,  # double_slash_redirecting
            1 if '-' in dominio else -1,  # Prefix_Suffix
            1 if subdominio.count('.') >= 2 else -1,  # having_Sub_Domain
            1 if parsed.scheme == 'https' else -1,  # SSLfinal_State
            1,  # Domain_registeration_length (asumir válido)
            -1,  # Favicon (requeriría scrapping)
            1 if parsed.port not in [80, 443, None] else -1,  # port
            1 if 'https' in dominio else -1  # HTTPS_token
        ]
        
        # Scrapping para características avanzadas
        try:
            response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Añadir características de HTML
            features.extend([
                1 if len(response.history) > 1 else -1,  # Redirect
                1 if soup.find('iframe') else -1,  # Iframe
                1 if 'event.button==2' in str(soup) else -1,  # RightClick
                1 if 'window.open' in str(soup) else -1,  # popUpWidnow
                -1, -1, -1, -1, -1, -1  # Placeholders para el resto
            ])
        except:
            features.extend([-1]*18)
            
        return features[:30]
        
    except Exception as e:
        print(f"Error procesando URL {url}: {str(e)}")
        return [-1]*30

# 8. Función de verificación optimizada
columnas_originales = X_train.columns.tolist()  # <- Añade esto si ejecutas por partes

def verificar_phishing(url):
    try:
        features = extraer_caracteristicas_url(url)
        features_df = pd.DataFrame([features], columns=columnas_originales)
        prediccion = modelo.predict(features_df)[0]
        return "⚠️ PHISHING" if prediccion == 1 else "✅ LEGÍTIMO"
    except Exception as e:
        return f"Error: {str(e)}"

# 9. Pruebas
test_urls = [
    "https://www.google.com",
    "http://phishing-example.com/login@fake",
    "https://www.paypal.com.security-update.com",
    "http://secure-facebook.com",
    "https://www.youtube.com"
]

print("\nResultados de prueba:")
for url in test_urls:
    print(f"{url} -> {verificar_phishing(url)}")