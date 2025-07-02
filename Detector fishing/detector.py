import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import re
from urllib.parse import urlparse
import joblib
import tldextract
from bs4 import BeautifulSoup
import requests

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"

# Descargar el archivo ARFF

local_filename, _ = urllib.request.urlretrieve(url)

# Leer el archivo ARFF
data, meta = arff.loadarff(local_filename)
df = pd.DataFrame(data)
print(df['Result'].value_counts())

# Convertir bytes a strings (para columnas categóricas)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.decode('utf-8')

# Paso crítico: Inspeccionar valores ORIGINALES en 'Result'
print("\nValores únicos ORIGINALES en 'Result':", df['Result'].unique())

# Opción CORRECTA (mapear strings)
df['Result'] = df['Result'].map({'-1': 0, '1': 1})

# Verificar resultado
print("\nValores después del mapeo:", df['Result'].unique())
print("Valores nulos en 'Result':", df['Result'].isna().sum())

# Continuar con el resto del código (esto ya no entrará al 'else')
if not df.empty:
    X = df.drop('Result', axis=1)
    y = df['Result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("\nModelo entrenado exitosamente!")

#Paso 6: Entrenamiento y evaluacion del modelo
#Entrenar el modelo, pero agregamos evaluacion
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#Predecir en el conjunto de prueba
y_pred = model.predict(X_test)

#Metricas de evaluacion
print("\n== Metricas de Evaluación ==")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))


#Paso 7: Optimizacio de Hiperparametros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Mejores parámetros:", grid_search.best_params_)
best_model = grid_search.best_estimator_

#Paso 8: Guardar el modelo para producción
joblib.dump(best_model, 'modelo_phishing.pkl')
print("\n✅ Modelo guardado correctamente como 'modelo_phishing.pkl'")
print(f"Mejores parámetros usados: {grid_search.best_params_}")


#Paso 9: Crear verificador de Links
modelo = joblib.load('modelo_phishing.pkl')

def extraer_caracteristicas_url(url):
    caracteristicas = []

    try:
        ext = tldextract.extract(url)
        dominio_principal = f"{ext.domain}.{ext.suffix}"
        subdominios = ext.subdomain.split('.') if ext.subdomain else []
        parsed = urlparse(url)
        
        # 1. having_IP_Address
        caracteristicas.append(1 if re.match(r'\d+\.\d+\.\d+\.\d+', dominio_principal) else -1)
        # 2. URL_Length
        caracteristicas.append(len(url))
        
        # 3. Shortining_Service
        caracteristicas.append(1 if any(s in url.lower() for s in ['bit.ly', 'goo.gl', 'tinyurl', 'ow.ly']) else -1)
        
        # 4. having_At_Symbol
        caracteristicas.append(1 if '@' in url and not url.split('@')[1].endswith(dominio_principal) else -1)

        # 5. double_slash_redirecting
        caracteristicas.append(1 if '//' in parsed.path else -1)
        
        # 6. Prefix_Suffix
        caracteristicas.append(1 if '-' in dominio_principal or any(part.isdigit() for part in subdominios) else -1)
        
        # 7. having_Sub_Domain
        caracteristicas.append(1 if len(subdominios) > 2 else (-1 if not subdominios else 0))
        
        # 8. SSLfinal_State (simplificado)
        caracteristicas.append(1 if parsed.scheme == 'https' else -1)
        
        # 10. port
        caracteristicas.append(1 if parsed.port not in [80, 443, None] else -1)

        # 11. HTTPS_token
        caracteristicas.append(1 if 'https' in dominio_principal.lower() else -1)

        # 12. Request_URL
        caracteristicas.append(1 if 'request' in url else -1)

        try:
            # 13. Request_URL: % de recursos externos (requiere analizar HTML)
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            recursos = [tag['src'] or tag['href'] for tag in soup.find_all(['img', 'script', 'link'])]
            externos = sum(1 for r in recursos if r and not urlparse(r).netloc.endswith(dominio_principal))
            caracteristicas.append(1 if externos/len(recursos) > 0.5 else -1) if recursos else -1
            
            # 14. URL_of_Anchor: % de anclas a dominios externos
            anchors = [a['href'] for a in soup.find_all('a', href=True)]
            externos_anchors = sum(1 for a in anchors if a and not a.startswith('#') and not urlparse(a).netloc.endswith(dominio_principal))
            caracteristicas.append(1 if externos_anchors/len(anchors) > 0.5 else -1) if anchors else -1
        
            # 15. Links_in_tags: % de links en tags <meta>, <script>, etc.
            links_tags = len(soup.find_all(['meta', 'script', 'style']))
            caracteristicas.append(1 if links_tags/len(soup.find_all()) > 0.5 else -1)
            
            # 16. SFH: Formularios que envían a dominios externos
            forms = soup.find_all('form')
            externos_forms = sum(1 for f in forms if f.get('action') and not urlparse(f['action']).netloc.endswith(dominio_principal))
            caracteristicas.append(1 if externos_forms/len(forms) > 0.5 else -1) if forms else -1
            
            # 17. Submitting_to_email: Formularios con mailto:
            mailto_forms = sum(1 for f in forms if f.get('action', '').startswith('mailto:'))
            caracteristicas.append(1 if mailto_forms > 0 else -1)
            
            # 18. Abnormal_URL: Dominio no coincide con URL visible
            caracteristicas.append(1 if any(palabra in dominio_principal.lower() for palabra in 
                                    ['login', 'verify', 'secure', 'update', 'account']) else -1)
            # 19. Redirect: Redirecciones HTTP (códigos 3XX)
            historia_redirect = [r.url for r in response.history]
            caracteristicas.append(1 if len(historia_redirect) > 1 else -1)
            
            # 20. on_mouseover: Uso de JavaScript en onmouseover
            mouseover_scripts = sum(1 for s in soup.find_all('script') if 'onmouseover' in s.text)
            caracteristicas.append(1 if mouseover_scripts > 0 else -1)

            # 21. RightClick: Deshabilita clic derecho
            right_click_disabled = (
                'event.button==2' in str(soup) or 
                'contextmenu' in str(soup)
            )
            caracteristicas.append(1 if right_click_disabled else -1)
            
            # 22. popUpWidnow: Ventanas emergentes
            popups = (
                soup.find_all('script', string=re.compile(r'window\.open|alert\(')) or
                soup.find_all('div', {'class': re.compile(r'popup|modal')})
            )
            caracteristicas.append(1 if popups else -1)

            # 23. Iframe: Uso de iframes
            iframes = soup.find_all('iframe')
            caracteristicas.append(1 if iframes else -1)
            
            # 24. age_of_domain: Edad del dominio (simplificado)
            domain_age = -1  # Valor por defecto
            if 'whois' in str(soup).lower():
                domain_age = 1  # Si menciona WHOIS, asumimos dominio antiguo
            caracteristicas.append(domain_age)
            
            # 25. DNSRecord: DNS válido (verificación básica)
            dns_valid = (
                'dns' in str(soup).lower() or 
                'cloudflare' in str(soup).lower()
            )
            caracteristicas.append(1 if dns_valid else -1)
            
            # 26. web_traffic: Tráfico (simulación con Alexa rank)
            traffic = -1
            if 'alexa' in str(soup).lower():
                traffic = 1  # Si menciona Alexa, asumimos tráfico alto
            caracteristicas.append(traffic)
            
            # 27. Page_Rank: Pagerank simulado
            pagerank = -1
            if 'google' in str(soup).lower():
                pagerank = 1  # Si menciona Google, asumimos buen PageRank
            caracteristicas.append(pagerank)

            # 28. Google_Index: Indexado en Google
            google_index = 1 if 'google-analytics' in str(soup) else -1
            caracteristicas.append(google_index)
            
            # 29. Links_pointing_to_page: Backlinks (simulación)
            backlinks = 1 if 'backlink' in str(soup).lower() else -1
            caracteristicas.append(backlinks)
            
            # 30. Statistical_report: Reportes de seguridad
            security_report = 1 if 'security' in str(soup).lower() else -1
            caracteristicas.append(security_report)
            
        except Exception as e:
            # Si falla el request, llenar con -1
            caracteristicas.extend([-1]*10)
    except Exception as e:
        print(f"Error procesando URL {url}: {str(e)}")
        # Si falla la extracción, llenar con -1
        return [-1] * 30
    
    # Completar con ceros (temporal)
    while len(caracteristicas) < 30:
        caracteristicas.append(-1)
    
    return caracteristicas[:30]

# 1. Definir los nombres de las columnas (deben coincidir con el orden de las características)
columnas_originales = [
    'having_IP_Address', 'URL_Length', 'Shortining_Service',
    'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix', 
    'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length',
    'Favicon', 'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor',
    'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
    'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe',
    'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank',
    'Google_Index', 'Links_pointing_to_page', 'Statistical_report'
]
# 2. Función de verificación mejorada
def verificar_phishing(url):
    try:
        # Extraer características
        features = extraer_caracteristicas_url(url)
        
        # Debug: Mostrar características clave
        print("\n🔍 Debug para URL:", url)
        print("Características clave:")
        print("- having_At_Symbol:", features[3])  # Posición 3
        print("- Prefix_Suffix:", features[5])     # Posición 5 
        print("- Abnormal_URL:", features[18])     # Posición 18
        
        # Convertir a DataFrame con nombres de columnas
        features_df = pd.DataFrame([features], columns=columnas_originales)

        # Predecir
        prediccion = modelo.predict(features_df)[0]
        return "⚠️ PHISHING" if prediccion == 1 else "✅ LEGÍTIMO"
    except Exception as e:
        return f"Error: {str(e)}"

# 3. Pruebas con casos más evidentes
test_urls = [
    "https://www.google.com",                      # Legítima
    "https://www.paypal.com",                      # Legítima 
    "http://phishing-example.com/login@paypal.com",# Phishing (tiene @)
    "https://www.paypal.com.security-update.com",  # Phishing (subdominio sospechoso)
    "http://secure-facebook.com",                  # Phishing (dominio falso)
    "https://facebook.com.login.user.verify.com"   # Phishing (trampa)
]

print("\n=== Resultados de Prueba ===")
for url in test_urls:
    resultado = verificar_phishing(url)
    print(f"{url.ljust(50)} -> {resultado}")