import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                            accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report)
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ==================== CLASES POO ====================

class DataLoader:
    """Clase para cargar y gestionar datos"""
    
    def __init__(self):
        self.data = None
        self.filename = None
    
    def load_csv(self, uploaded_file):
        """Carga un archivo CSV"""
        try:
            self.data = pd.read_csv(uploaded_file)
            self.filename = uploaded_file.name
            return True
        except Exception as e:
            st.error(f"Error al cargar el archivo: {str(e)}")
            return False
    
    def get_info(self):
        """Retorna información del dataset"""
        if self.data is not None:
            return {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'dtypes': self.data.dtypes.to_dict(),
                'missing': self.data.isnull().sum().to_dict(),
                'head': self.data.head(),
                'describe': self.data.describe()
            }
        return None


class DataPreprocessor:
    """Clase para preprocesamiento y transformación de datos"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def prepare_data(self, data, target_column, feature_columns):
        """Prepara los datos separando X e y"""
        X = data[feature_columns].copy()
        y = data[target_column].copy()
        return X, y
    
    def handle_missing_values(self, X):
        """Maneja valores faltantes"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        
        return X
    
    def encode_categorical(self, X):
        """Codifica variables categóricas"""
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        return X
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Divide los datos en train y test"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self, X_train, X_test):
        """Escala las características"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled


class ModelTrainer:
    """Clase para entrenamiento y ajuste del modelo"""
    
    def __init__(self, algorithm_type):
        self.algorithm_type = algorithm_type
        self.model = None
        self.is_fitted = False
    
    def create_model(self, **kwargs):
        """Crea el modelo según el algoritmo seleccionado"""
        if self.algorithm_type == "linear":
            self.model = LinearRegression()
        elif self.algorithm_type == "logistic":
            self.model = LogisticRegression(max_iter=1000, **kwargs)
        elif self.algorithm_type == "ridge":
            alpha = kwargs.get('alpha', 1.0)
            self.model = Ridge(alpha=alpha)
        elif self.algorithm_type == "lasso":
            alpha = kwargs.get('alpha', 1.0)
            self.model = Lasso(alpha=alpha)
        elif self.algorithm_type == "cart":
            max_depth = kwargs.get('max_depth', 5)
            self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        elif self.algorithm_type == "knn":
            n_neighbors = kwargs.get('n_neighbors', 5)
            self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
        elif self.algorithm_type == "neural":
            hidden_layers = kwargs.get('hidden_layers', (100, 50))
            self.model = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=1000, random_state=42)
        
        return self.model
    
    def train(self, X_train, y_train):
        """Entrena el modelo"""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
    
    def cross_validate(self, X, y, cv=5):
        """Realiza validación cruzada"""
        if self.model is not None:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_squared_error')
            return -scores.mean(), scores.std()
        return None, None


class ModelEvaluator:
    """Clase para evaluación y métricas del modelo"""
    
    def __init__(self, model, problem_type="regression"):
        self.model = model
        self.problem_type = problem_type
        self.metrics = {}
    
    def predict(self, X):
        """Realiza predicciones"""
        return self.model.predict(X)
    
    def evaluate_regression(self, y_true, y_pred):
        """Evalúa modelo de regresión"""
        self.metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R²': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        }
        return self.metrics
    
    def evaluate_classification(self, y_true, y_pred):
        """Evalúa modelo de clasificación"""
        self.metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        self.metrics['Confusion Matrix'] = confusion_matrix(y_true, y_pred)
        return self.metrics


class MLPipeline:
    """Pipeline completo de Machine Learning"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.trainer = None
        self.evaluator = None
    
    def load_data(self, uploaded_file):
        """Carga los datos"""
        return self.data_loader.load_csv(uploaded_file)
    
    def prepare_pipeline(self, target_column, feature_columns, test_size, random_state):
        """Prepara el pipeline de datos"""
        X, y = self.preprocessor.prepare_data(
            self.data_loader.data, target_column, feature_columns
        )
        X = self.preprocessor.handle_missing_values(X)
        X = self.preprocessor.encode_categorical(X)
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(
            X, y, test_size, random_state
        )
        X_train_scaled, X_test_scaled = self.preprocessor.scale_features(
            X_train, X_test
        )
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, algorithm_type, X_train, y_train, **kwargs):
        """Entrena el modelo"""
        self.trainer = ModelTrainer(algorithm_type)
        self.trainer.create_model(**kwargs)
        self.trainer.train(X_train, y_train)
        return self.trainer.model
    
    def evaluate_model(self, model, X_test, y_test, problem_type):
        """Evalúa el modelo"""
        self.evaluator = ModelEvaluator(model, problem_type)
        y_pred = self.evaluator.predict(X_test)
        
        if problem_type == "regression":
            metrics = self.evaluator.evaluate_regression(y_test, y_pred)
        else:
            metrics = self.evaluator.evaluate_classification(y_test, y_pred)
        
        return metrics, y_pred


# ==================== INTERFAZ STREAMLIT ====================

st.set_page_config(
    page_title="ML System Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado mejorado
st.markdown("""
<style>
    /* Fondo principal */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tarjetas de métricas */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin: 10px 0;
    }
    
    /* Botones de algoritmos */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 70px;
        font-size: 16px;
        font-weight: 600;
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255,255,255,0.2);
        color: white;
        transition: all 0.3s ease;
        margin: 5px 0;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, rgba(255,255,255,0.25) 0%, rgba(255,255,255,0.15) 100%);
        border: 2px solid rgba(255,255,255,0.4);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
    }
    
    /* Títulos */
    h1, h2, h3 {
        color: white !important;
        font-weight: 700 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: white;
        background: transparent;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.1) 100%);
    }
    
    /* Contenedores de expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        color: white !important;
        font-weight: 600;
    }
    
    /* Dataframes */
    .stDataFrame {
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
    }
    
    /* Métricas de Streamlit */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: white;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.8);
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header con animación
st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); 
    backdrop-filter: blur(10px); border-radius: 20px; margin-bottom: 30px; border: 2px solid rgba(255,255,255,0.2);'>
        <h1 style='font-size: 3em; margin: 0; color: white;'>🤖 ML System Pro</h1>
        <p style='font-size: 1.3em; color: rgba(255,255,255,0.9); margin: 10px 0;'>
            Sistema Profesional de Machine Learning con POO
        </p>
    </div>
""", unsafe_allow_html=True)

# Inicializar pipeline
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = MLPipeline()
if 'step' not in st.session_state:
    st.session_state.step = 1

# Sidebar mejorado
with st.sidebar:
    st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(102,126,234,0.8) 0%, rgba(118,75,162,0.8) 100%); 
        padding: 20px; border-radius: 15px; margin-bottom: 20px; text-align: center;'>
            <h2 style='margin: 0; color: white;'>📊 Panel de Control</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Paso 1: Carga de datos
    st.markdown("### 📁 Paso 1: Cargar Datos")
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.session_state.pipeline.load_data(uploaded_file):
            st.session_state.step = 2
            data_info = st.session_state.pipeline.data_loader.get_info()
            
            st.success(f"✅ {data_info['shape'][0]} filas × {data_info['shape'][1]} columnas")
            
            # Paso 2: Configuración
            st.markdown("### ⚙️ Paso 2: Configuración")
            
            problem_type = st.radio(
                "Tipo de problema:",
                ["Regresión", "Clasificación"],
                label_visibility="visible"
            )
            
            target_column = st.selectbox(
                "🎯 Variable objetivo (Y):",
                data_info['columns']
            )
            
            available_features = [col for col in data_info['columns'] if col != target_column]
            feature_columns = st.multiselect(
                "📊 Variables predictoras (X):",
                available_features,
                default=available_features[:min(5, len(available_features))]
            )
            
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("% Test", 10, 40, 20) / 100
            with col2:
                random_state = st.number_input("Semilla", 0, 100, 42)
            
            if len(feature_columns) > 0:
                st.session_state.step = 3

# Contenido principal
if st.session_state.step == 1:
    # Pantalla de bienvenida
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%); 
            backdrop-filter: blur(10px); padding: 40px; border-radius: 20px; text-align: center; 
            border: 2px solid rgba(255,255,255,0.3); margin-top: 50px;'>
                <h2 style='color: white; margin-bottom: 20px;'>👋 ¡Bienvenido!</h2>
                <p style='color: rgba(255,255,255,0.9); font-size: 1.1em; line-height: 1.8;'>
                    <strong>Sistema completo de Machine Learning</strong> con arquitectura POO<br><br>
                    📁 Sube tu dataset CSV<br>
                    🎯 Selecciona algoritmo<br>
                    📊 Obtén resultados profesionales<br><br>
                    <strong>👈 Comienza subiendo un archivo CSV</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Algoritmos disponibles
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); 
            backdrop-filter: blur(10px); padding: 30px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.2);'>
                <h3 style='color: white; text-align: center; margin-bottom: 20px;'>🎯 Algoritmos Disponibles</h3>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px; color: white;'>
                    <div style='background: rgba(59, 130, 246, 0.3); padding: 15px; border-radius: 10px; text-align: center;'>
                        <strong>📈 Regresión Lineal</strong>
                    </div>
                    <div style='background: rgba(139, 92, 246, 0.3); padding: 15px; border-radius: 10px; text-align: center;'>
                        <strong>📊 Ridge & Lasso</strong>
                    </div>
                    <div style='background: rgba(34, 197, 94, 0.3); padding: 15px; border-radius: 10px; text-align: center;'>
                        <strong>🌿 Reg. Logística</strong>
                    </div>
                    <div style='background: rgba(251, 146, 60, 0.3); padding: 15px; border-radius: 10px; text-align: center;'>
                        <strong>🌳 Árbol CART</strong>
                    </div>
                    <div style='background: rgba(236, 72, 153, 0.3); padding: 15px; border-radius: 10px; text-align: center;'>
                        <strong>🔗 K-NN</strong>
                    </div>
                    <div style='background: rgba(99, 102, 241, 0.3); padding: 15px; border-radius: 10px; text-align: center;'>
                        <strong>🧠 Red Neuronal</strong>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

elif st.session_state.step >= 2:
    # Mostrar información del dataset
    data_info = st.session_state.pipeline.data_loader.get_info()
    
    # Tabs para exploración de datos
    tab1, tab2, tab3 = st.tabs(["📊 Vista General", "📈 Exploración", "🎯 Seleccionar Algoritmo"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📦 Total Filas", data_info['shape'][0])
        with col2:
            st.metric("📊 Total Columnas", data_info['shape'][1])
        with col3:
            st.metric("🔢 Variables Numéricas", len(data_info['describe'].columns))
        with col4:
            missing_total = sum(data_info['missing'].values())
            st.metric("⚠️ Valores Faltantes", missing_total)
        
        st.markdown("#### 🔍 Vista Previa del Dataset")
        st.dataframe(data_info['head'], use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Estadísticas Descriptivas")
            st.dataframe(data_info['describe'], use_container_width=True)
        
        with col2:
            st.markdown("#### ⚠️ Valores Faltantes por Columna")
            missing_df = pd.DataFrame({
                'Columna': list(data_info['missing'].keys()),
                'Faltantes': list(data_info['missing'].values())
            })
            missing_df = missing_df[missing_df['Faltantes'] > 0]
            if len(missing_df) > 0:
                fig = px.bar(missing_df, x='Columna', y='Faltantes', 
                           title="Valores Faltantes",
                           color='Faltantes',
                           color_continuous_scale='Reds')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No hay valores faltantes")
    
    with tab2:
        st.markdown("#### 📊 Distribución de Variables")
        
        numeric_cols = data_info['describe'].columns.tolist()
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Selecciona una variable para visualizar:", numeric_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(st.session_state.pipeline.data_loader.data, 
                                 x=selected_col, 
                                 title=f"Histograma: {selected_col}",
                                 color_discrete_sequence=['#667eea'])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(st.session_state.pipeline.data_loader.data, 
                           y=selected_col, 
                           title=f"Box Plot: {selected_col}",
                           color_discrete_sequence=['#764ba2'])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlación
        st.markdown("#### 🔗 Matriz de Correlación")
        corr_matrix = st.session_state.pipeline.data_loader.data[numeric_cols].corr()
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       color_continuous_scale='RdBu_r',
                       title="Matriz de Correlación")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if st.session_state.step >= 3:
            st.markdown("### 🎯 Selecciona un Algoritmo de Machine Learning")
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Grid de algoritmos
            col1, col2, col3 = st.columns(3)
            
            algorithms = [
                ("linear", "📈 Regresión Lineal", "Predicción lineal simple y efectiva", col1),
                ("ridge", "📊 Regresión Ridge", "Regularización L2 para evitar overfitting", col2),
                ("lasso", "📉 Regresión Lasso", "Regularización L1 con selección de features", col3),
                ("logistic", "🌿 Regresión Logística", "Clasificación binaria/multiclase", col1),
                ("cart", "🌳 Árbol CART", "Árbol de decisión interpretable", col2),
                ("knn", "🔗 K-NN Regresión", "Predicción por vecinos cercanos", col3),
                ("neural", "🧠 Red Neuronal", "Perceptrón multicapa profundo", col1),
            ]
            
            for algo_id, name, desc, col in algorithms:
                with col:
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
                        backdrop-filter: blur(10px); padding: 15px; border-radius: 12px; 
                        border: 2px solid rgba(255,255,255,0.2); margin-bottom: 15px; height: 140px;'>
                            <h4 style='color: white; margin: 0 0 10px 0;'>{name}</h4>
                            <p style='color: rgba(255,255,255,0.8); font-size: 0.9em; margin: 0;'>{desc}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"Ejecutar {name}", key=f"btn_{algo_id}"):
                        st.session_state.algorithm = algo_id
                        st.rerun()
            
            # Ejecutar modelo si se seleccionó
            if 'algorithm' in st.session_state:
                st.markdown("---")
                st.markdown(f"### 🚀 Entrenando Modelo: {st.session_state.algorithm.upper()}")
                
                with st.spinner("⏳ Preparando datos y entrenando modelo..."):
                    try:
                        # Preparar pipeline
                        X_train, X_test, y_train, y_test = st.session_state.pipeline.prepare_pipeline(
                            target_column, feature_columns, test_size, random_state
                        )
                        
                        # Hiperparámetros
                        params = {}
                        if st.session_state.algorithm in ["ridge", "lasso"]:
                            with st.sidebar:
                                st.markdown("#### 🎛️ Hiperparámetros")
                                params['alpha'] = st.slider("Alpha (Regularización)", 0.1, 10.0, 1.0, 0.1)
                        elif st.session_state.algorithm == "cart":
                            with st.sidebar:
                                st.markdown("#### 🎛️ Hiperparámetros")
                                params['max_depth'] = st.slider("Profundidad Máxima", 1, 20, 5)
                        elif st.session_state.algorithm == "knn":
                            with st.sidebar:
                                st.markdown("#### 🎛️ Hiperparámetros")
                                params['n_neighbors'] = st.slider("K Vecinos", 1, 20, 5)
                        
                        # Entrenar
                        model = st.session_state.pipeline.train_model(
                            st.session_state.algorithm, X_train, y_train, **params
                        )
                        
                        # Evaluar
                        prob_type = "regression" if problem_type == "Regresión" else "classification"
                        metrics, y_pred = st.session_state.pipeline.evaluate_model(
                            model, X_test, y_test, prob_type
                        )
                        
                        st.success("✅ ¡Modelo entrenado exitosamente!")
                        
                        # Métricas en cards
                        st.markdown("#### 📊 Métricas de Evaluación")
                        
                        metric_cols = st.columns(len([k for k in metrics.keys() if k != 'Confusion Matrix']))
                        for i, (key, value) in enumerate(metrics.items()):
                            if key != 'Confusion Matrix':
                                with metric_cols[i]:
                                    if isinstance(value, float):
                                        st.metric(key, f"{value:.4f}")
                                    else:
                                        st.metric(key, value)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Visualizaciones
                        st.markdown("#### 📈 Visualizaciones del Modelo")
                        
                        # Visualización especial para árbol de decisión
                        if st.session_state.algorithm == "cart":
                            st.markdown("#### 🌳 Estructura del Árbol de Decisión")
                            
                            fig, ax = plt.subplots(figsize=(20, 10))
                            plot_tree(model, 
                                     feature_names=feature_columns,
                                     filled=True,
                                     rounded=True,
                                     fontsize=10,
                                     ax=ax)
                            ax.set_facecolor('none')
                            fig.patch.set_alpha(0.0)
                            
                            # Cambiar colores para fondo oscuro
                            for text in ax.texts:
                                text.set_color('white')
                            
                            st.pyplot(fig)
                            plt.close()
                            
                            # Información del árbol
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🌿 Profundidad del Árbol", model.get_depth())
                            with col2:
                                st.metric("🍃 Número de Hojas", model.get_n_leaves())
                            with col3:
                                n_nodes = model.tree_.node_count
                                st.metric("🔵 Total de Nodos", n_nodes)
                            
                            # Importancia de características
                            st.markdown("#### 📊 Importancia de Características")
                            feature_importance = pd.DataFrame({
                                'Feature': feature_columns,
                                'Importancia': model.feature_importances_
                            }).sort_values('Importancia', ascending=False)
                            
                            fig = px.bar(feature_importance, 
                                        x='Importancia', 
                                        y='Feature',
                                        orientation='h',
                                        title="Importancia de cada Variable",
                                        color='Importancia',
                                        color_continuous_scale='Viridis')
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white')
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                        
                        if prob_type == "regression":
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Gráfico de predicciones vs reales
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=y_test, 
                                    y=y_pred,
                                    mode='markers',
                                    marker=dict(
                                        size=8,
                                        color=y_pred,
                                        colorscale='Viridis',
                                        showscale=True,
                                        line=dict(width=1, color='white')
                                    ),
                                    name='Predicciones'
                                ))
                                
                                # Línea de referencia
                                min_val = min(y_test.min(), y_pred.min())
                                max_val = max(y_test.max(), y_pred.max())
                                fig.add_trace(go.Scatter(
                                    x=[min_val, max_val],
                                    y=[min_val, max_val],
                                    mode='lines',
                                    line=dict(color='red', width=2, dash='dash'),
                                    name='Línea Ideal'
                                ))
                                
                                fig.update_layout(
                                    title="Predicciones vs Valores Reales",
                                    xaxis_title="Valores Reales",
                                    yaxis_title="Predicciones",
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='white'),
                                    showlegend=True
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Gráfico de residuos
                                residuals = y_test - y_pred
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=y_pred,
                                    y=residuals,
                                    mode='markers',
                                    marker=dict(
                                        size=8,
                                        color=residuals,
                                        colorscale='RdBu',
                                        showscale=True,
                                        line=dict(width=1, color='white')
                                    ),
                                    name='Residuos'
                                ))
                                
                                fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
                                
                                fig.update_layout(
                                    title="Gráfico de Residuos",
                                    xaxis_title="Predicciones",
                                    yaxis_title="Residuos",
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='white'),
                                    showlegend=True
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Distribución de errores
                            st.markdown("#### 📊 Distribución de Errores")
                            fig = px.histogram(
                                x=residuals,
                                nbins=30,
                                title="Distribución de Residuos",
                                labels={'x': 'Residuos', 'y': 'Frecuencia'},
                                color_discrete_sequence=['#667eea']
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white')
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        else:  # Clasificación
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Matriz de confusión
                                cm = metrics['Confusion Matrix']
                                fig = go.Figure(data=go.Heatmap(
                                    z=cm,
                                    text=cm,
                                    texttemplate='%{text}',
                                    colorscale='Blues',
                                    showscale=True
                                ))
                                
                                fig.update_layout(
                                    title="Matriz de Confusión",
                                    xaxis_title="Predicción",
                                    yaxis_title="Real",
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='white')
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Métricas en barras
                                metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                                metric_values = [metrics[m] for m in metric_names]
                                
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=metric_names,
                                        y=metric_values,
                                        marker=dict(
                                            color=['#667eea', '#764ba2', '#f093fb', '#4facfe'],
                                            line=dict(color='white', width=2)
                                        ),
                                        text=[f"{v:.3f}" for v in metric_values],
                                        textposition='auto'
                                    )
                                ])
                                
                                fig.update_layout(
                                    title="Métricas de Clasificación",
                                    yaxis_title="Valor",
                                    yaxis_range=[0, 1],
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='white')
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Información del modelo
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("#### ℹ️ Información del Modelo")
                        
                        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                        
                        with info_col1:
                            st.markdown(f"""
                                <div class='metric-card'>
                                    <h4 style='color: #667eea; margin: 0;'>🤖 Algoritmo</h4>
                                    <p style='font-size: 1.2em; font-weight: bold; margin: 10px 0 0 0;'>
                                        {st.session_state.algorithm.upper()}
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with info_col2:
                            st.markdown(f"""
                                <div class='metric-card'>
                                    <h4 style='color: #764ba2; margin: 0;'>📊 Features</h4>
                                    <p style='font-size: 1.2em; font-weight: bold; margin: 10px 0 0 0;'>
                                        {len(feature_columns)}
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with info_col3:
                            st.markdown(f"""
                                <div class='metric-card'>
                                    <h4 style='color: #f093fb; margin: 0;'>🎓 Train</h4>
                                    <p style='font-size: 1.2em; font-weight: bold; margin: 10px 0 0 0;'>
                                        {len(X_train)} muestras
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with info_col4:
                            st.markdown(f"""
                                <div class='metric-card'>
                                    <h4 style='color: #4facfe; margin: 0;'>🧪 Test</h4>
                                    <p style='font-size: 1.2em; font-weight: bold; margin: 10px 0 0 0;'>
                                        {len(X_test)} muestras
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Interpretación de resultados
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("#### 💡 Interpretación de Resultados")
                        
                        if prob_type == "regression":
                            r2 = metrics['R²']
                            rmse = metrics['RMSE']
                            
                            if r2 >= 0.8:
                                quality = "excelente"
                                color = "#22c55e"
                                emoji = "🎯"
                            elif r2 >= 0.6:
                                quality = "buena"
                                color = "#eab308"
                                emoji = "✅"
                            elif r2 >= 0.4:
                                quality = "moderada"
                                color = "#f97316"
                                emoji = "⚠️"
                            else:
                                quality = "baja"
                                color = "#ef4444"
                                emoji = "❌"
                            
                            st.markdown(f"""
                                <div style='background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
                                backdrop-filter: blur(10px); padding: 25px; border-radius: 15px; 
                                border-left: 5px solid {color};'>
                                    <h4 style='color: white; margin: 0 0 15px 0;'>{emoji} Análisis del Modelo</h4>
                                    <p style='color: rgba(255,255,255,0.9); line-height: 1.8; margin: 0;'>
                                        <strong>Calidad del modelo: {quality}</strong><br><br>
                                        • El modelo explica el <strong>{r2*100:.1f}%</strong> de la variabilidad en los datos.<br>
                                        • El error promedio (RMSE) es de <strong>{rmse:.4f}</strong> unidades.<br>
                                        • El error absoluto medio (MAE) es de <strong>{metrics['MAE']:.4f}</strong> unidades.<br>
                                        {'• ¡Excelente ajuste! El modelo es confiable para hacer predicciones.' if r2 >= 0.8 else ''}
                                        {'• Buen ajuste. El modelo puede ser útil pero considere mejorar los features.' if 0.6 <= r2 < 0.8 else ''}
                                        {'• Ajuste moderado. Considere probar otros algoritmos o hacer feature engineering.' if 0.4 <= r2 < 0.6 else ''}
                                        {'• Ajuste bajo. Recomendado probar otros algoritmos o revisar los datos.' if r2 < 0.4 else ''}
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        else:  # Clasificación
                            acc = metrics['Accuracy']
                            f1 = metrics['F1-Score']
                            
                            if acc >= 0.9:
                                quality = "excelente"
                                color = "#22c55e"
                                emoji = "🎯"
                            elif acc >= 0.8:
                                quality = "buena"
                                color = "#eab308"
                                emoji = "✅"
                            elif acc >= 0.7:
                                quality = "moderada"
                                color = "#f97316"
                                emoji = "⚠️"
                            else:
                                quality = "baja"
                                color = "#ef4444"
                                emoji = "❌"
                            
                            st.markdown(f"""
                                <div style='background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
                                backdrop-filter: blur(10px); padding: 25px; border-radius: 15px; 
                                border-left: 5px solid {color};'>
                                    <h4 style='color: white; margin: 0 0 15px 0;'>{emoji} Análisis del Modelo</h4>
                                    <p style='color: rgba(255,255,255,0.9); line-height: 1.8; margin: 0;'>
                                        <strong>Calidad del modelo: {quality}</strong><br><br>
                                        • Precisión general (Accuracy): <strong>{acc*100:.1f}%</strong><br>
                                        • F1-Score: <strong>{f1:.4f}</strong> (balance entre precisión y recall)<br>
                                        • Precision: <strong>{metrics['Precision']*100:.1f}%</strong> de predicciones positivas son correctas<br>
                                        • Recall: <strong>{metrics['Recall']*100:.1f}%</strong> de casos positivos son detectados<br>
                                        {'• ¡Excelente rendimiento! El modelo clasifica con alta precisión.' if acc >= 0.9 else ''}
                                        {'• Buen rendimiento general. Útil para aplicaciones prácticas.' if 0.8 <= acc < 0.9 else ''}
                                        {'• Rendimiento moderado. Considere balancear clases o ajustar hiperparámetros.' if 0.7 <= acc < 0.8 else ''}
                                        {'• Rendimiento bajo. Recomendado revisar datos y probar otros algoritmos.' if acc < 0.7 else ''}
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Botones de acción
                        st.markdown("<br>", unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("🔄 Entrenar Otro Modelo", use_container_width=True):
                                del st.session_state.algorithm
                                st.rerun()
                        
                        with col2:
                            # Crear CSV de resultados
                            results_df = pd.DataFrame({
                                'Real': y_test,
                                'Predicción': y_pred,
                                'Error': y_test - y_pred if prob_type == "regression" else None
                            })
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Descargar Resultados CSV",
                                data=csv,
                                file_name="resultados_ml.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col3:
                            st.button("📊 Nueva Visualización", use_container_width=True, disabled=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error al entrenar el modelo: {str(e)}")
                        st.exception(e)

# Footer elegante
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
    backdrop-filter: blur(10px); padding: 20px; border-radius: 15px; text-align: center; 
    border: 1px solid rgba(255,255,255,0.2);'>
        <p style='color: rgba(255,255,255,0.8); margin: 0;'>
            🎓 <strong>ML System Pro</strong> | Desarrollado con Programación Orientada a Objetos<br>
            📚 Sistema completo de Machine Learning Supervisado<br>
            💻 Arquitectura: DataLoader → DataPreprocessor → ModelTrainer → ModelEvaluator
        </p>
    </div>
""", unsafe_allow_html=True)