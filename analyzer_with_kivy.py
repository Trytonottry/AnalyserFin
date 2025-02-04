from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.label import Label
import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from dash import Dash, dcc, html
from transformers import pipeline

# --- Автоматическая установка необходимых библиотек ---
REQUIRED_LIBRARIES = [
    "requests", "numpy", "pandas", "dash", "plotly", "scikit-learn", "yfinance", "kivy", "transformers"
]

for library in REQUIRED_LIBRARIES:
    try:
        __import__(library)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", library])

# --- Функции для анализа данных --- (анализ акций, криптовалют, экономики и новостей)
def fetch_financial_data(symbol, period="3mo", interval="1d"):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)
    data.reset_index(inplace=True)
    return data

def fetch_crypto_data(crypto_id, vs_currency="usd", days="30"):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency={vs_currency}&days={days}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        prices = data.get("prices", [])
        return pd.DataFrame(prices, columns=["timestamp", "price"])
    else:
        raise Exception(f"Ошибка API CoinGecko: {response.status_code}, {response.text}")

def fetch_gdp_data():
    url = "http://api.worldbank.org/v2/country/USA/indicator/NY.GDP.MKTP.CD?format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if len(data) > 1:
            return pd.DataFrame(data[1])
        else:
            raise Exception("Некорректный ответ от World Bank API")
    else:
        raise Exception(f"Ошибка API World Bank: {response.status_code}, {response.text}")

def fetch_news(news_api_key, query, from_date, to_date, language="en", page_size=20):
    url = (
        f"https://newsapi.org/v2/everything?q={query}&"
        f"from={from_date}&to={to_date}&"
        f"language={language}&"
        f"pageSize={page_size}&"
        f"apiKey={news_api_key}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        raise Exception(f"Ошибка API NewsAPI: {response.status_code}, {response.text}")

def sentiment_analysis(texts):
    sentiment_analyzer = pipeline("sentiment-analysis")
    return sentiment_analyzer(texts)

def train_predict(data_values):
    data = np.array(data_values)
    X = np.arange(len(data)).reshape(-1, 1)
    y = data

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    return X, y, predictions

def build_dashboard(financial_data=None, crypto_data=None, gdp_data=None, sentiment_data=None):
    app = Dash(__name__)
    layout_elements = []

    if financial_data is not None:
        financial_data["date"] = financial_data["Date"]
        target_data = financial_data["Close"]
        X, y, predictions = train_predict(target_data)

        fig_financial = go.Figure()
        fig_financial.add_trace(go.Scatter(x=financial_data["date"], y=target_data, mode='lines+markers', name='Реальные данные'))
        fig_financial.add_trace(go.Scatter(x=financial_data["date"], y=predictions, mode='lines', name='Прогноз'))
        fig_financial.update_layout(title="Анализ акций", xaxis_title="Дата", yaxis_title="Цена закрытия")

        layout_elements.append(html.Div([
            html.H3("График акций:"),
            dcc.Graph(figure=fig_financial)
        ]))

    if crypto_data is not None:
        crypto_data["date"] = pd.to_datetime(crypto_data["timestamp"], unit="ms")
        target_data = crypto_data["price"]
        X, y, predictions = train_predict(target_data)

        fig_crypto = go.Figure()
        fig_crypto.add_trace(go.Scatter(x=crypto_data["date"], y=target_data, mode='lines+markers', name='Реальные данные'))
        fig_crypto.add_trace(go.Scatter(x=crypto_data["date"], y=predictions, mode='lines', name='Прогноз'))
        fig_crypto.update_layout(title="Анализ криптовалюты", xaxis_title="Дата", yaxis_title="Цена (USD)")

        layout_elements.append(html.Div([
            html.H3("График криптовалют:"),
            dcc.Graph(figure=fig_crypto)
        ]))

    if gdp_data is not None:
        gdp_data = gdp_data.sort_values(by="date")
        target_data = gdp_data["value"].dropna()
        X, y, predictions = train_predict(target_data)

        fig_gdp = go.Figure()
        fig_gdp.add_trace(go.Scatter(x=gdp_data["date"], y=target_data, mode='lines+markers', name='Реальные данные'))
        fig_gdp.add_trace(go.Scatter(x=gdp_data["date"], y=predictions, mode='lines', name='Прогноз'))
        fig_gdp.update_layout(title="Прогноз ВВП", xaxis_title="Дата", yaxis_title="ВВП (доллары США)")

        layout_elements.append(html.Div([
            html.H3("График ВВП:"),
            dcc.Graph(figure=fig_gdp)
        ]))

    if sentiment_data is not None:
        sentiment_scores = [res['score'] for res in sentiment_data if 'score' in res]
        sentiment_labels = [res['label'] for res in sentiment_data if 'label' in res]

        fig_sentiment = go.Figure(data=[
            go.Bar(x=list(range(len(sentiment_data))), y=sentiment_scores, marker_color=['green' if s == 'POSITIVE' else 'red' for s in sentiment_labels])
        ])
        fig_sentiment.update_layout(title="Тональность новостей", xaxis_title="Новости", yaxis_title="Уровень уверенности")

        layout_elements.append(html.Div([
            html.H3("Тональность новостей:"),
            dcc.Graph(figure=fig_sentiment)
        ]))

    app.layout = html.Div([
        html.H1("Единый анализ данных"),
        *layout_elements
    ])

    app.run_server(debug=True)

# --- Реализация Kivy интерфейса для настройки ---
class AnalyzerApp(App):
    def build(self):
        self.selected_analysis = None
        layout = BoxLayout(orientation='vertical', spacing=10, padding=20)

        self.spinner = Spinner(
            text='Выберите анализ',
            values=('Акции', 'Криптовалюты', 'Экономика', 'Новости'),
            size_hint=(1, 0.2)
        )
        layout.add_widget(self.spinner)

        self.param_input = TextInput(hint_text='Введите параметры (например, тикер, ID криптовалюты и т.д.)', multiline=False)
        layout.add_widget(self.param_input)

        self.button = Button(text='Запустить анализ', size_hint=(1, 0.2))
        self.button.bind(on_press=self.process_input)
        layout.add_widget(self.button)

        self.result_label = Label(text='', size_hint=(1, 0.2))
        layout.add_widget(self.result_label)

        return layout

    def process_input(self, _):
        analysis_type = self.spinner.text
        user_input = self.param_input.text

        try:
            if analysis_type == 'Акции':
                self.result_label.text = 'Анализ акций запущен...'
                symbol, period = user_input.split(',')
                financial_data = fetch_financial_data(symbol.strip(), period.strip())
                build_dashboard(financial_data=financial_data)

            elif analysis_type == 'Криптовалюты':
                self.result_label.text = 'Анализ криптовалют запущен...'
                crypto_id, days = user_input.split(',')
                crypto_data = fetch_crypto_data(crypto_id.strip(), days=days.strip())
                build_dashboard(crypto_data=crypto_data)

            elif analysis_type == 'Экономика':
                self.result_label.text = 'Анализ ВВП запущен...'
                gdp_data = fetch_gdp_data()
                build_dashboard(gdp_data=gdp_data)

            elif analysis_type == 'Новости':
                self.result_label.text = 'Анализ новостей запущен...'
                news_api_key, query, start_date, end_date = user_input.split(',')
                news_articles = fetch_news(news_api_key.strip(), query.strip(), start_date.strip(), end_date.strip())
                texts = [article.get("title", "") for article in news_articles]
                sentiment_data = sentiment_analysis(texts)
                build_dashboard(sentiment_data=sentiment_data)

            else:
                self.result_label.text = 'Неверный выбор анализа.'
        except Exception as e:
            self.result_label.text = f'Ошибка: {e}'

if __name__ == '__main__':
    AnalyzerApp().run()