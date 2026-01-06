import dash
from dash import dcc, html, Output, Input, State
import dash_bootstrap_components as dbc
import base64
import io
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS

from detect import load_models, detect_language
from features import extract_mfcc_from_bytes
from pydub import AudioSegment

# ======================
# MODELS
# ======================
models = load_models("models")

lang_codes = {
    'Arabic': 'ar',
    'French': 'fr-FR',
    'German': 'de-DE',
    'Japanese': 'ja-JP',
    'Spanish': 'es-ES'
}

translate_codes = {
    'Arabic': 'ar',
    'French': 'fr',
    'German': 'de',
    'Japanese': 'ja',
    'Spanish': 'es'
}

# ======================
# APP
# ======================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ======================
# LAYOUT
# ======================
app.layout = dbc.Container([

    dbc.Row([
        dbc.Col([

            html.H2(
                "Détecteur de langue, reconnaissance vocale, traduction et synthèse vocale",
                className="text-center mb-4"
            ),
            html.Hr(),

            dbc.Card(
                dbc.CardBody([

                    dcc.Upload(
                        id='upload-audio',
                        children=html.Div([
                            html.I(className="bi bi-upload", style={'fontSize': '24px'}),
                            html.Br(),
                            'Glisse et dépose fichier audio ici ou ',
                            html.A('Clique pour sélectionner un fichier')
                        ]),
                        style={
                            'width': '100%',
                            'height': '120px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'paddingTop': '25px',
                            'backgroundColor': '#f8f9fa'
                        },
                        multiple=False
                    ),

                ])
            ),

            html.Br(),

            # ===== AUDIO ORIGINAL (AVANT DÉTECTION) =====
            html.Div(id="original-audio-player"),

            # ===== LOADING + RÉSULTATS AUDIO =====
            dcc.Loading(
                id="loading-audio",
                type="default",
                color="#5bc0eb",
                children=html.Div(id='audio-results')
            ),

            # ===== BLOC TRADUCTION =====
            html.Div(
                id="translation-block",
                style={"display": "none"},
                children=[
                    html.Label("Traduire le message vers :", className="fw-bold"),
                    dcc.Dropdown(
                        id='target-lang',
                        options=[{'label': l, 'value': l} for l in translate_codes.keys()],
                        placeholder="Choisir une langue",
                        style={'width': '60%'}
                    ),
                    html.Br(),
                    dbc.Button(
                        "Traduire",
                        id="btn-translate",
                        color="primary",
                        n_clicks=0
                    )
                ]
            ),

            html.Br(),

            # ===== LOADING TRADUCTION =====
            dcc.Loading(
                id="loading-translation",
                type="default",
                color="#5bc0eb",
                children=html.Div(id='translation-results')
            ),

        ], md=10, lg=8)
    ], justify="center")

], fluid=True, className="mt-4")

# ======================
# CALLBACK: AUDIO ORIGINAL (AVANT DÉTECTION)
# ======================
@app.callback(
    Output("original-audio-player", "children"),
    Input("upload-audio", "contents")
)
def play_original_audio(contents):
    if contents is None:
        return ""

    return html.Audio(
        src=contents,
        controls=True,
        style={
            "width": "100%",
            "marginTop": "10px"
        }
    )

# ======================
# CALLBACK: AUDIO (DÉTECTION + STT)
# ======================
@app.callback(
    Output('audio-results', 'children'),
    Output('translation-block', 'style'),
    Input('upload-audio', 'contents')
)
def process_audio(contents):
    if contents is None:
        return dbc.Alert("Aucun fichier audio téléchargé.", color="secondary"), {"display": "none"}

    _, content_string = contents.split(',')
    audio_bytes = base64.b64decode(content_string)

    features = extract_mfcc_from_bytes(audio_bytes)
    detected_language, scores = detect_language(features, models)

    scores_html = html.Ul([
        html.Li(f"{lang} : {score:.2f}") for lang, score in scores.items()
    ])

    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)

    try:
        text_google = recognizer.recognize_google(
            audio_data,
            language=lang_codes.get(detected_language, 'en-US')
        )
    except:
        text_google = ""

    audio_card = dbc.Card(
        dbc.CardBody([
            html.H4(f"Langue détectée : {detected_language}", className="text-primary"),
            html.H6("Scores par langue :", className="fw-bold"),
            scores_html,
            html.Hr(),
            html.H6("Reconnaissance vocale (Google) :", className="fw-bold"),
            html.P(text_google if text_google else "(Message non reconnu)")
        ]),
        className="shadow-sm mt-4"
    )

    return audio_card, {"display": "block"}

# ======================
# CALLBACK: TRADUCTION
# ======================
@app.callback(
    Output('translation-results', 'children'),
    Input('btn-translate', 'n_clicks'),
    State('upload-audio', 'contents'),
    State('target-lang', 'value')
)
def translate_audio(n_clicks, contents, target_lang):
    if n_clicks == 0 or contents is None or target_lang is None:
        return ""

    _, content_string = contents.split(',')
    audio_bytes = base64.b64decode(content_string)

    features = extract_mfcc_from_bytes(audio_bytes)
    detected_language, _ = detect_language(features, models)

    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)

    try:
        text_google = recognizer.recognize_google(
            audio_data,
            language=lang_codes.get(detected_language, 'en-US')
        )
    except:
        text_google = ""

    if not text_google:
        translated_text = "(Message non reconnu)"
        tts_audio_base64 = None
    else:
        try:
            translated_text = GoogleTranslator(
                source=translate_codes[detected_language],
                target=translate_codes[target_lang]
            ).translate(text_google)

            tts = gTTS(text=translated_text, lang=translate_codes[target_lang])
            mp3_io = io.BytesIO()
            tts.write_to_fp(mp3_io)
            mp3_io.seek(0)
            tts_audio_base64 = base64.b64encode(mp3_io.read()).decode()
        except:
            translated_text = "(Erreur de traduction)"
            tts_audio_base64 = None

    return dbc.Card(
        dbc.CardBody([
            html.H6(f"Traduction vers {target_lang} :", className="fw-bold"),
            html.P(translated_text),
            html.Audio(
                src=f"data:audio/mp3;base64,{tts_audio_base64}",
                controls=True,
                style={'width': '100%'}
            ) if tts_audio_base64 else None
        ]),
        className="shadow-sm mt-4"
    )

# ======================
# RUN
# ======================
if __name__ == '__main__':
    app.run(debug=True)
