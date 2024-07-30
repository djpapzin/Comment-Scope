import logging
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandas import to_datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from textblob import TextBlob
from wordcloud import WordCloud
import gensim
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode, DataReturnMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from gensim import corpora
import google.generativeai as genai

# ... (rest of the code remains the same)
