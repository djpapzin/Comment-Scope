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

# Load API key from Streamlit secrets
gemini_api_key = st.secrets["general"]["GEMINI_API_KEY"]
youtube_api_key = st.secrets["general"]["YOUTUBE_API_KEY"]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Gemini
genai.configure(api_key=gemini_api_key)

# --- Start of Gemini Integration ---

# Create the model
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Start a chat session
chat_session = model.start_chat()

# --- End of Gemini Integration ---

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    patterns = [
        r"(?<=v=)[^&]+",
        r"(?<=be\/)[^?]+",
        r"(?<=embed\/)[^\"?]+",
        r"(?<=youtu.be\/)[^\"?]+"
    ]
    for pattern in patterns:
        video_id = re.search(pattern, url)
        if video_id:
            return video_id.group(0)
    return None

# Function to perform sentiment analysis using TextBlob
def analyze_sentiment(comment):
    analysis = TextBlob(comment)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Function to scrape YouTube comments
def scrape_youtube_comments(youtube_api_key, video_id):
    youtube = build('youtube', 'v3', developerKey=youtube_api_key, cache_discovery=False)
    comments = []
    try:
        next_page_token = None
        page_count = 0
        progress_bar = st.progress(0)
        while True:
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append([
                    comment["authorDisplayName"],
                    comment["textDisplay"],
                    comment["likeCount"],
                    comment["publishedAt"],
                    item["snippet"]["totalReplyCount"],
                    analyze_sentiment(comment["textDisplay"])
                ])

                if "replies" in item:
                    for reply in item["replies"]["comments"]:
                        reply_comment = reply["snippet"]
                        comments.append([
                            reply_comment["authorDisplayName"],
                            reply_comment["textDisplay"],
                            reply_comment["likeCount"],
                            reply_comment["publishedAt"],
                            0,
                            analyze_sentiment(reply_comment["textDisplay"])
                        ])

            if "nextPageToken" in response:
                next_page_token = response["nextPageToken"]
            else:
                break

            page_count += 1
            progress_bar.progress(min(page_count / 10, 1.0))

        df = pd.DataFrame(comments, columns=["Name", "Comment", "Likes", "Time", "Reply Count", "Sentiment"])
        # Convert 'Time' to datetime in the DataFrame
        df['Time'] = pd.to_datetime(df['Time'], utc=True)  # Convert 'Time' to datetime
        total_comments = len(comments)
        return df, total_comments

    except HttpError as e:
        logging.error(f"HTTP error occurred: {e}")
        st.error(f"HTTP error occurred: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Error scraping comments: {e}")
        st.error(f"Error scraping comments: {e}")
        return None, None

def generate_reply(model, chat_session, comment, video_description):
    # TO DO
    pass

# Function to generate a word cloud
def generate_word_cloud(text, stopwords=None, colormap='viridis', contour_color='steelblue'):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords, colormap=colormap, contour_color=contour_color).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Function to analyze comment length
def analyze_comment_length(df):
    comment_lengths = df["Comment"].str.len()
    st.write("Comment Length Statistics:")
    st.write(f"Average Length: {comment_lengths.mean():.2f} characters")
    st.write(f"Median Length: {comment_lengths.median()} characters")
    st.write(f"Maximum Length: {comment_lengths.max()} characters")
    st.write(f"Minimum Length: {comment_lengths.min()} characters")
    fig, ax = plt.subplots()
    sns.histplot(comment_lengths, kde=True, ax=ax)
    ax.set_title("Comment Length Distribution")
    st.pyplot(fig)

# Function to get top commenters
def get_top_commenters(df, by="comments", top_n=10):
    if by == "comments":
        top_commenters = df["Name"].value_counts().head(top_n)
    elif by == "likes":
        top_commenters = df.groupby("Name")["Likes"].sum().sort_values(ascending=False).head(top_n)
    else:
        st.error("Invalid option for 'by'. Choose 'comments' or 'likes'.")
        return
    st.write(f"Top {top_n} Commenters by {by.capitalize()}:\")")
    st.write(top_commenters)

# Function to export visualization
def export_visualization(fig, filename):
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    st.success(f"Visualization saved as {filename}")

# Function to analyze sentiment over time
def analyze_sentiment_over_time(df):
    df["Date"] = pd.to_datetime(df["Time"]).dt.date
    df['Time'] = pd.to_datetime(df['Time'])
    sentiment_over_time = df.groupby(["Date", "Sentiment"]).size().unstack(fill_value=0)
    fig = px.line(sentiment_over_time, title='Sentiment Over Time')
    st.plotly_chart(fig)

# Function to display an interactive data table
def display_interactive_table(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gridOptions = gb.build()
    AgGrid(df, gridOptions=gridOptions, enable_enterprise_modules=True, update_mode=GridUpdateMode.SELECTION_CHANGED, data_return_mode=DataReturnMode.FILTERED_AND_SORTED)

# Function to extract topics from comments
def extract_topics(df, num_topics=5, num_words=10):
    if df['Comment'].isnull().all():
        st.write("No comments to analyze.")
        return

    comments = df['Comment'].str.lower().str.split()
    comments = [comment for comment in comments if comment]

    if not comments:
        st.write("No valid comments to analyze.")
        return

    dictionary = corpora.Dictionary(comments)
    if len(dictionary) == 0:
        st.write("No tokens found in comments.")
        return

    corpus = [dictionary.doc2bow(comment) for comment in comments]
    if not corpus:
        st.write("No terms in the corpus to analyze.")
        return

    lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=2)
    topics = lda_model.print_topics(num_words=num_words)

# Function to get trending videos
def get_trending_videos(youtube_api_key):
    youtube = build('youtube', 'v3', developerKey=youtube_api_key, cache_discovery=False)
    request = youtube.videos().list(part="snippet,statistics", chart="mostPopular", regionCode="US", maxResults=10)
    response = request.execute()
    videos = []
    for item in response["items"]:
        video = {
            "videoId": item["id"],
            "title": item["snippet"]["title"],
            "channelTitle": item["snippet"]["channelTitle"],
            "viewCount": item["statistics"].get("viewCount", 0),
            "likeCount": item["statistics"].get("likeCount", 0),
            "commentCount": item["statistics"].get("commentCount", 0)
        }
        videos.append(video)
    return videos

# Function to display video metadata
def display_video_metadata(video):
    st.write("Video Title:", video["title"])
    st.write("Channel Title:", video["channelTitle"])
    st.write("View Count:", video["viewCount"])
    st.write("Like Count:", video["likeCount"])
    st.write("Comment Count:", video["commentCount"])

# Function to calculate user engagement score
def calculate_engagement(df):
    df["EngagementScore"] = df["Likes"] + df["Reply Count"] * 2 + df["Sentiment"].apply(lambda x: 1 if x == 'Positive' else (-1 if x == 'Negative' else 0))
    return df

# Function to monitor API quota
def api_quota_monitor(gemini_api_key):
    genai.configure(api_key= gemini_api_key)  # Configure for Gemini
    try:
        # Make a simple request to Gemini to get quota info
        response = genai.GenerativeModel(
            model="gemini-1.5-flash",  # Use Gemini Pro model
            prompt="Hello, world!",
        )
        return response.get('quota_remaining', None)
    except HttpError as e:
        print(f"An HTTP error occurred: {e.resp.status} {e.content}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None

# Function to summarize comments
def summarize_comments(model, comments):
    """Summarizes all comments in the list.

    Args:
        model: The Gemini model object.
        comments: A list of comments to summarize.

    Returns:
        A string containing the summary of all comments.
    """
    if not comments:
        return "No comments to summarize."
    
    # Combine all comments into a single string
    all_comments = "\n\n".join(comments)
    prompt = f"Summarize the following YouTube comments:\n\n{all_comments}"
    try:
        response = model.generate_text(
            model="gemini-1.5-flash",  # Use Gemini Pro model
            prompt=prompt,
        )
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error summarizing comments: {e}")
        return "Error summarizing comments."


# Streamlit App
st.title("YouTube Comment Scraper and Analyzer")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()
if 'filtered_df' not in st.session_state:
    st.session_state['filtered_df'] = pd.DataFrame()

video_url = st.text_input("Enter YouTube video URL")

# Scrape Comments Button
if st.button("Scrape Comments", key="scrape_comments_button"):  # Unique key
    video_id = extract_video_id(video_url)
    if video_id:
        with st.spinner("Scraping comments..."):
            progress_bar = st.progress(0)
            df, total_comments = scrape_youtube_comments(youtube_api_key, video_id)
            progress_bar.progress(1)
            if df is None or total_comments is None:
                st.error("Error scraping comments. Please try again.")
            else:
                st.success(f"Scraping complete! Total Comments: {total_comments}")
                # Store the original DataFrame
                st.session_state['df'] = df.copy()  # Store the scraped DataFrame
                st.write(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download CSV", data=csv, file_name="youtube_comments.csv", mime="text/csv")

                # Sentiment Analysis Visualization
                st.subheader("Sentiment Analysis")
                sentiment_counts = df['Sentiment'].value_counts()
                fig, ax = plt.subplots()
                ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
                ax.axis('equal')
                st.pyplot(fig)
                export_visualization(fig, "sentiment_analysis.png")

                # Generate Word Cloud
                st.subheader("Word Cloud")
                all_comments = ' '.join(df['Comment'])
                generate_word_cloud(all_comments)

                # Comment Length Analysis
                st.subheader("Comment Length Analysis")
                analyze_comment_length(df)

                # Top Commenters
                st.subheader("Top Commenters")
                top_commenters_by_comments = st.checkbox("Top Commenters by Number of Comments")
                top_commenters_by_likes = st.checkbox("Top Commenters by Total Likes")
                top_n = st.number_input("Number of Top Commenters", min_value=1, value=10, step=1)

                if top_commenters_by_comments:
                    get_top_commenters(df, by="comments", top_n=top_n)

                if top_commenters_by_likes:
                    get_top_commenters(df, by="likes", top_n=top_n)

                # Sentiment Analysis Over Time
                st.subheader("Sentiment Analysis Over Time")
                analyze_sentiment_over_time(df)

                # Interactive Data Table
                st.subheader("Interactive Comment Table")
                display_interactive_table(df)

                # Topic Extraction
                st.subheader("Topic Extraction")
                extract_topics(df)

                # User Engagement Score
                st.subheader("User Engagement Score")
                df = calculate_engagement(df)
                st.write(df[["Name", "Comment", "EngagementScore"]].sort_values(by="EngagementScore", ascending=False))

                # Comment Summary
                st.subheader("Comment Summary")
                st.write(summarize_comments(model, df["Comment"].tolist()))

# Display trending videos
st.header("Trending Videos")
trending_videos = get_trending_videos(youtube_api_key)
if trending_videos:
    video_selection = st.selectbox("Select a trending video", [f"{video['title']} (by {video['channelTitle']})" for video in trending_videos])
    selected_video = next(video for video in trending_videos if f"{video['title']} (by {video['channelTitle']})" == video_selection)
    display_video_metadata(selected_video)
    

    if st.button("Scrape Comments for Trending Video", key="scrape_trending_comments_button"):  # Unique key
        video_id = selected_video['videoId']
        with st.spinner("Scraping comments..."):
            progress_bar = st.progress(0)
            df, total_comments = scrape_youtube_comments(youtube_api_key, video_id)
            progress_bar.progress(1)
            if df is None or total_comments is None:
                st.error("Error scraping comments. Please try again.")
            else:
                st.success(f"Scraping complete! Total Comments: {total_comments}")
                st.write(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download CSV", data=csv, file_name="youtube_comments.csv", mime="text/csv")

                # Sentiment Analysis Visualization
                st.subheader("Sentiment Analysis")
                sentiment_counts = df['Sentiment'].value_counts()
                fig, ax = plt.subplots()
                ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
                ax.axis('equal')
                st.pyplot(fig)
                export_visualization(fig, "sentiment_analysis.png")

                # Generate Word Cloud
                st.subheader("Word Cloud")
                all_comments = ' '.join(df['Comment'])
                generate_word_cloud(all_comments)

                # Comment Length Analysis
                st.subheader("Comment Length Analysis")
                analyze_comment_length(df)

                # Top Commenters
                st.subheader("Top Commenters")
                top_commenters_by_comments = st.checkbox("Top Commenters by Number of Comments")
                top_commenters_by_likes = st.checkbox("Top Commenters by Total Likes")
                top_n = st.number_input("Number of Top Commenters", min_value=1, value=10, step=1)

                if top_commenters_by_comments:
                    get_top_commenters(df, by="comments", top_n=top_n)

                if top_commenters_by_likes:
                    get_top_commenters(df, by="likes", top_n=top_n)

                # Sentiment Analysis Over Time
                st.subheader("Sentiment Analysis Over Time")
                analyze_sentiment_over_time(df)

                # Interactive Data Table
                st.subheader("Interactive Comment Table")
                display_interactive_table(df)

                # Topic Extraction
                st.subheader("Topic Extraction")
                extract_topics(df)

                # User Engagement Score
                st.subheader("User Engagement Score")
                df = calculate_engagement(df)
                st.write(df[["Name", "Comment", "EngagementScore"]].sort_values(by="EngagementScore", ascending=False))

                # Comment Summary
                st.subheader("Comment Summary")
                st.write(summarize_comments(model, df["Comment"].tolist()))


# API Quota Monitor
st.sidebar.subheader("API Quota Monitor")

quota_remaining = api_quota_monitor(gemini_api_key)
if quota_remaining is not None:
    st.sidebar.write(f"API Quota Remaining: {quota_remaining}")
else:
    st.sidebar.write("Unable to retrieve API quota information.") 