import logging
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from textblob import TextBlob
from wordcloud import WordCloud
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode, DataReturnMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold  # Import necessary types
import uuid  # Import uuid module
from youtube_transcript_api import YouTubeTranscriptApi  # For transcript retrieval
import emoji  # For emoji support

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

# --- Initialize Gemini Pro Exp Model ---
gemini_pro_exp_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-exp-0801",
    generation_config=generation_config,
)

gemini_pro_exp_chat_session = gemini_pro_exp_model.start_chat()
# --- End of Gemini Pro Exp Initialization ---

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
        progress_bar = st.progress(0, text="Scrutinizing comments...")
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
            progress_bar.progress(min(page_count / 10, 1.0), text=f"Scrutinizing comments... (Page {page_count})")  # More detailed progress update

        df = pd.DataFrame(comments, columns=["Name", "Comment", "Likes", "Time", "Reply Count", "Sentiment"])
        df['Time'] = pd.to_datetime(df['Time'], utc=True)
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
    st.write(f"Top {top_n} Commenters by {by.capitalize()}:")
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
    grid_response = AgGrid(
        df,
        gridOptions=gridOptions,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED
    )
    return grid_response

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

# Function to summarize comments
def summarize_comments(comments):
    if not comments:
        return "No comments to summarize."
    
    all_comments = "\n\n".join(comments)
    prompt = f"Summarize the following YouTube comments:\n\n{all_comments}"
    try:
        # Updated safety settings
        response = chat_session.send_message(prompt, safety_settings=[
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
        ])
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error summarizing comments: {e}")
        st.error(f"Error summarizing comments: {e}")  # Display error message to the user
        return "Error summarizing comments. Please try again later."

# Function to get top comments by likes
def get_top_comments_by_likes(df, top_n=3):
    top_comments = df.nlargest(top_n, "Likes")
    return top_comments[["Name", "Comment", "Likes"]]

# Function to perform in-depth analysis with Gemini Pro Exp
def in_depth_analysis(comments):
    if not comments:
        return "No comments to analyze."

    # Process comments within a thread as a single unit
    threads = []
    current_thread = []
    for comment in comments:
        if comment.startswith("@"):  # Assuming replies start with "@"
            current_thread.append(comment)
        else:
            if current_thread:
                threads.append("\n".join(current_thread))
            current_thread = [comment]
    if current_thread:
        threads.append("\n".join(current_thread))

    all_comments = "\n\n---\n\n".join(threads)
    prompt = f"Provide an in-depth analysis of the following YouTube comment threads, focusing on the overall sentiment, key themes and topics, and any interesting patterns or insights you can identify:\n\n{all_comments}"
    try:
        response = gemini_pro_exp_chat_session.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error performing in-depth analysis: {e}")
        st.error(f"Error performing in-depth analysis: {e}")  # Display error message to the user
        return "Error performing in-depth analysis. Please try again later."

# Function to perform comparative analysis
def comparative_analysis(dfs, video_ids):
    if not dfs:
        return "No data to compare."

    all_comments = []
    for i, df in enumerate(dfs):
        comments = "\n\n".join(df["Comment"].tolist())
        all_comments.append(f"Comments for Video {video_ids[i]}:\n{comments}")

    all_comments_str = "\n\n---\n\n".join(all_comments)
    prompt = f"Compare and contrast the comments across the following YouTube videos, focusing on the overall sentiment, key themes and topics, and any interesting patterns or insights you can identify:\n\n{all_comments_str}"
    try:
        response = gemini_pro_exp_chat_session.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error performing comparative analysis: {e}")
        return "Error performing comparative analysis."

# Function to generate video summary
def generate_video_summary(video_id, comments):
    youtube = build('youtube', 'v3', developerKey=youtube_api_key, cache_discovery=False)
    request = youtube.videos().list(part="snippet", id=video_id)
    response = request.execute()
    video_details = response["items"][0]["snippet"]

    title = video_details["title"]
    description = video_details["description"]

    # Get transcript
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        logging.error(f"Error fetching transcript: {e}")
        transcript_text = "Transcript not available."

    all_comments = "\n\n".join(comments)

    prompt = f"""
    Generate a comprehensive summary of the YouTube video with the following title, description, and transcript:

    Title: {title}
    Description: {description}
    Transcript: {transcript_text}

    Consider the following comments from viewers:

    {all_comments}

    The summary should include:

    * Key topics covered in the video
    * Main points discussed
    * Overall sentiment of the viewers based on the comments
    * Any interesting patterns or insights from the comments

    Please provide a concise and informative summary.
    """

    try:
        response = gemini_pro_exp_chat_session.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error generating video summary: {e}")
        return "Error generating video summary."

# Streamlit App
st.set_page_config(page_title="Youtube Comment AI Scrutinizer 🔬", page_icon="🔬") # Set page title and favicon
st.title("Youtube Comment AI Scrutinizer 🔬")  # Add emoji to the title

# Initialize session state
if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()
if 'filtered_df' not in st.session_state:
    st.session_state['filtered_df'] = pd.DataFrame()

# --- Single Video Analysis ---
st.header("Single Video Analysis")
video_url = st.text_input("Enter YouTube video URL")

# Scrape Comments Button
if st.button("Scrutinize", key="scrape_comments_button"):
    video_id = extract_video_id(video_url)
    if video_id:
        with st.spinner("Scrutinizing comments..."):
            progress_bar = st.progress(0, text="Scrutinizing comments...")
            df, total_comments = scrape_youtube_comments(youtube_api_key, video_id)
            progress_bar.progress(1.0, text="Scrutinizing comments...")
            if df is None or total_comments is None:
                st.error("Error scraping comments. Please try again.")
            else:
                st.success(f"Scrutinizing Complete! Total Comments: {total_comments}")
                st.session_state['df'] = df.copy()
                st.session_state['filtered_df'] = df.copy()

                # Get video details
                youtube = build('youtube', 'v3', developerKey=youtube_api_key, cache_discovery=False)
                request = youtube.videos().list(part="snippet,statistics", id=video_id)
                response = request.execute()
                video_details = response["items"][0]

                # Display video thumbnail and details
                st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")
                st.write(f"**Title:** {video_details['snippet']['title']}")
                st.write(f"**Views:** {video_details['statistics']['viewCount']}")
                st.write(f"**Likes:** {video_details['statistics']['likeCount']}")

                # Comments Summary
                with st.expander("Comments Summary", expanded=True):
                    try:
                        summary = summarize_comments(df["Comment"].tolist())
                        sentiment = analyze_sentiment(summary)  # Analyze sentiment of the summary
                        emoji_for_sentiment = emoji.emojize(
                            ":thumbs_up:" if sentiment == "Positive"
                            else ":thumbs_down:" if sentiment == "Negative"
                            else ":neutral_face:"
                        )
                        st.write(f"{emoji_for_sentiment} {summary}")  # Add emoji to the summary
                    except Exception as e:
                        st.error(f"Error summarizing comments: {e}")

                # Sentiment Analysis Visualization
                with st.expander("Sentiment Analysis", expanded=False):
                    sentiment_counts = df['Sentiment'].value_counts()
                    fig, ax = plt.subplots()
                    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
                    ax.axis('equal')
                    st.pyplot(fig)
                    export_visualization(fig, "sentiment_analysis.png")

                # Generate Word Cloud
                with st.expander("Word Cloud", expanded=False):
                    all_comments = ' '.join(df['Comment'])
                    generate_word_cloud(all_comments)

                # Comment Length Analysis
                with st.expander("Comment Length Analysis", expanded=False):
                    analyze_comment_length(df)

                # Top Commenters
                with st.expander("Top Commenters", expanded=False):
                    st.subheader("Top Commenters")
                    top_commenters_by_comments = st.checkbox("Top Commenters by Number of Comments")
                    top_commenters_by_likes = st.checkbox("Top Commenters by Total Likes")
                    top_commenters_by_likes_sorted = st.checkbox("Comments Sorted by Likes")
                    top_n = st.number_input("Number of Top Commenters", min_value=1, value=10, step=1)

                    if top_commenters_by_comments:
                        get_top_commenters(df, by="comments", top_n=top_n)

                    if top_commenters_by_likes:
                        get_top_commenters(df, by="likes", top_n=top_n)

                    if top_commenters_by_likes_sorted:
                        st.write(df[["Name", "Comment", "Likes"]].sort_values(by="Likes", ascending=False))

                # Top Comments by Likes
                with st.expander("Top Comments by Likes", expanded=False):
                    st.write(get_top_comments_by_likes(df))

                # Sentiment Analysis Over Time
                with st.expander("Sentiment Analysis Over Time", expanded=False):
                    analyze_sentiment_over_time(df)

                # Interactive Data Table
                with st.expander("Interactive Comment Table", expanded=False):
                    grid_response = display_interactive_table(df)
                    st.session_state['filtered_df'] = grid_response['data']
                    json = st.session_state['filtered_df'].to_json(orient='records')
                    st.download_button(label="Download JSON", data=json, file_name="youtube_comments.json", mime="application/json")

                # User Engagement Score
                with st.expander("User Engagement Score", expanded=False):
                    df = calculate_engagement(df)
                    st.write(df[["Name", "Comment", "EngagementScore"]].sort_values(by="EngagementScore", ascending=False))

                # In-Depth Analysis with Gemini Pro Exp
                with st.expander("In-Depth Analysis (Gemini Pro Exp)", expanded=False):
                    st.write(in_depth_analysis(df["Comment"].tolist()))

                # Video Summary
                with st.expander("Video Summary (Gemini Pro Exp)", expanded=False):
                    summary = generate_video_summary(video_id, df["Comment"].tolist())
                    st.write(summary)

# --- Comparative Analysis ---
st.header("Comparative Analysis")
video_urls = st.text_area("Enter YouTube video URLs (one per line)")
video_urls = video_urls.strip().splitlines()

if st.button("Compare", key="compare_comments_button"):
    video_ids = []
    dfs = []
    for url in video_urls:
        video_id = extract_video_id(url)
        if video_id:
            video_ids.append(video_id)
            with st.spinner(f"Scrutinizing comments for video {video_id}..."):
                df, total_comments = scrape_youtube_comments(youtube_api_key, video_id)
                if df is not None:
                    dfs.append(df)

    if dfs:
        with st.spinner("Performing comparative analysis..."):
            analysis = comparative_analysis(dfs, video_ids)
            st.write(analysis)

# --- Trending Videos ---
st.header("Trending Videos")
trending_videos = get_trending_videos(youtube_api_key)
if trending_videos:
    video_selection = st.selectbox("Select a trending video", [f"{video['title']} (by {video['channelTitle']})" for video in trending_videos])
    selected_video = next(video for video in trending_videos if f"{video['title']} (by {video['channelTitle']})" == video_selection)
    
    # Display video thumbnail
    st.image(f"https://img.youtube.com/vi/{selected_video['videoId']}/hqdefault.jpg")

    # Display video metadata below the thumbnail
    display_video_metadata(selected_video)
    
    # Generate a unique key using uuid
    unique_key = str(uuid.uuid4())
    if st.button("Scrutinize", key=f"scrape_trending_comments_button_{unique_key}"):  # Changed button label
        video_id = selected_video['videoId']
        with st.spinner("Scrutinizing comments..."):
            progress_bar = st.progress(0, text="Scrutinizing comments...")
            df, total_comments = scrape_youtube_comments(youtube_api_key, video_id)
            progress_bar.progress(1.0, text="Scrutinizing comments...")
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
                with st.expander("Top Commenters", expanded=False):
                    top_commenters_by_comments = st.checkbox("Top Commenters by Number of Comments")
                    top_commenters_by_likes = st.checkbox("Top Commenters by Total Likes")
                    top_n = st.number_input("Number of Top Commenters", min_value=1, value=10, step=1)

                    if top_commenters_by_comments:
                        get_top_commenters(df, by="comments", top_n=top_n)

                    if top_commenters_by_likes:
                        get_top_commenters(df, by="likes", top_n=top_n)

                # Top Comments by Likes
                with st.expander("Top Comments by Likes", expanded=False):
                    st.write(get_top_comments_by_likes(df))

                # Sentiment Analysis Over Time
                st.subheader("Sentiment Analysis Over Time")
                analyze_sentiment_over_time(df)

                # Interactive Data Table
                st.subheader("Interactive Comment Table")
                display_interactive_table(df)

                # User Engagement Score
                st.subheader("User Engagement Score")
                df = calculate_engagement(df)
                st.write(df[["Name", "Comment", "EngagementScore"]].sort_values(by="EngagementScore", ascending=False))

                # Comments Summary
                st.subheader("Comments Summary")
                prompt = f"Summarize the following YouTube comments in a neutral and unbiased manner, providing an overview of the video's content and the discussion in the comments section. Please include the main topics, key points, and any notable trends or insights, without taking a stance or making assumptions. The comments are as follows:\n\n{df['Comment'].tolist()}\n\nPlease format the summary in bullet points."
                try:
                    # Updated safety settings
                    response = chat_session.send_message(prompt, safety_settings=[
                        {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
                        {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
                    ])
                    st.write(response.text.strip())
                except Exception as e:
                    st.error(f"Error summarizing comments: {e}")

                # In-Depth Analysis with Gemini Pro Exp
                with st.expander("In-Depth Analysis (Gemini Pro Exp)", expanded=False):
                    st.write(in_depth_analysis(df["Comment"].tolist()))

                # Video Summary
                with st.expander("Video Summary (Gemini Pro Exp)", expanded=False):
                    summary = generate_video_summary(video_id, df["Comment"].tolist())
                    st.write(summary)
