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
from sklearn.metrics.pairwise import cosine_similarity  # For cosine similarity calculation
import numpy as np  # For numerical operations

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
    system_instruction= """You are a YouTube comment specialist powered by Google's Gemini AI. You excel at understanding and analyzing user comments on YouTube videos. You should be able to:

* **Summarize comments:** Provide concise and insightful summaries of comments, capturing the overall sentiment and main topics discussed.
* **Identify key themes and topics:** Analyze comments to identify recurring themes, topics, and patterns in the discussion.
* **Analyze sentiment:** Accurately determine the sentiment of comments (positive, negative, or neutral).
* **Compare and contrast comments:** Analyze comments across multiple videos to identify differences in sentiment and discussion points.
* **Generate video summaries:** Utilize video transcripts and comments to create comprehensive summaries that include key topics, main points, viewer sentiment, and notable patterns.
* **Be objective and unbiased:** Present information in a neutral and unbiased manner, avoiding personal opinions or assumptions.
* **Provide clear and concise responses:** Communicate your findings in a clear, concise, and easy-to-understand format.

Remember to use your advanced AI capabilities to provide in-depth analysis and valuable insights.""",
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
                comments.append({
                    "Comment": comment["textDisplay"],
                    "Likes": comment["likeCount"],
                    "Name": comment["authorDisplayName"],
                    "Time": comment["publishedAt"],
                    "Reply Count": comment.get("totalReplyCount", 0)  # Add Reply Count
                })

                if "replies" in item:
                    for reply in item["replies"]["comments"]:
                        reply_comment = reply["snippet"]
                        comments.append({
                            "Comment": reply_comment["textDisplay"],
                            "Likes": reply_comment["likeCount"],
                            "Name": reply_comment["authorDisplayName"],
                            "Time": reply_comment["publishedAt"],
                            "Reply Count": 0  # Replies don't have replies
                        })

            if "nextPageToken" in response:
                next_page_token = response["nextPageToken"]
            else:
                break

            page_count += 1
            progress_bar.progress(min(page_count / 10, 1.0), text=f"Scrutinizing comments... (Page {page_count})")  # More detailed progress update

        df = pd.DataFrame(comments)
        df["Sentiment"] = df["Comment"].apply(analyze_sentiment)  # Add this line to perform sentiment analysis
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
        # Adjusted safety settings
        response = chat_session.send_message(prompt, safety_settings=[
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
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

# --- New Functions for Community Consensus ---

def identify_controversial_topics(comments):
    # TODO: Implement logic to identify controversial topics
    # This is a placeholder, you'll need to use NLP techniques
    # to analyze the comments and identify topics with significant
    # positive and negative sentiment.
    return ["Example Controversial Topic 1", "Example Controversial Topic 2"]

def summarize_community_consensus(comments, topic):
    # TODO: Implement logic to summarize the majority opinion
    # This is a placeholder, you'll need to use NLP techniques
    # to analyze the comments related to the topic and summarize
    # the majority opinion.
    return f"Majority opinion on {topic}: This is a placeholder for the summary."

# --- End of New Functions ---

# --- Function for Chat with Comments ---
def chat_with_comments(df, question):
    # Generate embeddings for each comment
    comment_embeddings = []
    for comment in df["Comment"]:
        embedding = genai.embed_content(
            model="models/embedding-001",
            content=comment,
            task_type="RETRIEVAL_DOCUMENT"
        )
        comment_embeddings.append(embedding["embedding"])

    # Generate embedding for the question
    question_embedding = genai.embed_content(
        model="models/embedding-001",
        content=question,
        task_type="RETRIEVAL_QUERY"
    )["embedding"]

    # Calculate cosine similarity
    similarities = cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(comment_embeddings))

    # Get the index of the most similar comment
    most_similar_index = np.argmax(similarities)

    # Get the most similar comment
    most_similar_comment = df["Comment"].iloc[most_similar_index]

    # Use Gemini Pro Exp to generate an answer
    prompt = f"""
    You are a helpful AI assistant. Answer the question based on the context provided.

    Context:
    {most_similar_comment}

    Question:
    {question}

    Answer:
    """
    try:
        response = gemini_pro_exp_chat_session.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error in chat_with_comments: {e}")
        return "Error answering your question. Please try again later."

# --- End of Chat with Comments Function ---

# Streamlit App
st.set_page_config(page_title="CommentScope: Powered by Gemini AI", page_icon="🔬") # Set page title and favicon

# --- Modified Title Section ---
st.markdown("""
    <h1 style="font-size:36px;">🔬 CommentScope</h1>
    <h2 style="font-size:18px;">Powered by Gemini AI</h2>
""", unsafe_allow_html=True)
# --- End of Modified Title Section ---

# Initialize session state
if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()
if 'filtered_df' not in st.session_state:
    st.session_state['filtered_df'] = pd.DataFrame()

# --- Single Video Analysis ---
st.header("Single Video Analysis")
video_url = st.text_input("Enter YouTube video URL")

# Scrape Comments Button
if st.button("Scrutinize Comments"):
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

                # --- Community Consensus ---
                with st.expander("Community Consensus", expanded=False):
                    controversial_topics = identify_controversial_topics(df["Comment"].tolist())
                    if controversial_topics:
                        for topic in controversial_topics:
                            st.write(f"**Topic:** {topic}")
                            consensus = summarize_community_consensus(df["Comment"].tolist(), topic)
                            st.write(consensus)
                    else:
                        st.write("No controversial topics identified.")

                # --- Chat with Comments ---
                with st.expander("Chat with Comments", expanded=False):
                    user_question = st.text_input("Ask a question about the comments:")
                    if st.button("Ask"):
                        if user_question:
                            with st.spinner("Thinking..."):
                                answer = chat_with_comments(df, user_question)
                                st.write(answer)

# --- Comparative Analysis ---
st.header("Comparative Analysis")
video_urls = st.text_area("Enter YouTube video URLs (one per line)")
video_urls = video_urls.strip().splitlines()

if st.button("Compare"):
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
    if st.button("Scrutinize Comments", key=f"scrape_trending_comments_button_{unique_key}"):
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
                        {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
                        {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
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

                # --- Chat with Comments ---
                with st.expander("Chat with Comments", expanded=False):
                    user_question = st.text_input("Ask a question about the comments:")
                    if st.button("Ask"):
                        if user_question:
                            with st.spinner("Thinking..."):
                                answer = chat_with_comments(df, user_question)
                                st.write(answer)
