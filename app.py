import os
import pandas as pd
from googleapiclient.discovery import build
import logging
import streamlit as st
from googleapiclient.errors import HttpError

# Define the analyze_sentiment function
def analyze_sentiment(text):
    # This is a simple sentiment analysis function that returns a score between -1 and 1
    # You can replace this with a more complex sentiment analysis function or library
    if text == "":
        return 0
    else:
        return 1

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
                    analyze_sentiment(comment["textDisplay"])  # Analyze sentiment here
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
                            analyze_sentiment(reply_comment["textDisplay"])  # Analyze sentiment here
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
