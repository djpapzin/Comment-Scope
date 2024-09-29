# CommentScope: Powered by Gemini AI

CommentScope is a cutting-edge Streamlit app that leverages Google's Gemini AI to provide in-depth analysis of YouTube comments. It offers powerful insights and visualizations to help content creators, marketers, and researchers understand their audience better.

## [**Test The Live Demo Here**](https://commentscope.streamlit.app/)

<table>
  <tr>
    <td><img src="demo_images/main_menu.jpeg" alt="Main Demo"></td>
    <td><img src="demo_images/sentiment_analysis.jpeg" alt="Sentiment Analysis Demo"></td>
    <td><img src="demo_images/word_cloud.jpeg" alt="Word Cloud Demo"></td>
  </tr>
  <tr>
    <td><img src="demo_images/comment_length_analysis.jpeg" alt="Comment Length Demo"></td>
    <td><img src="demo_images/user_engagement_score.jpeg" alt="User Engagement Demo"></td>
    <td><img src="demo_images/sentiment_overtime.jpeg" alt="Sentiment Overtime Demo"></td>
  </tr>
  <tr>
    <td><img src="demo_images/comments_summary.jpeg" alt="Comments Summary Demo"></td>
    <td><img src="demo_images/top_commenters.jpeg" alt="Top Commenters Demo"></td>
    <td><img src="demo_images/collapsed_menu.jpeg" alt="Collapsed Menu Demo"></td>
  </tr>
</table>

## Features

- **Gemini-Powered Comment Summarization:** Generate concise and insightful summaries of all comments and replies.
- **Sentiment Analysis:** Analyze the emotional tone of comments using advanced AI techniques.
- **In-Depth Analysis:** Identify key themes, topics, and patterns in comment threads using Gemini Pro Exp.
- **Comparative Analysis:** Compare comments across multiple videos to reveal differences in audience sentiment and discussion points.
- **Video Summary Generation:** Create comprehensive summaries combining video transcripts and comments.
- **Interactive Data Table:** Explore, sort, filter, and download comment data.
- **User Engagement Score:** Measure audience engagement based on likes, replies, and sentiment.
- **Trending Videos Analysis:** Analyze comments on currently trending YouTube videos.
- **Visualizations:** Gain insights through sentiment analysis charts, word clouds, and comment length distributions.
- **Chat with Comments:** Ask questions about the comments and receive AI-generated answers.

## Key Requirements

- **Python 3.7+:** The foundation for the application.
- **Streamlit:**  For building the interactive web app.
- **google-generativeai:** To interact with Google's Gemini AI.
- **youtube-transcript-api:** For fetching YouTube video transcripts.
- **google-api-python-client:** For accessing the YouTube Data API.

**Note:** A complete list of requirements is available in the `requirements.txt` file.

## Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/djpapzin/Comment-Scope.git  
   cd Comment-Scope  
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   - Create a `.streamlit/secrets.toml` file
   - Add your API keys:
     ```toml
     [general]
     GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
     YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"
     ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).

3. Enter a YouTube video URL or select a trending video to analyze.

4. Explore the various analysis features and visualizations provided by CommentScope.

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to submit pull requests, report issues, or request features.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgements

- Google Gemini AI
- YouTube Data API
- Streamlit
- All other open-source libraries used in this project

## Disclaimer

This app is for educational and research purposes only. It is not intended for commercial use or to violate any terms of service. Please use responsibly and ethically.
