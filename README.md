# YouTube Comment Analyzer Powered by Gemini AI

This Streamlit app leverages the power of Google's Gemini AI to analyze YouTube comments and provide insightful data visualizations and summaries.

## [**Check Out The Demo Here**](https://youtube-comment-ai-scrutinizer.streamlit.app/)

<table>
  <tr>
    <td><img src="demo_images/main_menu.jpeg" alt="Main Demo"></td>
    <td><img src="demo_images/sentiment_analysis.jpeg" alt="Sentiment Analysis Demo"></td>
    <td><img src="demo_images/word_cloud.jpeg" alt="Word Cloud Demo"></td>
  </tr>
  <tr>
    <td><img src="demo_images/comment_length_analysis.jpeg" alt="Comment Length Demo"></td>
    <td><img src="demo_images/user_engagement_score.jpeg" alt="User Engagement Demo"></td>
    <td><img src="demo_images/sentiment_overtime.jpeg" alt="Sentiment Overtime Demo"></td
  </tr>
  <tr>
    <td><img src="demo_images/comments_summary.jpeg" alt="Comments Summary Demo"></td>
    <td><img src="demo_images/top_commenters.jpeg" alt="Top Commenters Demo"></td>
    <td><img src="demo_images/collapsed_menu.jpeg" alt="Collapsed Menu Demo"></td>

  </tr>
</table>

## Features

This application utilizes the advanced capabilities of Google's Gemini AI to provide a comprehensive analysis of YouTube comments. Key features include:

- **Gemini-Powered Comment Summarization:**  Leveraging Gemini's large context window, the app generates concise and insightful summaries of all comments and replies, capturing the essence of the discussion.
- **Sentiment Analysis:**  Gemini helps analyze the sentiment of comments, providing a clear understanding of the overall emotional tone of the audience.
- **In-Depth Analysis:**  Using the Gemini Pro Exp model, the app delves deeper into comment threads, identifying key themes, topics, and patterns.
- **Comparative Analysis:**  Compare and contrast comments across multiple videos, revealing differences in audience sentiment and discussion points.
- **Video Summary Generation:**  Gemini combines video transcripts and comments to create comprehensive summaries, offering a holistic view of the video's content and audience reception.

## Requirements

- Python 3.7+
- Streamlit
- pandas
- matplotlib
- seaborn
- textblob
- wordcloud
- plotly
- st_aggrid
- google-api-python-client
- google-generativeai
- youtube-transcript-api
- emoji

## Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `secrets.toml` file in the `.streamlit` directory and add your API keys:
   ```toml
   [general]
   GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
   YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"
   ```

   **Note:** You can obtain a Gemini API key from [https://generativeai.google.com/](https://generativeai.google.com/) and a YouTube API key from [https://console.developers.google.com/](https://console.developers.google.com/).

## Usage

1. Run the app:
   ```bash
   streamlit run app.py
   ```

2. Enter a YouTube video URL in the text input field.
3. Click the "Scrutinize" button to analyze the comments using Gemini AI.

## Creative Name Suggestions

Here are 10 creative names that emphasize the Gemini AI integration:

1. **Gemini Insights for YouTube**
2. **YouTube Comment Navigator with Gemini**
3. **Gemini-Powered YouTube Comment Analyzer**
4. **CommentScope: Powered by Gemini AI**
5. **YouTube Comment Explorer with Gemini**
6. **Gemini Decoder for YouTube Comments**
7. **CommentLens: A Gemini AI Perspective**
8. **YouTube Comment Summarizer with Gemini**
9. **Gemini-Enhanced YouTube Comment Analysis**
10. **CommentAI: Powered by Gemini**

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgements

- Google Gemini AI
- YouTube Data API
- Streamlit
- st_aggrid

## Disclaimer

This app is for educational and research purposes only. It is not intended for commercial use or to violate any terms of service. Please use this app responsibly and ethically.
