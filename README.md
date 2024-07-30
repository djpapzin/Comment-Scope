# YouTube Comment Analyzer

This is a Streamlit web application that allows you to scrape and analyze comments from YouTube videos.

## Features

- **Scrape Comments:**  Scrape comments from any YouTube video using the YouTube Data API.
- **Sentiment Analysis:** Analyze the sentiment of comments using TextBlob.
- **Word Cloud:** Generate a word cloud to visualize the most frequent words in the comments.
- **Comment Length Analysis:** Analyze the distribution of comment lengths.
- **Top Commenters:** Identify the users who have posted the most comments or received the most likes.
- **Sentiment Analysis Over Time:** Track how sentiment changes over time.
- **Topic Extraction:** Identify the main topics discussed in the comments using Gensim.
- **User Engagement Score:** Calculate a score based on likes, replies, and sentiment to identify the most engaging comments.
- **Comment Summary:** Summarize the main points of the comments using Gemini Pro.
- **Trending Videos:** Display a list of trending videos on YouTube and scrape comments from them.

## Getting Started

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up API Keys:**
   - Create a Google Cloud Platform project and enable the YouTube Data API.
   - Obtain an API key and store it in a Streamlit secrets file named `secrets.toml`.
   - Create a Gemini Pro API key and store it in the same `secrets.toml` file.
   - The `secrets.toml` file should look like this:

     ```toml
     [general]
     YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"
     GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
     ```

3. **Run the App:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter a YouTube video URL in the text input field.
2. Click the "Scrape Comments" button.
3. The app will scrape comments, perform analysis, and display the results.

## Notes

- The app uses the YouTube Data API and Gemini Pro API. Make sure you have API keys set up correctly.
- The app requires a Streamlit environment.
- The app may require adjustments based on your specific API keys and environment.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.
