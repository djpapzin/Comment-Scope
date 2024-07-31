# Youtube Comment AI Scrutinizer

This Streamlit app scrapes YouTube comments, analyzes them using sentiment analysis and word clouds, and provides insights into user engagement and trending topics.

## Features

- Scrapes comments from any YouTube video URL.
- Performs sentiment analysis on comments.
- Generates word clouds to visualize common words.
- Analyzes comment length distribution.
- Identifies top commenters by number of comments or total likes.
- Visualizes sentiment over time.
- Provides an interactive data table for exploring comments.
- Calculates user engagement scores.
- Summarizes comments using the Gemini API.
- Displays trending videos and allows scraping comments for them.

## Setup

1. **Create a Google Cloud Project:**
   - Go to [https://console.cloud.google.com/](https://console.cloud.google.com/) and create a new project.
2. **Enable the Generative AI API:**
   - In your project, go to "APIs & Services" -> "Library" and search for "Generative AI API".
   - Enable the API.
3. **Get an API Key:**
   - In your project, go to "APIs & Services" -> "Credentials" -> "Create credentials" -> "API key".
   - Copy the API key and store it securely.
4. **Enable the YouTube Data API v3:**
   - In your project, go to "APIs & Services" -> "Library" and search for "YouTube Data API v3".
   - Enable the API.
5. **Get a YouTube Data API Key:**
   - Go to [https://console.cloud.google.com/apis/credentials](https://console.cloud.google.com/apis/credentials) and create a new API key.
   - Enable the YouTube Data API v3.
   - Copy the API key and store it securely.
6. **Create a `.secrets.toml` file:**
   - Create a file named `.secrets.toml` in the root directory of the project.
   - Add the following lines to the file, replacing the placeholders with your API keys:

   ```toml
   [general]
   GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
   YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"
   ```

7. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

8. **Run the App:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter a YouTube video URL in the text input field.
2. Click the "Scrape Comments" button.
3. The app will scrape comments, perform analysis, and display the results.
4. Explore the interactive data table, word cloud, sentiment analysis, and other visualizations.
5. Download the comments as a CSV or JSON file.

## Notes

- The app uses the Gemini API for comment summarization.
- The app requires an internet connection to access the YouTube Data API and Gemini API.
- The app may take some time to scrape comments, depending on the number of comments on the video.
- The app uses the `st_aggrid` library for the interactive data table. You may need to install it separately: `pip install st_aggrid`

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
