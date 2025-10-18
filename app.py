import streamlit as st
from googleapiclient.discovery import build
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import re
import matplotlib.pyplot as plt
import logging
from urllib.parse import urlparse, parse_qs
from wordcloud import WordCloud

# ----------------- CONFIG -----------------
API_KEY = "AIzaSyDsEJeFwjtBLn7-exyAyC_4XvDw7yFvX9M"   # 🔑 Replace with your valid API key
# ------------------------------------------

# Set up logging
logging.basicConfig(filename='errors.log', level=logging.ERROR)

# Load HuggingFace sentiment analysis model (BERT based)
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype="float32")

sentiment_model = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU
)

# YouTube API client
youtube = build("youtube", "v3", developerKey=API_KEY)

# ----------- Helper Functions -----------

def extract_video_id(url):
    """Extract video ID from full YouTube URL"""
    parsed_url = urlparse(url)
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    elif parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]
    return None

def get_comments(video_id, max_comments=500):
    """Fetch comments with pagination"""
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100
        )
        while request and len(comments) < max_comments:
            response = request.execute()
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
            request = youtube.commentThreads().list_next(request, response)

        return comments

    except Exception as e:
        logging.error(f"Error fetching comments: {e}")
        st.error(f"Error fetching comments: {e}")
        return []

def preprocess(text):
    """Clean comment text"""
    text = re.sub(r"http\S+", "", text)   # remove links
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text)  # remove special chars
    return text.strip()

def get_bert_sentiment(text):
    """Predict sentiment using BERT"""
    result = sentiment_model(text[:512])[0]
    label = result["label"]

    # Convert 5-star output into 3 classes
    if "1" in label or "2" in label:
        return "negative"
    elif "3" in label:
        return "neutral"
    else:
        return "positive"

def generate_wordcloud(texts, video_id):
    """Generate word cloud"""
    text = " ".join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most Frequent Words in Comments")
    plt.savefig(f"wordcloud_{video_id}.png")
    plt.close()

# ----------- Streamlit App -----------

st.title(" YouTube Sentiment Analysis with BERT")

youtube_url = st.text_input("Enter YouTube Video URL:")

if st.button("Analyze Video"):
    video_id = extract_video_id(youtube_url)

    if not video_id:
        st.error("❌ Invalid YouTube link. Please enter a valid URL.")
    else:
        with st.spinner("Fetching and analyzing comments using BERT..."):
            comments = get_comments(video_id, max_comments=500)

            if not comments:
                st.warning("⚠️ No comments fetched. Comments might be disabled for this video.")
            else:
                df = pd.DataFrame(comments, columns=["comment"])
                df["cleaned_comment"] = df["comment"].apply(preprocess)
                df["sentiment"] = df["cleaned_comment"].apply(get_bert_sentiment)

                # Sentiment distribution
                sentiment_counts = df["sentiment"].value_counts(normalize=True) * 100
                st.subheader("📊 Sentiment Distribution")
                st.write(sentiment_counts)

                # Save CSV
                csv_file = f"youtube_sentiment_results_{video_id}.csv"
                df.to_csv(csv_file, index=False)
                st.success(f"💾 Results saved to {csv_file}")

                # Pie chart
                plt.figure(figsize=(6, 6))
                plt.pie(
                    sentiment_counts,
                    labels=sentiment_counts.index,
                    autopct="%1.1f%%",
                    colors=["green", "gray", "red"]
                )
                plt.title(f"Sentiment Distribution for Video {video_id}")
                pie_chart_file = f"sentiment_pie_{video_id}.png"
                plt.savefig(pie_chart_file)
                plt.close()
                st.image(pie_chart_file)

                # Word cloud
                generate_wordcloud(df["cleaned_comment"], video_id)
                st.image(f"wordcloud_{video_id}.png")

                # Show DataFrame
                st.subheader("💬 Analyzed Comments")
                st.dataframe(df)
