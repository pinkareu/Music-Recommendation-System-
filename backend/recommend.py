import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import json
import random
import numpy as np
import requests


torch.manual_seed(1)  # reproducible results

# Classes / genres
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Playlist names per genre
playlist_names_dict = {
    "blues": ["🎸 Blue Vibes","🌊 Deep Blues","🎶 Soulful Strings","💙 Midnight Blues"],
    "classical": ["🎻 Timeless Strings","🏰 Royal Harmonies","🌌 Symphony Dreams","📜 Classical Chronicles"],
    "country": ["🤠 Country Roads","🌾 Prairie Tunes","🎵 Heartland Hits","🐎 Western Strings"],
    "disco": ["🕺 Disco Fever","✨ Glitter Ball","💃 Saturday Night Funk","🎉 Dancefloor Lights"],
    "hiphop": ["🔥 Flow Masters","🎤 Mic Drop","🏙️ City Rhymes","💥 Street Beats"],
    "jazz": ["🎷 Midnight Jazz","🎶 Blue Notes","☕ Coffee & Sax","🌙 Smooth Sway"],
    "metal": ["⚡ Heavy Riffs","🖤 Dark Thunder","🔥 Metal Storm","🎸 Shredders Unite"],
    "pop": ["🌟 Bubblegum Hits","🎉 Popcorn Party","🎈 Chart Toppers","✨ Pop Parade"],
    "reggae": ["🌴 Island Vibes","🎵 Irie Tunes","☀️ Sunshine Reggae","🌊 Rasta Rhythms"],
    "rock": ["🎸 Guitar Heroes","⚡ Thunder Strings","🤘 Electric Echoes","🔥 Rock Solid"]
}

# RESNET model definition
num_classes = 10
dropout_prob = 0.5

class RESNET(nn.Module):
    def __init__(self, name='ResnetOrig'):
        super(RESNET, self).__init__()
        self.name = name
        self.fc1 = nn.Linear(512*7*7, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x.view(-1, 512*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Helper: probability distribution graph
def plot_prob_distribution(input_probs, top5_probs, save_path):
    print(save_path)
    all_probs = [input_probs.detach().numpy()] + [p.numpy() for p in top5_probs]
    all_probs = np.vstack(all_probs)

    num_samples, num_classes = all_probs.shape
    bar_width = 0.1
    index = np.arange(num_classes)

    fig, ax = plt.subplots(figsize=(15,6))
    for i in range(num_samples):
        label = "Input Song" if i == 0 else f"Recommendation {i}"
        ax.bar(index + i*(bar_width+0.04), all_probs[i], bar_width, label=label)

    ax.set_xlabel("Classes")
    ax.set_ylabel("Probability")
    ax.set_title("Probability Distribution")
    ax.set_xticks(index + (num_samples-1)*(bar_width+0.04)/2)
    ax.set_xticklabels(classes)
    ax.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# Helper: create playlist from top 5 songs stored in S3
def create_top5_playlist(top5_songs, genre_prob_dict, input_song_probs):
    pred_genre_idx = torch.argmax(input_song_probs)
    pred_genre = classes[pred_genre_idx]
    playlist_title = random.choice(playlist_names_dict[pred_genre])

    # Base S3 URL
    s3_base_url = "https://music-files-rec.s3.amazonaws.com"

    # Ensure frontend audio folder exists
    local_audio_dir = "../frontend/build/static/audio"
    os.makedirs(local_audio_dir, exist_ok=True)

    playlist_songs = []

    for i, (song, _) in enumerate(top5_songs, 1):
        song_name = song.removesuffix(".png")
        genre_folder = song_name.split('.')[0]
        # Construct S3 URL
        song_url = f"{s3_base_url}/genres_original/{genre_folder}/{song_name}.wav"
        # Local path
        local_path = os.path.join(local_audio_dir, f"{song_name}.wav")

        # Download if not already exists
        if not os.path.exists(local_path):
            try:
                response = requests.get(song_url, stream=True)
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                print(f"Error downloading {song_name}: {e}")
                continue

        playlist_songs.append({
            "title": f"Mystery Melody {i}",
            "path": f"/audio/{song_name}.wav"
        })

    return {
        "playlist_title": playlist_title,
        "songs": playlist_songs
    }



# Main recommendation function
def recommendation(input_audio_path="../frontend/build/static/audio/input_song.mp3"):
    print(input_audio_path)
    if not os.path.exists(input_audio_path):
        return {"error": "No input song found"}

    # Load pre-trained VGG16
    vgg16 = models.vgg16(pretrained=True)
    print("here")

    # Load audio
    y, sr = librosa.load(input_audio_path, sr=None)
    clipped_song = y[:sr*30]

    # Create mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=clipped_song, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Save spectrogram image
    os.makedirs("../frontend/build/static/images", exist_ok=True)
    spectrogram_file = "../frontend/build/static/images/input_spectrogram.png"
    plt.figure(figsize=(15,6))
    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
    plt.axis('off')
    plt.savefig(spectrogram_file, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Transform spectrogram for model
    transform_224 = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    spect_tensor = transform_224(Image.open(spectrogram_file).convert('RGB'))
    spect_tensor = spect_tensor.unsqueeze(0)  # add batch dim

    # Extract features with VGG16
    features = vgg16.features(spect_tensor)
    features_tensor = torch.from_numpy(features.detach().numpy())
    song_features = features_tensor

    # Load custom RESNET model
    model = RESNET()
    model_path = "model/model_ResnetOrig_bs32_lr0.001_epoch26"
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # Predict genre probabilities
    input_probs = model(song_features)
    input_probs = F.softmax(input_probs, dim=-1)

    # Compute cosine similarity with dataset
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    similarity_dict = {}

    with open("static/genre_prob_dist_dict.json", "r") as f:
        genre_prob_dict = json.load(f)
    for song, probs in genre_prob_dict.items():
        genre_prob_dict[song] = torch.tensor(probs)

    for song, prob_dist in genre_prob_dict.items():
        similarity = cos_sim(input_probs, prob_dist)
        similarity_dict[song] = similarity

    top5_songs = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)[:5]

    # Prepare playlist
    playlist_data = create_top5_playlist(top5_songs, genre_prob_dict, input_probs)

    # Plot probability distribution graph
    prob_graph_path = "../frontend/build/static/images/prob_distribution.png"
    top5_probs = [genre_prob_dict[song[0]].squeeze(0) for song in top5_songs]
    plot_prob_distribution(input_probs, top5_probs, prob_graph_path)

    # Return data for frontend
    return {
        "input_audio": "/audio/input_song",
        "spectrogram": "/images/input_spectrogram.png",
        "probability_graph": "/images/prob_distribution.png",
        "playlist": playlist_data
    }
