import React, { useState } from "react";

export default function Playlist({ data }) {
  const { playlist, spectrogram, probability_graph } = data;
  const [expandedSpectrogram, setExpandedSpectrogram] = useState(false);

  if (!playlist) return null;

  return (
    <div style={{ margin: "2rem auto", maxWidth: "1000px", color: "#fff", padding: "1rem" }}>
      {/* Playlist cover */}
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          gap: "2rem",
          alignItems: "center",
          marginBottom: "2rem",
          backgroundColor: "rgba(25, 25, 25, 0.9)",
          borderRadius: "16px",
          padding: "1rem",
          boxShadow: "0 6px 20px rgba(0,0,0,0.6)",
        }}
      >
        {/* Playlist cover/spectrogram */}
        {spectrogram && (
          <img
            src={spectrogram}
            alt="Playlist Spectrogram"
            style={{
              width: "200px",
              borderRadius: "12px",
              cursor: "pointer",
              transition: "transform 0.2s",
            }}
            onClick={() => setExpandedSpectrogram(true)}
          />
        )}
        {/* Playlist info */}
        <div style={{ flex: 1 }}>
          <h2 style={{ fontSize: "2rem", margin: 0 }}>{playlist.playlist_title}</h2>
          <p style={{ fontSize: "1rem", marginTop: "0.5rem", lineHeight: 1.5 }}>
            This playlist was generated based on your uploaded audio. Each track is selected to match
            the mood and style of your song. Click the spectrogram to expand it for a detailed view of your song visualized.
          </p>
        </div>
      </div>

      {/* Expandable Spectrogram Modal */}
      {expandedSpectrogram && (
        <div
          onClick={() => setExpandedSpectrogram(false)}
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            backgroundColor: "rgba(0,0,0,0.8)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            zIndex: 1000,
            cursor: "zoom-out",
          }}
        >
          <img
            src={spectrogram}
            alt="Expanded Spectrogram"
            style={{ maxWidth: "90%", maxHeight: "90%", borderRadius: "12px" }}
          />
        </div>
      )}

      {/* Probability graph */}
      {probability_graph && (
        <div style={{ textAlign: "center", marginBottom: "2rem" }}>
          <h3 style={{ fontSize: "1.2rem", marginBottom: "0.5rem" }}>Probability Distribution</h3>
          <p style={{ fontSize: "0.9rem", color: "#ccc" }}>
            This graph shows the predicted probabilities for different musical categories.
          </p>
          <img
            src={probability_graph}
            alt="Probability Graph"
            style={{
              width: "90%",
              maxWidth: "800px",
              borderRadius: "12px",
              boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
            }}
          />
        </div>
      )}

      {/* Songs list */}
      <div>
        <h3 style={{ fontSize: "1.5rem", marginBottom: "1rem" }}>Tracks</h3>
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          {playlist.songs.map((song, index) => (
            <div
              key={index}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "1rem",
                backgroundColor: "rgba(30, 30, 30, 0.85)",
                borderRadius: "12px",
                padding: "0.75rem 1rem",
                boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
                transition: "background 0.2s",
              }}
            >
              {/* Album art (using spectrogram) */}
              {spectrogram && (
                <img
                  src={spectrogram}
                  alt="Album Art"
                  style={{ width: "60px", height: "60px", borderRadius: "8px" }}
                />
              )}

              {/* Song info */}
              <div style={{ flex: 1 }}>
                <p style={{ margin: 0, fontWeight: "600" }}>{song.title}</p>
              </div>

              {/* Audio player */}
              <audio controls src={song.path} style={{ width: "200px" }}></audio>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
