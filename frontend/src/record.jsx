import { useState, useRef } from "react";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import ClipLoader from "react-spinners/ClipLoader"; 
import Playlist from "./playlist/playlist"; // import the Playlist component

export default function LandingPage() {
  const [audioFile, setAudioFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [recording, setRecording] = useState(false);
  const [playlistData, setPlaylistData] = useState(null);
  const [processing, setProcessing] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunks = useRef([]);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAudioFile(file);
      setAudioUrl(URL.createObjectURL(file));
    }
  };

  const recommendPlaylist = async (file) => {
    setProcessing(true);
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://localhost:8000/recommend", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();
    console.log(result);
    setPlaylistData(result);
    setProcessing(false);
  };

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new MediaRecorder(stream);
    audioChunks.current = [];

    mediaRecorderRef.current.ondataavailable = (event) => {
      if (event.data.size > 0) audioChunks.current.push(event.data);
    };

    mediaRecorderRef.current.onstop = () => {
      const audioBlob = new Blob(audioChunks.current, { type: "audio/webm" });
      setAudioUrl(URL.createObjectURL(audioBlob));
      setAudioFile(new File([audioBlob], "recording.webm", { type: "audio/webm" }));
    };

    mediaRecorderRef.current.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    setRecording(false);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        background: "linear-gradient(135deg, #1db954 0%, #191414 100%)",
        flexDirection: "column",
        padding: "2rem",
      }}
    >
      {/* Playlist display */}
      {playlistData && <Playlist data={playlistData} />}

      {/* Upload/Record Card */}
      <Card
        style={{
          width: "100%",
          maxWidth: 480,
          padding: "2rem",
          borderRadius: "2rem",
          backgroundColor: "rgba(40, 40, 40, 0.85)",
          color: "#fff",
          boxShadow: "0 16px 40px rgba(0,0,0,0.6)",
          backdropFilter: "blur(10px)",
          position: "relative",
          marginTop: "2rem",
        }}
      >
        {processing && (
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: "rgba(0,0,0,0.6)",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              borderRadius: "2rem",
              zIndex: 10,
              flexDirection: "column",
            }}
          >
            <ClipLoader color="#1db954" size={60} />
            <p style={{ color: "#fff", marginTop: "1rem" }}>Processing...</p>
          </div>
        )}

        <CardContent>
          <h1
            style={{
              fontSize: "1.8rem",
              fontWeight: "700",
              marginBottom: "1.5rem",
              textAlign: "center",
            }}
          >
            Upload or Record Your Song
          </h1>

          {/* Upload */}
          <div style={{ marginBottom: "1.5rem" }}>
            <input
              type="file"
              accept="audio/mp3"
              onChange={handleFileUpload}
              style={{
                width: "100%",
                padding: "0.5rem",
                borderRadius: "0.5rem",
                border: "none",
                backgroundColor: "#404040",
                color: "#fff",
              }}
            />
          </div>

          {/* Record */}
          <div style={{ marginBottom: "1.5rem", textAlign: "center" }}>
            {!recording ? (
              <Button
                variant="contained"
                style={{
                  backgroundColor: "#1db954",
                  color: "#fff",
                  borderRadius: "999px",
                  textTransform: "none",
                  fontWeight: "600",
                  padding: "0.75rem 2rem",
                }}
                onClick={startRecording}
              >
                🎤 Start Recording
              </Button>
            ) : (
              <Button
                variant="contained"
                style={{
                  backgroundColor: "#e02424",
                  color: "#fff",
                  borderRadius: "999px",
                  textTransform: "none",
                  fontWeight: "600",
                  padding: "0.75rem 2rem",
                }}
                onClick={stopRecording}
              >
                ⏹ Stop Recording
              </Button>
            )}
          </div>

          {/* Preview */}
          {audioUrl && (
            <div style={{ marginTop: "1rem", textAlign: "center" }}>
              <p style={{ marginBottom: "0.5rem" }}>Preview:</p>
              <audio controls src={audioUrl} style={{ width: "100%" }}></audio>
            </div>
          )}

          {/* Submit */}
          {audioFile && (
            <div style={{ marginTop: "1.5rem", textAlign: "center" }}>
              <Button
                variant="contained"
                style={{
                  backgroundColor: "#1db954",
                  color: "#fff",
                  width: "100%",
                  padding: "0.75rem",
                  fontWeight: "600",
                  borderRadius: "999px",
                  textTransform: "none",
                }}
                onClick={() => recommendPlaylist(audioFile)}
              >
                Upload & Build Playlist
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
