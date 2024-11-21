import React, { useState } from "react";

function App() {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);

  const generateImages = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/generate/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ num_images: 1 }),
      });

      const data = await response.json();
      setImages(data.images); // Les images au format base64
    } catch (error) {
      console.error("Erreur lors de la génération d'images :", error);
    }
    setLoading(false);
  };

  return (
    <div style={{ textAlign: "center" }}>
      <h1>GAN Image Generator</h1>
      <button onClick={generateImages} disabled={loading}>
        {loading ? "Loading..." : "Generate Images"}
      </button>
      <div style={{ display: "flex", flexWrap: "wrap", marginTop: 20 }}>
        {images.map((img, index) => (
          <img
            key={index}
            src={`data:image/png;base64,${img}`}
            alt="Generated"
            style={{
              width: "256px", // Taille ajustée pour 128x128 (augmentation à 256px pour affichage)
              height: "256px", // Hauteur ajustée pour 128x128
              margin: "10px",
              objectFit: "cover", // Assurer que l'image occupe toute la zone sans déformation
            }}
          />
        ))}
      </div>
    </div>
  );
}

export default App;
