import { useState } from "react";
import * as tf from "@tensorflow/tfjs";
import Resizer from "react-image-file-resizer";
import styles from "./App.module.css";

const App = () => {
  const [emotion, setEmotion] = useState(null);
  const [imageURL, setImageURL] = useState(null);

  const emotionLabels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
  };

  const handleImageUpload = async (event) => {
    const imageFile = event.target.files[0];
    const imageURL = URL.createObjectURL(imageFile);
    setImageURL(imageURL);
    const resizedImage = await resizeImage(imageFile);
    const imageTensor = await preprocessAndCreateTensor(resizedImage);
    const model = await tf.loadLayersModel("./model.json");
    const prediction = model.predict(imageTensor);
    const predictionArray = await prediction.data();

    const predictedEmotionIndex = tf.argMax(predictionArray).dataSync()[0];
    const predictedEmotion = emotionLabels[predictedEmotionIndex];

    setEmotion(predictedEmotion);
  };

  const resizeImage = (file) => {
    return new Promise((resolve) => {
      Resizer.imageFileResizer(
        file,
        48,
        48,
        "JPEG",
        100,
        0,
        (resizedFile) => {
          resolve(resizedFile);
        },
        "blob",
        48,
        48
      );
    });
  };

  const preprocessAndCreateTensor = async (imageFile) => {
    const image = new Image();
    image.src = URL.createObjectURL(imageFile);

    return new Promise((resolve) => {
      image.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = 48;
        canvas.height = 48;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(image, 0, 0, 48, 48);
        const imageData = ctx.getImageData(0, 0, 48, 48);
        const data = new Float32Array(48 * 48); // Solo se necesita un valor por pixel
        for (let i = 0; i < imageData.data.length / 4; i++) {
          const offset = i * 4;
          const r = imageData.data[offset];
          const g = imageData.data[offset + 1];
          const b = imageData.data[offset + 2];
          const grayscale = (r + g + b) / 3.0;
          data[i] = grayscale; // Guarda solo el valor de escala de grises
        }
        const imageTensor = tf.tensor(data, [1, 48, 48, 1]);

        resolve(imageTensor);
      };
    });
  };

  return (
    <div className={styles.container}>
      <input
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        className={styles.input}
        id="fileInput"
      />

      <label htmlFor="fileInput" className={styles.label}>
        Upload Image
      </label>

      {/* Mostrar la emoción en el componente */}
      {emotion && <p>{emotion}</p>}
      {/* Mostrar la imagen cargada en el componente */}
      {imageURL && (
        <div className={styles.imgContainer}>
          <img src={imageURL} alt="User uploaded file" />
        </div>
      )}
    </div>
  );
};

export default App;
