import React from 'react';
import { useDropzone } from 'react-dropzone';
import { Button, message } from 'antd';
import * as tf from '@tensorflow/tfjs';

export default function ImageUpload({ model, embeddings, setResults, setLoading }) {
  const { getRootProps, getInputProps } = useDropzone({
    accept: {'image/*': []},
    multiple: false,
    onDrop: async ([file]) => {
      if (!model) {
        message.error('Model not loaded yet');
        return;
      }
      
      try {
        setLoading(true);
        const img = await loadImage(file);
        const embedding = await getEmbedding(model, img);
        const results = findSimilar(embedding, embeddings);
        setResults(results);
      } catch (error) {
        console.error('Error processing image:', error);
        message.error('Error processing image');
      } finally {
        setLoading(false);
      }
    }
  });
  
  return (
    <div {...getRootProps()} style={{
      border: '2px dashed #1890ff',
      borderRadius: '8px',
      padding: '2rem',
      textAlign: 'center',
      cursor: 'pointer',
      marginBottom: '2rem'
    }}>
      <input {...getInputProps()} />
      <Button type="primary" size="large">
        Upload Tile Image
      </Button>
      <p style={{ marginTop: '1rem', color: '#666' }}>
        Drag & drop or click to select
      </p>
    </div>
  );
}

async function loadImage(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.src = e.target.result;
      img.onload = () => resolve(img);
    };
    reader.readAsDataURL(file);
  });
}

async function getEmbedding(model, img) {
  const tensor = tf.browser.fromPixels(img)
    .resizeBilinear([224, 224])
    .div(255)
    .expandDims(0);
    
  // For MobileNet, use predict instead of infer
  const activation = model.infer(tensor);
  const embedding = await activation.data();
  
  // Clean up tensors
  tensor.dispose();
  activation.dispose();
  
  return Array.from(embedding);
}

function findSimilar(inputEmbedding, embeddings, topK = 5) {
  if (!embeddings || !embeddings.length) {
    console.error('No embeddings available');
    return [];
  }
  
  return embeddings
    .map(emb => ({
      ...emb,
      similarity: cosineSimilarity(inputEmbedding, emb.features)
    }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topK);
}

function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) {
    console.error('Invalid vectors for similarity calculation');
    return 0;
  }
  
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  
  if (normA === 0 || normB === 0) return 0;
  return dot / (normA * normB);
}