import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button, Select, Space, Tooltip, message } from 'antd';
import { ReloadOutlined, UploadOutlined } from '@ant-design/icons';
import * as tf from '@tensorflow/tfjs';

export default function ImageUpload({ 
  model, 
  getImageEmbedding, 
  embeddings, 
  setResults, 
  setLoading,
  availableModels,
  selectedModelKey,
  setSelectedModelKey,
  isModelLoading,
  generateEmbeddings,
  isGeneratingEmbeddings
}) {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [imageName, setImageName] = useState('');
  const { getRootProps, getInputProps } = useDropzone({
    accept: {'image/*': []},
    multiple: false,
    onDrop: async ([file]) => {
      if (!model) {
        message.error('Model not loaded yet');
        return;
      }
      
      if (embeddings.length === 0) {
        message.warning('No reference embeddings available. Please click "Generate Memory" first.');
        return;
      }
      
      try {
        setLoading(true);
        
        // Store the image preview and name
        setUploadedImage(URL.createObjectURL(file));
        setImageName(file.name);
        
        const img = await loadImage(file);
        const embedding = await getImageEmbedding(img);
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
  
  // Handle model change
  const handleModelChange = (value) => {
    setResults([]);  // Clear previous results
    setUploadedImage(null);  // Clear uploaded image
    setSelectedModelKey(value);
  };
  
  // Handle generate embeddings click
  const handleGenerateClick = async () => {
    if (isGeneratingEmbeddings) return;
    
    try {
      setResults([]);  // Clear previous results
      await generateEmbeddings();
      message.success(`Memory generated successfully using ${availableModels.find(m => m.key === selectedModelKey).name}`);
    } catch (error) {
      console.error('Error generating embeddings:', error);
      message.error('Failed to generate memory');
    }
  };
  
  return (
    <div>
      {/* Model selection and memory generation controls */}
      <div style={{ 
        marginBottom: '2rem', 
        padding: '1rem', 
        background: '#f5f5f5', 
        borderRadius: '8px' 
      }}>
        <h3>Model Selection</h3>
        <Space direction="vertical" style={{ width: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <Select
              style={{ width: '200px' }}
              value={selectedModelKey}
              onChange={handleModelChange}
              loading={isModelLoading}
              disabled={isModelLoading || isGeneratingEmbeddings}
              options={availableModels.map(model => ({
                value: model.key,
                label: model.name
              }))}
            />
            {/* <Tooltip title="Generate embeddings for all reference images using the selected model">
              <Button 
                type="primary" 
                onClick={handleGenerateClick} 
                loading={isGeneratingEmbeddings}
                disabled={isModelLoading || !model}
                icon={<ReloadOutlined />}
              >
                Generate Memory
              </Button>
            </Tooltip> */}
          </div>
          
          <div style={{ fontSize: '0.9rem', color: '#666' }}>
            {isModelLoading ? (
              'Loading model...'
            ) : model ? (
              `${availableModels.find(m => m.key === selectedModelKey).name} model loaded`
            ) : (
              'No model loaded'
            )}
            {embeddings.length > 0 && (
              `, ${embeddings.length} images in memory`
            )}
          </div>
        </Space>
      </div>
      
      {/* Dropzone for image upload */}
      <div {...getRootProps()} style={{
        border: '2px dashed #1890ff',
        borderRadius: '8px',
        padding: '2rem',
        textAlign: 'center',
        cursor: 'pointer',
        marginBottom: '2rem',
        backgroundColor: embeddings.length === 0 ? '#f5f5f5' : 'white',
      }}>
        <input {...getInputProps()} />
        <Button 
          type="primary" 
          size="large" 
          icon={<UploadOutlined />}
          disabled={!model || embeddings.length === 0}
        >
          Upload Tile Image
        </Button>
        <p style={{ marginTop: '1rem', color: '#666' }}>
          {!model ? (
            'Please wait for model to load'
          ) : embeddings.length === 0 ? (
            'Please generate memory first'
          ) : (
            'Drag & drop or click to select'
          )}
        </p>
      </div>
      
      {/* Uploaded image preview */}
      {uploadedImage && (
        <div style={{ marginBottom: '2rem', textAlign: 'center' }}>
          <h3>Uploaded Image:</h3>
          <img 
            src={uploadedImage}
            alt="Uploaded tile"
            style={{
              maxWidth: '100%',
              maxHeight: '200px',
              objectFit: 'contain',
              borderRadius: '4px',
              marginTop: '1rem'
            }}
          />
          <p style={{ marginTop: '0.5rem', fontWeight: 'bold' }}>{imageName}</p>
        </div>
      )}
    </div>
  );
}

// Helper functions
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

function findSimilar(inputEmbedding, embeddings, topK = 5) {
  if (!embeddings || !embeddings.length) {
    console.error('No embeddings available');
    return [];
  }
  
  const similarityArray = embeddings.map(emb => {
    // Add more validation
    if (!emb.features || !Array.isArray(emb.features)) {
      console.error('Invalid embedding features for', emb.id);
      return { ...emb, similarity: 0 };
    }
    
    const similarity = cosineSimilarity(inputEmbedding, emb.features);
    
    return { ...emb, similarity };
  });
  
  // Sort by similarity descending and take topK
  const results = similarityArray
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topK);
  
  return results;
}

function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) {
    console.error(`Invalid vectors for similarity calculation: a=${a?.length}, b=${b?.length}`);
    return 0;
  }
  
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  
  if (normA === 0 || normB === 0) return 0;
  return dot / (normA * normB);
}