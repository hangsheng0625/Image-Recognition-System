import React, { useState } from 'react';
import { Layout, Spin, Typography } from 'antd';
import ImageUpload from './components/ImageUpload';
import ResultsDisplay from './components/ResultsDisplay';
import useModelSelector from './hooks/useModelSelector.jsx';

const { Header, Content, Footer } = Layout;
const { Title } = Typography;

function App() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  
  // Use our custom hook for model selection and management
  const {
    model,
    selectedModelKey,
    setSelectedModelKey,
    isModelLoading,
    embeddings,
    isGeneratingEmbeddings,
    generateEmbeddings,
    getImageEmbedding,
    availableModels
  } = useModelSelector();

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#fff', padding: '0 2rem' }}>
        <Title level={3} style={{ margin: '1rem 0' }}>
          Image Recognition System
        </Title>
      </Header>
      
      <Content style={{ padding: '2rem' }}>
        <div style={{ background: '#fff', padding: '2rem', borderRadius: '4px' }}>
          <ImageUpload
            model={model}
            getImageEmbedding={getImageEmbedding}
            embeddings={embeddings}
            setResults={setResults}
            setLoading={setLoading}
            availableModels={availableModels}
            selectedModelKey={selectedModelKey}
            setSelectedModelKey={setSelectedModelKey}
            isModelLoading={isModelLoading}
            generateEmbeddings={generateEmbeddings}
            isGeneratingEmbeddings={isGeneratingEmbeddings}
          />
          
          {loading ? (
            <div style={{ textAlign: 'center', margin: '3rem 0' }}>
              <Spin size="large" />
              <p style={{ marginTop: '1rem' }}>Processing image...</p>
            </div>
          ) : (
            <ResultsDisplay results={results} />
          )}
        </div>
      </Content>
      
      <Footer style={{ textAlign: 'center' }}>
        Tile Recognition System using TensorFlow.js and Pre-trained Models
      </Footer>
    </Layout>
  );
}

export default App;