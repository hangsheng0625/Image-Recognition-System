import React, { useState } from 'react';
import ImageUpload from './components/ImageUpload';
import ResultsDisplay from './components/ResultsDisplay';
import useTensorFlow from './hooks/useTensorflow';
import { Spin, Alert } from 'antd';

function App() {
  const { model, companyEmbeddings, loading: modelLoading, error } = useTensorFlow();
  const [results, setResults] = useState([]);
  const [processing, setProcessing] = useState(false);

  return (
    <div className="container" style={{ 
      maxWidth: '1200px', 
      margin: '0 auto', 
      padding: '2rem 1rem' 
    }}>
      <h1 style={{ textAlign: 'center', marginBottom: '2rem' }}>Tile Finder</h1>
      
      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          style={{ marginBottom: '2rem' }}
        />
      )}
      
      {modelLoading ? (
        <div style={{ textAlign: 'center', padding: '3rem' }}>
          <Spin size="large" />
          <div style={{ marginTop: '1rem' }}>Loading TensorFlow model...</div>
        </div>
      ) : (
        <>
          <ImageUpload 
            model={model} 
            embeddings={companyEmbeddings} 
            setResults={setResults}
            setLoading={setProcessing}
          />
          
          {processing ? (
            <div style={{ textAlign: 'center', padding: '2rem' }}>
              <Spin size="large" />
              <div style={{ marginTop: '1rem' }}>Processing image...</div>
            </div>
          ) : (
            <ResultsDisplay results={results} />
          )}
        </>
      )}
    </div>
  );
}

export default App;