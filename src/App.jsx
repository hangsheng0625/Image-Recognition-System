import { useState } from 'react';
import { Layout, Spin, Alert } from 'antd';
import ImageUpload from './components/ImageUpload';
import ResultsDisplay from './components/ResultsDisplay';
import useTensorFlow from './hooks/useTensorFlow';

const { Content } = Layout;

export default function App() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const { model, companyEmbeddings, loading: modelLoading, error } = useTensorFlow();

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Content style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto' }}>
        <h1 style={{ textAlign: 'center' }}>Tile Finder</h1>
        
        {error && (
          <Alert
            message="Error"
            description={error}
            type="error"
            showIcon
            style={{ marginBottom: '1rem' }}
          />
        )}
        
        {modelLoading ? (
          <div style={{ textAlign: 'center', marginTop: '2rem' }}>
            <Spin size="large" />
            <p style={{ marginTop: '1rem' }}>Loading TensorFlow model...</p>
          </div>
        ) : (
          <>
            <ImageUpload
              model={model}
              embeddings={companyEmbeddings}
              setResults={setResults}
              setLoading={setLoading}
            />
            
            {loading ? (
              <div style={{ textAlign: 'center', marginTop: '2rem' }}>
                <Spin size="large" />
                <p>Processing image...</p>
              </div>
            ) : (
              <ResultsDisplay results={results} />
            )}
          </>
        )}
      </Content>
    </Layout>
  );
}