export default function ResultsDisplay({ results }) {
    if (!results.length) return null;
  
    return (
      <div style={{ marginTop: '2rem' }}>
        <h3>Similar Tiles Found:</h3>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))',
          gap: '1rem',
          marginTop: '1rem'
        }}>
          {results.map((result, index) => (
            <div key={index} style={{ textAlign: 'center' }}>
              <img 
                src={result.imageUrl}
                alt="Similar tile"
                style={{ 
                  width: '100%', 
                  height: '150px',
                  objectFit: 'cover',
                  borderRadius: '4px'
                }}
              />
              <div style={{ marginTop: '0.5rem', fontSize: '0.9em' }}>
                {(result.similarity * 100).toFixed(1)}% match
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }