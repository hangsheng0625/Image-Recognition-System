export default function ResultsDisplay({ results }) {
    if (!results || !results.length) return null;
  
    return (
      <div style={{ marginTop: '2rem' }}>
        <h3>Similar Tiles Found:</h3>
        {/* Vertical stack instead of grid */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '1.5rem',
            marginTop: '1rem'
          }}
        >
          {results.map((result, index) => {
            // Extract filename from URL if available
            const fileName = result.imageUrl
              ? result.imageUrl.split('/').pop()
              : result.id || `Tile ${index + 1}`;
  
            return (
              <div
                key={index}
                style={{
                  textAlign: 'center',
                  border: '1px solid #eee',
                  borderRadius: '8px',
                  padding: '1rem',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                }}
              >
                <img
                  src={result.imageUrl}
                  alt={`Similar tile ${index + 1}`}
                  onError={(e) => {
                    console.error(`Failed to load image: ${result.imageUrl}`);
                    e.target.src =
                      'https://via.placeholder.com/150?text=Image+Not+Found';
                  }}
                  style={{
                    width: '100%',
                    height: '150px',
                    objectFit: 'cover',
                    borderRadius: '4px'
                  }}
                />
                <div style={{ marginTop: '0.8rem' }}>
                  <div
                    style={{
                      fontWeight: 'bold',
                      whiteSpace: 'normal',    // allow wrapping
                      wordWrap: 'break-word'   // wrap long filenames
                    }}
                  >
                    {fileName}
                  </div>
                  <div
                    style={{
                      marginTop: '0.5rem',
                      fontSize: '0.9em',
                      backgroundColor: '#f5f5f5',
                      padding: '4px 8px',
                      borderRadius: '4px',
                      display: 'inline-block'
                    }}
                  >
                    {(result.similarity * 100).toFixed(1)}% match
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }
  