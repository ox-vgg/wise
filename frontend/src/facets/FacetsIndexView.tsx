import React from 'react';
import { Card } from 'antd';

const FacetsIndexView: React.FC<{ state: any }> = ({ state }) => {
  const getSlug = (feature_extractor_id: string) => {
    const parts = feature_extractor_id.split('/');
    return parts.length > 1 ? parts[1] : feature_extractor_id;
  };

  return (
    <div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
        {state.facets.map((facet: any) => (
          <Card
            key={facet.id}
            title={`${facet.name} Facet`}
            style={{ width: 300, cursor: 'pointer' }}
            hoverable
            onClick={() => {
              if (!window.__INITIAL_STATE__) {
                window.location.href = `/facets.html?facet=${facet.name.toLowerCase()}&feature_extractor=${getSlug(facet.feature_extractor_id)}`;
              } else {
                window.location.href = `${state.project_name ? '/' + state.project_name : ''}/facets/${facet.name.toLowerCase()}/${getSlug(facet.feature_extractor_id)}/`;
              }
            }}
          >
            {facet.preview_faces && facet.preview_faces.length > 0 && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px' }}>
                {facet.preview_faces.slice(0, 4).map((face: any, idx: number) => (
                  <div key={idx} style={{ position: 'relative', width: '100%', aspectRatio: '1 / 1', backgroundColor: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                    <div style={{ position: 'relative', width: '100%' }}>
                      <img
                        src={`${state.project_name ? '/' + state.project_name : ''}/thumbnail?media_id=${face.media_id}&timestamp=${face.timestamp}`}
                        style={{ width: '100%', height: 'auto', display: 'block' }}
                        alt="Preview Face"
                      />
                      {face.bbox && (
                        <div
                          style={{
                            position: 'absolute',
                            border: '2px solid yellow',
                            left: `${face.bbox.x * 100}%`,
                            top: `${face.bbox.y * 100}%`,
                            width: `${face.bbox.w * 100}%`,
                            height: `${face.bbox.h * 100}%`,
                            pointerEvents: 'none'
                          }}
                        />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
            <div style={{ textAlign: 'center', marginTop: '16px', color: '#888', fontSize: '12px' }}>
              Feature Extractor: {getSlug(facet.feature_extractor_id)}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default FacetsIndexView;