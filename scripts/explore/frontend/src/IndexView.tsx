import React from 'react';
import { Card } from 'antd';

const IndexView: React.FC<{ state: any }> = ({ state }) => {
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
            onClick={() => window.location.href = `/${state.project_name}/explore/${facet.name.toLowerCase()}/${getSlug(facet.feature_extractor_id)}/`}
          >
            <p>Feature extractor: {getSlug(facet.feature_extractor_id)}</p>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default IndexView;