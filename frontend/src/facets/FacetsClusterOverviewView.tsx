import React, { useEffect, useState } from 'react';
import { Card, Pagination, Spin, Select } from 'antd';

const FacetsClusterOverviewView: React.FC<{ state: any }> = ({ state }) => {
  const [clusters, setClusters] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [layout, setLayout] = useState('2x2');
  const [totalClusters, setTotalClusters] = useState(state.total_clusters);

  useEffect(() => {
    setLoading(true);
    fetch(`${state.project_name ? '/' + state.project_name : ''}/api/facets/${state.facet.id}/clusters?page=${page}&page_size=${pageSize}`)
      .then(res => res.json())
      .then(data => {
        setClusters(data.clusters);
        setTotalClusters(data.total);
        setLoading(false);
      });
  }, [state.facet.id, page, pageSize, state.project_name]);

  const getGridLayout = () => {
    switch (layout) {
      case '1x1': return { gridTemplateColumns: '1fr', limit: 1, cardWidth: 'calc(20% - 13px)' };
      case '3x2': return { gridTemplateColumns: '1fr 1fr 1fr', limit: 6, cardWidth: 'calc(25% - 12px)' };
      case '3x3': return { gridTemplateColumns: '1fr 1fr 1fr', limit: 9, cardWidth: 'calc(25% - 12px)' };
      case '2x2':
      default: return { gridTemplateColumns: '1fr 1fr', limit: 4, cardWidth: 'calc(20% - 13px)' };
    }
  };

  const gridConfig = getGridLayout();

  const getSlug = (feature_extractor_id: string) => {
    const parts = feature_extractor_id.split('/');
    return parts.length > 1 ? parts[1] : feature_extractor_id;
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 16, alignItems: 'center', gap: '16px' }}>
        <Pagination
          current={page}
          onChange={(p, s) => { setPage(p); setPageSize(s); }}
          onShowSizeChange={(_, size) => { setPageSize(size); setPage(1); }}
          total={totalClusters}
          pageSize={pageSize}
        />
        <Select value={layout} onChange={setLayout} style={{ width: 100 }}>
          <Select.Option value="1x1">1x1 Grid</Select.Option>
          <Select.Option value="2x2">2x2 Grid</Select.Option>
          <Select.Option value="3x2">3x2 Grid</Select.Option>
          <Select.Option value="3x3">3x3 Grid</Select.Option>
        </Select>
      </div>
      {loading ? <Spin /> : (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
          {clusters.map((cluster: any) => (
            <Card
              key={cluster.id}
              title={
                <div>
                  {cluster.cluster_label}
                  <div style={{ fontSize: '12px', fontWeight: 'normal', color: '#888' }}>
                    ({cluster.size} instance{cluster.size === 1 ? '' : 's'} in {cluster.unique_media_count || '?'} video{cluster.unique_media_count === 1 ? '' : 's'})
                  </div>
                </div>
              }
              style={{ width: gridConfig.cardWidth, minWidth: 300, cursor: 'pointer' }}
              hoverable
              onClick={() => {
                if (!window.__INITIAL_STATE__) {
                  window.location.href = `/facets.html?facet=${state.facet.name.toLowerCase()}&feature_extractor=${getSlug(state.facet.feature_extractor_id)}&cluster_id=${cluster.id}`;
                } else {
                  window.location.href = `${state.project_name ? '/' + state.project_name : ''}/facets/${state.facet.name.toLowerCase()}/${getSlug(state.facet.feature_extractor_id)}/cluster/${cluster.id}/`;
                }
              }}
            >
              <div style={{ display: 'grid', gridTemplateColumns: gridConfig.gridTemplateColumns, gap: '4px' }}>
                {cluster.representative_faces?.slice(0, gridConfig.limit).map((face: any, idx: number) => (
                  <div key={idx} style={{ position: 'relative', width: '100%', aspectRatio: '1 / 1', backgroundColor: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                    <div style={{ position: 'relative', width: '100%' }}>
                      <img
                        src={`${state.project_name ? '/' + state.project_name : ''}/thumbnail?media_id=${face.media_id}&timestamp=${face.timestamp}`}
                        style={{ width: '100%', height: 'auto', display: 'block' }}
                        alt="Representative Face"
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
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default FacetsClusterOverviewView;
