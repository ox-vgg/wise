import React, { useEffect, useState } from 'react';
import { Card, Pagination, Spin, Descriptions, Modal } from 'antd';

const VideoPlayerWithPoster: React.FC<{ project_name: string, face: any }> = ({ project_name, face }) => {
  const baseUrl = project_name ? `/${project_name}` : '';
  const poster = `${baseUrl}/thumbnail?media_id=${face.media_id}&timestamp=${face.timestamp}`;
  const [hasPlayed, setHasPlayed] = useState(false);

  return (
    <div style={{ position: 'relative', width: '100%', backgroundColor: '#000' }}>
      <video
        controls
        src={`${baseUrl}/media/${face.media_id}#t=${face.timestamp}`}
        poster={poster}
        style={{ width: '100%', height: 'auto', display: 'block' }}
        autoPlay={false}
        onPlay={() => setHasPlayed(true)}
      />
      {face.bbox && !hasPlayed && (
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
  );
};

const FacetsClusterDetailView: React.FC<{ state: any }> = ({ state }) => {
  const [faces, setFaces] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [selectedFace, setSelectedFace] = useState<any>(null);

  useEffect(() => {
    setLoading(true);
    fetch(`${state.project_name ? '/' + state.project_name : ''}/api/facets/cluster/${state.cluster.id}/faces?page=${page}&page_size=${pageSize}`)
      .then(res => res.json())
      .then(data => {
        setFaces(data);
        setLoading(false);
      });
  }, [state.cluster.id, page, pageSize, state.project_name]);

  return (
    <div>
      {state.cluster.metadata && Object.keys(state.cluster.metadata).length > 0 && (
        <Card style={{ marginBottom: 16 }}>
          <Descriptions size="small" column={1} bordered>
            {Object.entries(state.cluster.metadata).map(([key, value]) => {
              const strValue = String(value);
              const isUrl = strValue.startsWith('http://') || strValue.startsWith('https://');
              return (
                <Descriptions.Item key={key} label={<span style={{ textTransform: 'capitalize' }}>{key.replace(/_/g, ' ')}</span>}>
                  {isUrl ? (
                    <a href={strValue} target="_blank" rel="noopener noreferrer">
                      {strValue}
                    </a>
                  ) : (
                    strValue
                  )}
                </Descriptions.Item>
              );
            })}
          </Descriptions>
        </Card>
      )}

      <Card
        title={
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <Pagination
              current={page}
              onChange={(p, s) => { setPage(p); setPageSize(s); }}
              onShowSizeChange={(_, size) => { setPageSize(size); setPage(1); }}
              total={state.cluster.size}
              pageSize={pageSize}
              size="small"
              style={{ margin: 0 }}
            />
          </div>
        }
      >
        {loading ? <Spin /> : (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {faces.map((face, idx) => (
              <Card key={idx} hoverable bodyStyle={{ padding: 0 }} onClick={() => setSelectedFace(face)}>
                <div style={{ position: 'relative', width: 120, height: 120, backgroundColor: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                  <div style={{ position: 'relative', width: '100%' }}>
                    <img
                      src={`${state.project_name ? '/' + state.project_name : ''}/thumbnail?media_id=${face.media_id}&timestamp=${face.timestamp}`}
                      style={{ width: '100%', height: 'auto', display: 'block' }}
                      alt="face"
                      title={`Vector ID: ${face.vector_id}`}
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
              </Card>
            ))}
          </div>
        )}
      </Card>

      <Modal
        title={selectedFace?.filename || "Video Frame Context"}
        open={!!selectedFace}
        onCancel={() => setSelectedFace(null)}
        footer={null}
        width={800}
        destroyOnClose
      >
        {selectedFace && <VideoPlayerWithPoster project_name={state.project_name} face={selectedFace} />}
      </Modal>
    </div>
  );
};

export default FacetsClusterDetailView;