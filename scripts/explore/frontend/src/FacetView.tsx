import React, { useEffect, useState } from 'react';
import { Card, Tag, Button, Pagination, Spin, Select, message, Segmented, Tooltip } from 'antd';
import { StarOutlined, StarFilled } from '@ant-design/icons';

const FacetView: React.FC<{ state: any }> = ({ state }) => {
  const [clusters, setClusters] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  
  const getInitialPage = () => {
    const params = new URLSearchParams(window.location.search);
    const p = params.get('page');
    return p ? parseInt(p, 10) : 1;
  };
  
  const [page, setPage] = useState(getInitialPage());
  const [pageSize, setPageSize] = useState(10);
  const [layout, setLayout] = useState('2x2');
  const [statusFilter, setStatusFilter] = useState('All');
  const [totalClusters, setTotalClusters] = useState(state.total_clusters);

  useEffect(() => {
    setLoading(true);
    fetch(`/${state.project_name}/api/facet/${state.facet.id}/clusters?page=${page}&page_size=${pageSize}&status_filter=${statusFilter}`)
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) {
          setClusters(data);
        } else {
          setClusters(data.clusters);
          setTotalClusters(data.total);
        }
        setLoading(false);
      });
  }, [state.facet.id, page, pageSize, statusFilter, state.project_name]);

  const handleStatusChange = (clusterId: number, newStatus: string) => {
    fetch(`/${state.project_name}/api/cluster/${clusterId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: newStatus })
    }).then(res => {
      if(res.ok) {
        setClusters(prevClusters => 
          prevClusters.map(c => c.id === clusterId ? { ...c, status: newStatus } : c)
        );
        message.success('Status updated');
      }
    });
  };

  const handleStarToggle = (clusterId: number, currentStarred: boolean) => {
    fetch(`/${state.project_name}/api/cluster/${clusterId}/star`, {
      method: 'POST'
    }).then(res => {
      if(res.ok) {
        setClusters(prevClusters => 
          prevClusters.map(c => c.id === clusterId ? { ...c, starred: !currentStarred } : c)
        );
        message.success(currentStarred ? 'Removed from starred' : 'Added to starred');
      }
    });
  };

  const getSlug = (feature_extractor_id: string) => {
    const parts = feature_extractor_id.split('/');
    return parts.length > 1 ? parts[1] : feature_extractor_id;
  };

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

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 16, alignItems: 'center', gap: '16px' }}>
        <Pagination 
          current={page} 
          onChange={(p, s) => { setPage(p); setPageSize(s); }} 
          onShowSizeChange={(current, size) => { setPageSize(size); setPage(1); }}
          total={totalClusters} 
          pageSize={pageSize}
        />
        <Select value={layout} onChange={setLayout} style={{ width: 100 }}>
          <Select.Option value="1x1">1x1 Grid</Select.Option>
          <Select.Option value="2x2">2x2 Grid</Select.Option>
          <Select.Option value="3x2">3x2 Grid</Select.Option>
          <Select.Option value="3x3">3x3 Grid</Select.Option>
        </Select>
        <span style={{ marginLeft: 16 }}>Show:</span>
        <Select value={statusFilter} onChange={(val) => { setStatusFilter(val); setPage(1); }} style={{ width: 120 }}>
          <Select.Option value="All">All</Select.Option>
          <Select.Option value="draft">Drafts</Select.Option>
          <Select.Option value="reviewed">Reviewed</Select.Option>
          <Select.Option value="published">Published</Select.Option>
          <Select.Option value="starred">Starred</Select.Option>
        </Select>
      </div>
      {loading ? <Spin /> : (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
          {clusters.map((cluster: any) => (
            <Card 
              key={cluster.id} 
              title={`${cluster.cluster_label} (${cluster.size} ${cluster.size === 1 ? 'instance' : 'instances'})`} 
              extra={
                <div onClick={(e) => e.stopPropagation()} style={{ display: 'flex', alignItems: 'center' }}>
                  <Tooltip title="Star this cluster so that its metadata can be updated in batch alongside all other starred clusters">
                    <div style={{ cursor: 'pointer', fontSize: '20px' }} onClick={() => handleStarToggle(cluster.id, cluster.starred)}>
                      {cluster.starred ? <StarFilled style={{ color: '#fadb14' }} /> : <StarOutlined />}
                    </div>
                  </Tooltip>
                </div>
              }
              style={{ width: gridConfig.cardWidth, minWidth: 250, cursor: 'pointer' }}
              hoverable
              onClick={() => window.location.href = `/${state.project_name}/explore/${state.facet.name.toLowerCase()}/${getSlug(state.facet.feature_extractor_id)}/cluster/${cluster.id}?from_page=${page}`}
            >
              <div style={{ display: 'grid', gridTemplateColumns: gridConfig.gridTemplateColumns, gap: '4px' }}>
                {cluster.representative_faces?.slice(0, gridConfig.limit).map((face: any, idx: number) => (
                  <div key={idx} style={{ position: 'relative', width: '100%', aspectRatio: '1 / 1', backgroundColor: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                    <div style={{ position: 'relative', width: '100%' }}>
                      <img 
                        src={`/${state.project_name}/api/thumbnail?media_id=${face.media_id}&timestamp=${face.timestamp}`}
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
              <div onClick={(e) => e.stopPropagation()} style={{ marginTop: '12px', display: 'flex', justifyContent: 'center' }}>
                <style>{`
                  .custom-segmented .ant-segmented-item-selected {
                    background-color: #bae0ff !important;
                    font-weight: 500;
                  }
                `}</style>
                <Segmented 
                  className="custom-segmented"
                  value={cluster.status} 
                  onChange={(val) => handleStatusChange(cluster.id, val as string)}
                  options={[
                    { label: 'Draft', value: 'draft' },
                    { label: 'Reviewed', value: 'reviewed' },
                    { label: 'Published', value: 'published' }
                  ]}
                  size="small"
                />
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default FacetView;
