import React, { useEffect, useState } from 'react';
import { Card, Button, Pagination, Spin, Select, message, Modal, Tooltip, Typography, Form, Input, Segmented } from 'antd';
import { PlusOutlined, CloseCircleFilled, MergeCellsOutlined, StarFilled, StarOutlined, QuestionCircleOutlined } from '@ant-design/icons';

const FacetView: React.FC<{ state: any }> = ({ state }) => {
  const [clusters, setClusters] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const getInitialPage = () => {
    const params = new URLSearchParams(window.location.search);
    const p = params.get('page');
    return p ? parseInt(p, 10) : 1;
  };

  const getInitialFilter = (param: string, defaultVal: string) => {
    const params = new URLSearchParams(window.location.search);
    return params.get(param) || defaultVal;
  };

  const [page, setPage] = useState(getInitialPage());
  const [pageSize, setPageSize] = useState(10);
  const [layout, setLayout] = useState('2x2');
  const [statusFilter, setStatusFilter] = useState(getInitialFilter('status_filter', 'All'));
  const [machineFeedbackFilter, setMachineFeedbackFilter] = useState(getInitialFilter('machine_feedback', 'All'));
  const [totalClusters, setTotalClusters] = useState(state.total_clusters);

  const [mergeQueue, setMergeQueue] = useState<any[]>(() => {
    const saved = sessionStorage.getItem(`merge_queue_${state.project_name}_${state.facet.id}`);
    try { return saved ? JSON.parse(saved) : []; } catch (e) { return []; }
  });

  useEffect(() => {
    sessionStorage.setItem(`merge_queue_${state.project_name}_${state.facet.id}`, JSON.stringify(mergeQueue));
  }, [mergeQueue, state.project_name, state.facet.id]);

  const [isMergeModalVisible, setIsMergeModalVisible] = useState(false);
  const [primaryClusterId, setPrimaryClusterId] = useState<number | null>(null);
  const [newClusterLabel, setNewClusterLabel] = useState('');

  useEffect(() => {
    setLoading(true);
    fetch(`/${state.project_name}/api/facet/${state.facet.id}/clusters?page=${page}&page_size=${pageSize}&status_filter=${statusFilter}&machine_feedback=${machineFeedbackFilter}`)
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
  }, [state.facet.id, page, pageSize, statusFilter, machineFeedbackFilter, state.project_name]);

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

  const handleAddToMergeQueue = (cluster: any) => {
    if (!mergeQueue.some(c => c.id === cluster.id)) {
      setMergeQueue([...mergeQueue, cluster]);
    }
  };

  const handleRemoveFromMergeQueue = (clusterId: number) => {
    setMergeQueue(mergeQueue.filter(c => c.id !== clusterId));
  };

  const handleInitiateMerge = () => {
    if (mergeQueue.length < 2) {
      message.error('Please select at least two clusters to merge.');
      return;
    }
    const largestCluster = mergeQueue.reduce((prev, current) => (prev.size > current.size) ? prev : current);
    setPrimaryClusterId(largestCluster.id);
    setNewClusterLabel(largestCluster.cluster_label || `Cluster ${largestCluster.id}`);
    setIsMergeModalVisible(true);
  };

  const handleConfirmMerge = () => {
    const secondary_cluster_ids = mergeQueue.filter(c => c.id !== primaryClusterId).map(c => c.id);
    const payload = { primary_cluster_id: primaryClusterId, secondary_cluster_ids, new_cluster_label: newClusterLabel };

    fetch(`/${state.project_name}/api/clusters/merge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(res => {
      if (res.ok) {
        message.success('Clusters merged successfully!');
        setMergeQueue([]);
        setIsMergeModalVisible(false);
        setLoading(true);
        fetch(`/${state.project_name}/api/facet/${state.facet.id}/clusters?page=${page}&page_size=${pageSize}&status_filter=${statusFilter}&machine_feedback=${machineFeedbackFilter}`)
          .then(r => r.json()).then(d => {
            if (Array.isArray(d)) {
              setClusters(d);
            } else {
              setClusters(d.clusters);
              setTotalClusters(d.total);
            }
            setLoading(false);
          });
      } else {
        message.error('Failed to merge clusters.');
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
    <div style={{ paddingBottom: mergeQueue.length > 0 ? '160px' : '0' }}>
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
        <span style={{ marginLeft: 16 }}>
          Machine Feedback
          <Tooltip title={
            <div style={{ fontSize: '12px' }}>
              <p><b>Uncertain Boundary:</b> The algorithm found ambiguous connections between known identities here. Needs manual review.</p>
              <p><b>Fragmented Ground Truth:</b> The algorithm disagrees with a 'Reviewed' cluster and believes it contains multiple distinct people.</p>
              <p><b>Identity Proposed:</b> The algorithm has found a high-confidence match based on a database of known identities. Click into the cluster to accept or reject the proposed name.</p>
            </div>
          }>
            <QuestionCircleOutlined style={{ marginLeft: 4, cursor: 'help' }} />
          </Tooltip>:
        </span>
        <Select value={machineFeedbackFilter} onChange={(val) => { setMachineFeedbackFilter(val); setPage(1); }} style={{ width: 220 }}>
          <Select.Option value="All">All Feedbacks</Select.Option>
          <Select.Option value="Uncertain Boundary">Uncertain Boundary</Select.Option>
          <Select.Option value="Fragmented Ground Truth">Fragmented Ground Truth</Select.Option>
          <Select.Option value="Identity Proposed">Identity Proposed</Select.Option>
        </Select>
      </div>

        {loading ? <Spin /> : (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', justifyContent: 'center' }}>
          {clusters.map((cluster: any) => (
            <Card
              key={cluster.id}
              title={
                <div>
                  {cluster.cluster_label || `Cluster ${cluster.id}`}
                  <div style={{ fontSize: '12px', fontWeight: 'normal', color: '#888' }}>
                    ({cluster.size} instances in {cluster.unique_media_count} videos)
                  </div>
                </div>
              }
              extra={
                <div onClick={(e) => e.stopPropagation()} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  {cluster.machine_feedback && (
                    <Tooltip title={
                      <div style={{ fontSize: '13px' }}>
                        <strong>Machine Feedback:</strong>
                        <ul style={{ margin: '4px 0 0 0', paddingLeft: '16px' }}>
                          {cluster.machine_feedback.split(', ').map((feedback: string) => (
                            <li key={feedback}>{feedback}</li>
                          ))}
                        </ul>
                      </div>
                    }>
                      <div style={{ cursor: 'help', fontSize: '18px', color: '#faad14', display: 'flex', alignItems: 'center' }}>
                        <svg viewBox="64 64 896 896" focusable="false" data-icon="robot" width="1em" height="1em" fill="currentColor" aria-hidden="true">
                          <path d="M840 478c0-4.4-3.6-8-8-8h-36v-66c0-39.8-32.2-72-72-72h-74V224c0-35.3-28.7-64-64-64H438c-35.3 0-64 28.7-64 64v108h-74c-39.8 0-72 32.2-72 72v66h-36c-4.4 0-8 3.6-8 8v164c0 4.4 3.6 8 8 8h36v122c0 39.8 32.2 72 72 72h424c39.8 0 72-32.2 72-72V650h36c4.4 0 8-3.6 8-8V478zM438 232h148v100H438V232zm316 540H270V396h484v376zm-388-212a40 40 0 1 1 80 0 40 40 0 1 1-80 0zm292 0a40 40 0 1 1 80 0 40 40 0 1 1-80 0z"></path>
                        </svg>
                      </div>
                    </Tooltip>
                  )}
                  <Tooltip title="Star this cluster so that its metadata can be updated in batch alongside all other starred clusters">
                    <div style={{ cursor: 'pointer', fontSize: '20px' }} onClick={(e) => { e.stopPropagation(); handleStarToggle(cluster.id, cluster.starred); }}>
                      {cluster.starred ? <StarFilled style={{ color: '#fadb14' }} /> : <StarOutlined />}
                    </div>
                  </Tooltip>
                  <Tooltip title="Add to Merge Queue">
                    <Button shape="circle" icon={<PlusOutlined />} onClick={(e) => { e.stopPropagation(); handleAddToMergeQueue(cluster); }} />
                  </Tooltip>
                </div>
              }
              style={{ width: gridConfig.cardWidth, minWidth: 250, cursor: 'pointer' }}
              hoverable
              onClick={() => window.location.href = `/${state.project_name}/explore/${state.facet.name.toLowerCase()}/${getSlug(state.facet.feature_extractor_id)}/cluster/${cluster.id}?from_page=${page}&status_filter=${encodeURIComponent(statusFilter)}&machine_feedback=${encodeURIComponent(machineFeedbackFilter)}`}
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

      {mergeQueue.length > 0 && (
        <div style={{ position: 'fixed', bottom: 0, left: 0, right: 0, background: '#f0f2f5', padding: '16px 24px', boxShadow: '0 -2px 8px rgba(0,0,0,0.1)', zIndex: 1000 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography.Title level={4} style={{ margin: 0 }}>Merge Queue ({mergeQueue.length})</Typography.Title>
            <div>
              <Button onClick={() => setMergeQueue([])} style={{ marginRight: 8 }}>Clear</Button>
              <Button type="primary" icon={<MergeCellsOutlined />} onClick={handleInitiateMerge}>Merge</Button>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '8px', marginTop: 12, overflowX: 'auto', paddingBottom: 8 }}>
            {mergeQueue.map(c => (
              <Tooltip key={c.id} title={`${c.cluster_label || `Cluster ${c.id}`} (${c.size} instances)`}>
                <Card size="small" style={{ flexShrink: 0, width: 200, position: 'relative' }}>
                  {c.cluster_label || `Cluster ${c.id}`}
                  <Button icon={<CloseCircleFilled />} size="small" shape="circle" style={{ position: 'absolute', top: 4, right: 4 }} onClick={() => handleRemoveFromMergeQueue(c.id)} />
                </Card>
              </Tooltip>
            ))}
          </div>
        </div>
      )}

      <Modal
        title="Confirm Cluster Merge"
        open={isMergeModalVisible}
        onOk={handleConfirmMerge}
        onCancel={() => setIsMergeModalVisible(false)}
        okText="Confirm Merge"
      >
        <p>Please select the primary cluster and confirm the new label.</p>
        <Form layout="vertical">
          <Form.Item label="Primary Cluster (to merge into)">
            <Select value={primaryClusterId} onChange={val => {
              setPrimaryClusterId(val);
              const selected = mergeQueue.find(c => c.id === val);
              if(selected) setNewClusterLabel(selected.cluster_label || `Cluster ${selected.id}`);
            }}>
              {mergeQueue.map(c => <Select.Option key={c.id} value={c.id}>{`${c.cluster_label || `Cluster ${c.id}`} (${c.size} instances)`}</Select.Option>)}
            </Select>
          </Form.Item>
          <Form.Item label="New Cluster Label">
            <Input value={newClusterLabel} onChange={e => setNewClusterLabel(e.target.value)} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default FacetView;
