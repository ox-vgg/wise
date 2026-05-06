import React, { useEffect, useState } from 'react';
import { Card, Button, Spin, Collapse, Modal, Typography, Form, Input, Select, message, Segmented, Tooltip } from 'antd';
import { EditOutlined, SaveOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';

const VideoPlayerWithPoster: React.FC<{ project_name: string, face: any }> = ({ project_name, face }) => {
  const [poster, setPoster] = useState(`/${project_name}/api/thumbnail?media_id=${face.media_id}&timestamp=${face.timestamp}`);
  const [hasPlayed, setHasPlayed] = useState(false);

  useEffect(() => {
    const highResUrl = `/${project_name}/api/thumbnail?media_id=${face.media_id}&timestamp=${face.timestamp}&high_res=true`;
    const img = new Image();
    img.src = highResUrl;
    img.onload = () => setPoster(highResUrl);
  }, [face.media_id, face.timestamp, project_name]);

  return (
    <div style={{ position: 'relative', width: '100%', backgroundColor: '#000' }}>
      <video
        controls
        src={`/${project_name}/api/media/${face.media_id}#t=${face.timestamp}`}
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

const ClusterView: React.FC<{ state: any }> = ({ state }) => {
  const [loading, setLoading] = useState(true);
  const [selectedFace, setSelectedFace] = useState<any>(null);
  const [groupedFaces, setGroupedFaces] = useState<any>({});

  const [clusterLabel, setClusterLabel] = useState(state.cluster.cluster_label);
  const [isEditingLabel, setIsEditingLabel] = useState(false);
  const [status, setStatus] = useState(state.cluster.status);
  const [metadataJson, setMetadataJson] = useState<any>(state.cluster.metadata || {});
  const [schema, setSchema] = useState<any[]>([]);
  const [isSchemaModalOpen, setIsSchemaModalOpen] = useState(false);
  const [schemaForm] = Form.useForm();

  useEffect(() => {
    fetch(`/${state.project_name}/api/cluster/${state.cluster.id}/faces_by_media`)
      .then(res => res.json())
      .then(data => {
        setGroupedFaces(data);
        setLoading(false);
      });

    fetch(`/${state.project_name}/api/facet/${state.facet.id}/schema`)
      .then(res => res.json())
      .then(data => setSchema(data));
  }, [state.cluster.id, state.project_name, state.facet.id]);

  const handleSaveAll = () => {
    const payload = {
      cluster_label: clusterLabel,
      status: status,
      metadata_json: metadataJson
    };
    fetch(`/${state.project_name}/api/cluster/${state.cluster.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(res => {
      if (res.ok) {
        message.success('All changes saved successfully!');
      } else {
        message.error('Failed to save changes.');
      }
    });
  };

  const handleAddSchemaField = () => {
    schemaForm.validateFields().then(values => {
      fetch(`/${state.project_name}/api/facet/${state.facet.id}/schema`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      }).then(res => res.json()).then(newField => {
        setSchema([...schema, newField]);
        setIsSchemaModalOpen(false);
        schemaForm.resetFields();
      });
    });
  };

  const handleAcceptProposal = () => {
    const proposal = metadataJson._proposed_identity;
    if (!proposal) return;

    const newMetadata = { ...metadataJson };
    delete newMetadata._proposed_identity;

    // Move proposed metadata into actual metadata
    Object.keys(proposal).forEach(key => {
        if (key !== 'similarity' && key !== 'name') {
            newMetadata[key] = proposal[key];
        }
    });

    setClusterLabel(proposal.name);
    setMetadataJson(newMetadata);
    setStatus('reviewed'); // Automatically mark as reviewed when a proposal is accepted

    // Save immediately
    fetch(`/${state.project_name}/api/cluster/${state.cluster.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        cluster_label: proposal.name,
        metadata_json: newMetadata,
        status: 'reviewed'
      })
    }).then(res => {
      if (res.ok) {
        message.success(`Identity resolved to ${proposal.name}!`);
      }
    });
  };

  const handleRejectProposal = () => {
    const newMetadata = { ...metadataJson };
    delete newMetadata._proposed_identity;
    setMetadataJson(newMetadata);

    fetch(`/${state.project_name}/api/cluster/${state.cluster.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        metadata_json: newMetadata
      })
    }).then(res => {
      if (res.ok) {
        message.info('Proposal rejected.');
      }
    });
  };

  return (
    <div>
      {metadataJson._proposed_identity && (
        <Card style={{ marginBottom: 16, backgroundColor: '#f6ffed', borderColor: '#ffe58f' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                    <Typography.Title level={4} style={{ margin: 0, color: '#faad14' }}>
                        Identity Proposal: {metadataJson._proposed_identity.name}
                    </Typography.Title>
                    <div style={{ margin: '4px 0 0 0', color: '#666', fontSize: '13px' }}>
                        <strong>Match Confidence:</strong> {(metadataJson._proposed_identity.similarity * 100).toFixed(1)}% <br/>
                        {Object.entries(metadataJson._proposed_identity).map(([key, value]) => {
                          if (key === 'similarity' || key === 'name' || value === null || value === '') return null;
                          const strValue = String(value);
                          const isUrl = strValue.startsWith('http://') || strValue.startsWith('https://');
                          return (
                            <span key={key} style={{ marginRight: '16px' }}>
                              <strong>{key.replace(/_/g, ' ')}:</strong> {isUrl ? <a href={strValue} target="_blank" rel="noreferrer">Link</a> : strValue}
                            </span>
                          );
                        })}
                    </div>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                    <Button type="primary" style={{ backgroundColor: '#52c41a' }} icon={<CheckCircleOutlined />} onClick={handleAcceptProposal}>
                        Accept
                    </Button>
                    <Button danger icon={<CloseCircleOutlined />} onClick={handleRejectProposal}>
                        Reject
                    </Button>
                </div>
            </div>
        </Card>
      )}

      <Card style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            {isEditingLabel ? (
              <Input
                value={clusterLabel}
                onChange={(e) => setClusterLabel(e.target.value)}
                onPressEnter={() => setIsEditingLabel(false)}
                style={{ fontSize: 24, width: 300 }}
              />
            ) : (
              <Typography.Title level={2} style={{ margin: 0 }}>
                {clusterLabel || `Cluster ${state.cluster.id}`}
                <Tooltip title="Edit Label">
                  <Button type="text" icon={<EditOutlined />} onClick={() => setIsEditingLabel(true)} />
                </Tooltip>
              </Typography.Title>
            )}
            <Segmented value={status} onChange={(val) => setStatus(val as string)} options={['draft', 'reviewed', 'published']} style={{ marginTop: 8 }} />
          </div>
          <div style={{ flex: 1, marginLeft: 24, maxWidth: 600 }}>
            <Form layout="vertical">
              {schema.map(field => (
                <Form.Item label={field.key_name} key={field.id}>
                  <Input
                    value={metadataJson[field.key_name] || ''}
                    onChange={e => setMetadataJson({...metadataJson, [field.key_name]: e.target.value})}
                  />
                </Form.Item>
              ))}
            </Form>
            <Button onClick={() => setIsSchemaModalOpen(true)}>Add Metadata Field</Button>
          </div>
          <div>
            <Button type="primary" icon={<SaveOutlined />} onClick={handleSaveAll}>Save All Changes</Button>
          </div>
        </div>
      </Card>

      <Card>
        {loading ? <Spin /> : (
          <Collapse accordion>
            {Object.entries(groupedFaces || {}).sort(([, a]: [string, any], [, b]: [string, any]) => b.faces.length - a.faces.length).map(([mediaId, mediaData]: [string, any]) => (
              <Collapse.Panel header={`${mediaData.filename} (${mediaData.faces.length} instances)`} key={mediaId}>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                  {mediaData.faces.map((face: any, idx: number) => (
                      <Card key={idx} hoverable bodyStyle={{ padding: 0 }} onClick={() => setSelectedFace({ ...face, media_id: mediaId, filename: mediaData.filename })}>
                        <div style={{ position: 'relative', width: 120, height: 120, backgroundColor: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                          <div style={{ position: 'relative', width: '100%' }}>
                            <img
                              src={`/${state.project_name}/api/thumbnail?media_id=${mediaId}&timestamp=${face.timestamp}`}
                              style={{ width: '100%', height: 'auto', display: 'block' }}
                              alt="face"
                              title={`Vector ID: ${face.vector_id}`}
                            />
                            {face.bbox && <div style={{ position: 'absolute', border: '2px solid yellow', left: `${face.bbox.x*100}%`, top: `${face.bbox.y*100}%`, width: `${face.bbox.w*100}%`, height: `${face.bbox.h*100}%`, pointerEvents: 'none' }} />}
                          </div>
                        </div>
                      </Card>
                  ))}
                </div>
              </Collapse.Panel>
            ))}
          </Collapse>
        )}
      </Card>

      <Modal open={!!selectedFace} onCancel={() => setSelectedFace(null)} footer={null} width={800} destroyOnClose title={selectedFace?.filename || "Video Frame Context"}>
        {selectedFace && <VideoPlayerWithPoster project_name={state.project_name} face={selectedFace} />}
      </Modal>
      <Modal title="Add New Metadata Field" open={isSchemaModalOpen} onOk={handleAddSchemaField} onCancel={() => setIsSchemaModalOpen(false)}>
        <Form form={schemaForm} layout="vertical">
          <Form.Item name="key_name" label="Field Name" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="data_type" label="Data Type" initialValue="text">
            <Select>
              <Select.Option value="text">Text</Select.Option>
              <Select.Option value="number">Number</Select.Option>
              <Select.Option value="url">URL</Select.Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default ClusterView;
