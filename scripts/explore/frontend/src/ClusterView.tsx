import React, { useEffect, useState } from 'react';
import { Card, Button, Pagination, Spin, Select, message, Modal, Segmented, Typography, Input, Form } from 'antd';

const VideoPlayerWithHighResPoster: React.FC<{ project_name: string, face: any }> = ({ project_name, face }) => {
  const lowResUrl = `/${project_name}/api/thumbnail?media_id=${face.media_id}&timestamp=${face.timestamp}`;
  const highResUrl = `/${project_name}/api/thumbnail?media_id=${face.media_id}&timestamp=${face.timestamp}&high_res=true`;
  const [poster, setPoster] = useState(lowResUrl);
  const [hasPlayed, setHasPlayed] = useState(false);

  useEffect(() => {
    setPoster(lowResUrl);
    const img = new Image();
    img.src = highResUrl;
    img.onload = () => setPoster(highResUrl);
  }, [lowResUrl, highResUrl]);

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
  const [faces, setFaces] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [status, setStatus] = useState(state.cluster.status);
  const [selectedFace, setSelectedFace] = useState<any>(null);

  // Metadata Panel State
  const [clusterLabel, setClusterLabel] = useState(state.cluster.cluster_label);
  const [schema, setSchema] = useState<any[]>([]);
  const [metadataJson, setMetadataJson] = useState<any>(state.cluster.metadata || {});
  const [isSchemaModalOpen, setIsSchemaModalOpen] = useState(false);
  const [schemaForm] = Form.useForm();

  useEffect(() => {
    setLoading(true);
    fetch(`/${state.project_name}/api/cluster/${state.cluster.id}/faces?page=${page}&page_size=${pageSize}`)
      .then(res => res.json())
      .then(data => {
        setFaces(data);
        setLoading(false);
      });
  }, [state.cluster.id, page, pageSize, state.project_name]);

  useEffect(() => {
    // Load Facet Metadata Schema
    fetch(`/${state.project_name}/api/facet/${state.facet.id}/schema`)
      .then(res => res.json())
      .then(data => setSchema(data))
      .catch(console.error);
  }, [state.facet.id, state.project_name]);

  const handleStatusChange = (newStatus: string) => {
    fetch(`/${state.project_name}/api/cluster/${state.cluster.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: newStatus })
    }).then(res => {
      if(res.ok) {
        setStatus(newStatus);
        message.success('Status updated');
      }
    });
  };

  const handleLabelChange = () => {
    if (!clusterLabel.trim()) return;
    fetch(`/${state.project_name}/api/cluster/${state.cluster.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cluster_label: clusterLabel.trim(), status: 'reviewed' })
    }).then(res => {
      if(res.ok) {
        setStatus('reviewed');
        message.success('Label updated and marked as reviewed');
      } else {
        message.error('Failed to update label');
      }
    }).catch(() => message.error('Failed to update label'));
  };

  const handleMetadataChange = (key: string, value: any) => {
    setMetadataJson({ ...metadataJson, [key]: value });
  };

  const handleMetadataSave = () => {
    fetch(`/${state.project_name}/api/cluster/${state.cluster.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ metadata_json: metadataJson, status: 'reviewed' })
    }).then(res => {
      if(res.ok) {
        setStatus('reviewed');
        message.success('Metadata updated and marked as reviewed');
      } else {
        message.error('Failed to save metadata');
      }
    });
  };

  const handleAddSchemaField = () => {
    schemaForm.validateFields().then(values => {
      fetch(`/${state.project_name}/api/facet/${state.facet.id}/schema`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key_name: values.key_name, data_type: values.data_type })
      }).then(async res => {
        if(res.ok) {
          const newField = await res.json();
          setSchema([...schema, newField]);
          setIsSchemaModalOpen(false);
          schemaForm.resetFields();
          message.success('Field added');
        } else {
          const data = await res.json();
          message.error(data.detail || 'Failed to add field');
        }
      });
    });
  };

  return (
    <div>
      {/* Metadata Panel */}
      <Card style={{ marginBottom: 16 }} bodyStyle={{ padding: '16px 24px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <span style={{ fontWeight: 'bold', width: '150px' }}>Cluster ID</span>
            <span>{state.cluster.id}</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <span style={{ fontWeight: 'bold', width: '150px' }}>Cluster Label</span>
            <Input
              value={clusterLabel}
              onChange={e => setClusterLabel(e.target.value)}
              onBlur={handleLabelChange}
              onPressEnter={handleLabelChange}
              style={{ maxWidth: '400px' }}
            />
          </div>

          {schema.map(field => (
             <div key={field.key_name} style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
               <span style={{ fontWeight: 'bold', width: '150px', textTransform: 'capitalize' }}>
                 {field.key_name.replace(/_/g, ' ')}
               </span>
               <Input
                 type={field.data_type === 'number' ? 'number' : field.data_type === 'date' ? 'date' : 'text'}
                 value={metadataJson[field.key_name] || ''}
                 onChange={e => handleMetadataChange(field.key_name, e.target.value)}
                 onBlur={handleMetadataSave}
                 onPressEnter={handleMetadataSave}
                 style={{ maxWidth: '400px' }}
                 placeholder={`Enter ${field.data_type}...`}
               />
             </div>
          ))}

          <div style={{ marginTop: '8px' }}>
            <Button type="dashed" onClick={() => setIsSchemaModalOpen(true)}>
               + Add Metadata Field
            </Button>
          </div>
        </div>
      </Card>

      <Card
        title={
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <Pagination
              current={page}
              onChange={(p, s) => { setPage(p); setPageSize(s); }}
              onShowSizeChange={(current, size) => { setPageSize(size); setPage(1); }}
              total={state.cluster.size}
              pageSize={pageSize}
              size="small"
              style={{ margin: 0 }}
            />
          </div>
        }
        extra={
          <>
            <style>{`
              .custom-segmented .ant-segmented-item-selected {
                background-color: #bae0ff !important;
                font-weight: 500;
              }
            `}</style>
            <Segmented
              className="custom-segmented"
              value={status}
              onChange={(val) => handleStatusChange(val as string)}
              options={[
                { label: 'Draft', value: 'draft' },
                { label: 'Reviewed', value: 'reviewed' },
                { label: 'Published', value: 'published' }
              ]}
            />
          </>
        }
      >
        {loading ? <Spin /> : (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {faces.map((face, idx) => (
              <Card key={idx} hoverable bodyStyle={{ padding: 0 }} onClick={() => setSelectedFace(face)}>
                <div style={{ position: 'relative', width: 120, height: 120, backgroundColor: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                  <div style={{ position: 'relative', width: '100%' }}>
                    <img
                      src={`/${state.project_name}/api/thumbnail?media_id=${face.media_id}&timestamp=${face.timestamp}`}
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
        title="Add New Metadata Field"
        open={isSchemaModalOpen}
        onOk={handleAddSchemaField}
        onCancel={() => { setIsSchemaModalOpen(false); schemaForm.resetFields(); }}
      >
        <Form form={schemaForm} layout="vertical">
          <Form.Item name="key_name" label="Field Name (e.g. reference_url)" rules={[{ required: true, message: 'Please enter a field name' }]}>
            <Input placeholder="Enter field name (no spaces)" />
          </Form.Item>
          <Form.Item name="data_type" label="Data Type" initialValue="string" rules={[{ required: true }]}>
            <Select>
              <Select.Option value="string">Text (String)</Select.Option>
              <Select.Option value="number">Number</Select.Option>
              <Select.Option value="date">Date</Select.Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>

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

export default ClusterView;