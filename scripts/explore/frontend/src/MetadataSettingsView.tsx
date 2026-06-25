import React, { useEffect, useState } from 'react';
import { Card, Button, Spin, Select, message, Table, Popconfirm, Modal, Form, Input } from 'antd';
import { DeleteOutlined, EditOutlined, SettingOutlined } from '@ant-design/icons';

const MetadataSettingsView: React.FC<{ state: any }> = ({ state }) => {
  const [schema, setSchema] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  // Batch edit state
  const [batchMetadata, setBatchMetadata] = useState<any>({});
  const [isSchemaModalOpen, setIsSchemaModalOpen] = useState(false);
  const [schemaForm] = Form.useForm();

  const loadSchema = () => {
    setLoading(true);
    fetch(`/${state.project_name}/api/facet/${state.facet.id}/schema`)
      .then(res => res.json())
      .then(data => {
        setSchema(data);
        setLoading(false);
      });
  };

  useEffect(() => {
    loadSchema();
  }, [state.facet.id, state.project_name]);

  const handleDeleteField = (schemaId: number) => {
    fetch(`/${state.project_name}/api/facet/${state.facet.id}/schema/${schemaId}`, {
      method: 'DELETE',
    }).then(res => {
      if(res.ok) {
        message.success('Field deleted');
        loadSchema();
      } else {
        message.error('Failed to delete field');
      }
    });
  };

  const handleUpdateField = (schemaId: number, keyName: string, dataType: string) => {
    fetch(`/${state.project_name}/api/facet/${state.facet.id}/schema/${schemaId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key_name: keyName, data_type: dataType })
    }).then(async res => {
      if(res.ok) {
        message.success('Field updated');
        loadSchema();
      } else {
        const data = await res.json();
        message.error(data.detail || 'Failed to update field');
      }
    });
  };

  const handleBatchSave = () => {
    fetch(`/${state.project_name}/api/facet/${state.facet.id}/metadata/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ metadata_json: batchMetadata })
    }).then(async res => {
      if(res.ok) {
        const data = await res.json();
        message.success(`Successfully updated ${data.updated_count} starred clusters`);
        setBatchMetadata({}); // clear after success
      } else {
        message.error('Failed to batch update metadata');
      }
    });
  };

  const handleClearStars = () => {
    fetch(`/${state.project_name}/api/facet/${state.facet.id}/stars`, {
      method: 'DELETE',
    }).then(res => {
      if(res.ok) {
        message.success('All stars cleared successfully');
      } else {
        message.error('Failed to clear stars');
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
          message.success('Field added');
          setIsSchemaModalOpen(false);
          schemaForm.resetFields();
          loadSchema();
        } else {
          const data = await res.json();
          message.error(data.detail || 'Failed to add field');
        }
      });
    });
  };

  const columns = [
    {
      title: 'Field Name',
      dataIndex: 'key_name',
      key: 'key_name',
      render: (text: string, record: any) => (
        <Input
          defaultValue={text}
          onBlur={(e) => {
            if (e.target.value !== text) {
              Modal.confirm({
                title: 'Are you sure you want to rename this field?',
                content: 'This will modify the metadata keys across all existing clusters. Data loss may occur if the key conflicts.',
                onOk: () => handleUpdateField(record.id, e.target.value, record.data_type)
              });
            }
          }}
        />
      )
    },
    {
      title: 'Data Type',
      dataIndex: 'data_type',
      key: 'data_type',
      render: (text: string, record: any) => (
        <Select
          defaultValue={text}
          onChange={(val) => {
            Modal.confirm({
              title: 'Are you sure you want to change the data type?',
              content: 'This may affect how values are rendered and validated.',
              onOk: () => handleUpdateField(record.id, record.key_name, val)
            });
          }}
        >
          <Select.Option value="string">Text (String)</Select.Option>
          <Select.Option value="number">Number</Select.Option>
          <Select.Option value="date">Date</Select.Option>
        </Select>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: any) => (
        <Popconfirm
          title="Delete this metadata field?"
          description="Are you absolutely sure? This will PERMANENTLY DELETE this metadata key and its values from ALL clusters in this facet!"
          onConfirm={() => handleDeleteField(record.id)}
          okText="Yes, Delete"
          cancelText="Cancel"
          okButtonProps={{ danger: true }}
        >
          <Button danger icon={<DeleteOutlined />} />
        </Popconfirm>
      )
    }
  ];

  const getSlug = (feature_extractor_id: string) => {
    const parts = feature_extractor_id.split('/');
    return parts.length > 1 ? parts[1] : feature_extractor_id;
  };

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <Button onClick={() => window.location.href = `/${state.project_name}/explore/${state.facet.name.toLowerCase()}/${getSlug(state.facet.feature_extractor_id)}/`}>
          &larr; Back to Facet Overview
        </Button>
      </div>

      <Card title="Manage Metadata Schema" style={{ marginBottom: 24 }}>
        <p>Edit or delete existing metadata fields below. <strong>Warning:</strong> Renaming or deleting fields will immediately affect all existing clusters that use them.</p>
        <div style={{ marginBottom: '16px' }}>
          <Button type="dashed" onClick={() => setIsSchemaModalOpen(true)}>
             + Add Metadata Field
          </Button>
        </div>
        <Table
          dataSource={schema}
          columns={columns}
          rowKey="id"
          loading={loading}
          pagination={false}
        />
      </Card>

      <Card title="Starred Clusters">
        <p>Enter values below to instantly apply them to <strong>ALL clusters you have marked with a Star</strong> in the overview grid. This action will also mark all those clusters as 'Reviewed'.</p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', maxWidth: '600px' }}>
          {schema.map(field => (
             <div key={field.key_name} style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
               <span style={{ fontWeight: 'bold', width: '150px', textTransform: 'capitalize' }}>
                 {field.key_name.replace(/_/g, ' ')}
               </span>
               <Input
                 type={field.data_type === 'number' ? 'number' : field.data_type === 'date' ? 'date' : 'text'}
                 value={batchMetadata[field.key_name] || ''}
                 onChange={e => setBatchMetadata({ ...batchMetadata, [field.key_name]: e.target.value })}
                 style={{ flex: 1 }}
                 placeholder={`Enter value to apply to all starred clusters...`}
               />
             </div>
          ))}
        </div>
        <div style={{ marginTop: '24px', display: 'flex', gap: '16px' }}>
          <Popconfirm
            title="Batch Update Metadata"
            description="Are you sure you want to overwrite these fields on ALL starred clusters?"
            onConfirm={handleBatchSave}
            okText="Yes, Update All"
            cancelText="Cancel"
          >
            <Button type="primary" disabled={Object.keys(batchMetadata).length === 0}>
              Apply to Starred Clusters
            </Button>
          </Popconfirm>

          <Popconfirm
            title="Clear All Stars"
            description="Are you sure you want to clear the star from ALL clusters in this facet?"
            onConfirm={handleClearStars}
            okText="Yes, Clear All"
            cancelText="Cancel"
            okButtonProps={{ danger: true }}
          >
            <Button danger>
              Clear All Stars
            </Button>
          </Popconfirm>
        </div>
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
    </div>
  );
};

export default MetadataSettingsView;
