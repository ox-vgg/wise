import React from 'react';
import { Layout, Typography, Button, message, Space } from 'antd';
import IndexView from './IndexView';
import FacetView from './FacetView';
import ClusterView from './ClusterView';
import MetadataSettingsView from './MetadataSettingsView';

const { Header, Content } = Layout;
const { Title } = Typography;

declare global {
  interface Window {
    __INITIAL_STATE__: any;
  }
}

const App: React.FC = () => {
  const state = window.__INITIAL_STATE__ || { view: 'index', facets: [] };
  
  const getSlug = (feature_extractor_id: string) => {
    const parts = feature_extractor_id.split('/');
    return parts.length > 1 ? parts[1] : feature_extractor_id;
  };

  const getBackUrl = () => {
    if (state.view !== 'cluster') return '#';
    const params = new URLSearchParams(window.location.search);
    const fromPage = params.get('from_page');
    let url = `/${state.project_name}/explore/${state.facet.name.toLowerCase()}/${getSlug(state.facet.feature_extractor_id)}/`;
    if (fromPage) {
      url += `?page=${fromPage}`;
    }
    return url;
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: '#fff' }}>
        <Title level={3} style={{ margin: 0 }}>
          {state.view === 'index' ? 'WISE Explore: Available Facets' : `WISE Explore: ${state.facet.name} Facet`}
        </Title>
        {state.view === 'facet' && (
          <Space>
            <Button type="link" onClick={() => window.location.href = `/${state.project_name}/explore/${state.facet.name.toLowerCase()}/${getSlug(state.facet.feature_extractor_id)}/metadata`}>
              Metadata Builder
            </Button>
            <Button type="primary" onClick={() => {
              fetch(`/${state.project_name}/api/facet/${state.facet.id}/publish`, { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                  if (data.published_clusters !== undefined) {
                    message.success(`Successfully published ${data.published_clusters} clusters.`);
                  } else if (data.status === "no reviewed clusters to publish") {
                    message.info('No reviewed clusters to publish.');
                  } else {
                    message.success('Publish operation completed.');
                  }
                })
                .catch(() => message.error('Failed to publish clusters.'));
            }}>
              Publish
            </Button>
          </Space>
        )}
        {state.view === 'cluster' && (
          <Button onClick={() => window.location.href = getBackUrl()}>
            &larr; Back to Clusters
          </Button>
        )}
      </Header>
      <Content style={{ padding: '24px' }}>
        {state.view === 'index' && <IndexView state={state} />}
        {state.view === 'facet' && <FacetView state={state} />}
        {state.view === 'cluster' && <ClusterView state={state} />}
        {state.view === 'metadata' && <MetadataSettingsView state={state} />}
      </Content>
    </Layout>
  );
}

export default App;