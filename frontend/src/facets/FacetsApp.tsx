import React, { useEffect, useState } from 'react';
import { Layout, Typography, Button, Spin, Alert } from 'antd';
import FacetsIndexView from './FacetsIndexView';
import FacetsClusterOverviewView from './FacetsClusterOverviewView';
import FacetsClusterDetailView from './FacetsClusterDetailView';

const { Header, Content } = Layout;
const { Title } = Typography;

declare global {
  interface Window {
    __INITIAL_STATE__: any;
  }
}

const FacetsApp: React.FC = () => {
  const [state, setState] = useState<any>(window.__INITIAL_STATE__);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!state) {
      // Dev mode: determine route from URL params
      const params = new URLSearchParams(window.location.search);
      const facetParam = params.get('facet');
      const featureExtractorParam = params.get('feature_extractor');
      const clusterParam = params.get('cluster_id');

      if (facetParam && featureExtractorParam && clusterParam) {
        fetch(`/api/facets/${facetParam}/${featureExtractorParam}/cluster/${clusterParam}/info`)
          .then(res => {
            if (!res.ok) throw new Error('Facets feature is disabled or unavailable.');
            return res.json();
          })
          .then(data => {
            setState({
              view: 'cluster',
              project_name: '',
              facet: data.facet,
              cluster: { id: data.id, cluster_label: data.cluster_label, metadata: data.metadata, size: data.size }
            });
          })
          .catch(err => setError(err.message));
      } else if (facetParam && featureExtractorParam) {
        fetch(`/api/facets/${facetParam}/${featureExtractorParam}/info`)
          .then(res => {
            if (!res.ok) throw new Error('Facets feature is disabled or unavailable.');
            return res.json();
          })
          .then(data => {
            setState({
              view: 'facet',
              project_name: '', // Use empty string since API is proxied directly at root
              facet: { id: data.id, name: data.name, feature_extractor_id: data.feature_extractor_id },
              total_clusters: data.total_clusters
            });
          })
          .catch(err => setError(err.message));
      } else {
        fetch(`/api/facets`)
          .then(res => {
            if (!res.ok) throw new Error('Facets feature is disabled or unavailable.');
            return res.json();
          })
          .then(data => {
            setState({
              view: 'index',
              project_name: '',
              facets: data
            });
          })
          .catch(err => setError(err.message));
      }
    }
  }, []);

  if (error) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20vh' }}>
        <Alert message="Error" description={error} type="error" showIcon />
      </div>
    );
  }

  if (!state) {
    return <Spin size="large" style={{ display: 'flex', justifyContent: 'center', marginTop: '20vh' }} />;
  }
  const getSlug = (feature_extractor_id: string) => {
    const parts = feature_extractor_id.split('/');
    return parts.length > 1 ? parts[1] : feature_extractor_id;
  };

  const handleBack = () => {
    if (state.view === 'cluster') {
      if (!window.__INITIAL_STATE__) {
        window.location.href = `/facets.html?facet=${state.facet.name.toLowerCase()}&feature_extractor=${getSlug(state.facet.feature_extractor_id)}`;
      } else {
        window.location.href = `${state.project_name ? '/' + state.project_name : ''}/facets/${state.facet.name.toLowerCase()}/${getSlug(state.facet.feature_extractor_id)}/`;
      }
    } else if (state.view === 'facet') {
      if (!window.__INITIAL_STATE__) {
        window.location.href = '/facets.html';
      } else {
        window.location.href = `${state.project_name ? '/' + state.project_name : ''}/facets/`;
      }
    } else {
      window.location.href = state.project_name ? `/${state.project_name}/` : '/';
    }
  };

  const renderTitle = () => {
    const indexUrl = window.__INITIAL_STATE__ ? `${state.project_name ? '/' + state.project_name : ''}/facets/` : '/facets.html';

    if (state.view === 'index') {
      return <span>WISE Facets</span>;
    }

    const facetUrl = window.__INITIAL_STATE__
      ? `${state.project_name ? '/' + state.project_name : ''}/facets/${state.facet.name.toLowerCase()}/${getSlug(state.facet.feature_extractor_id)}/`
      : `/facets.html?facet=${state.facet.name.toLowerCase()}&feature_extractor=${getSlug(state.facet.feature_extractor_id)}`;

    if (state.view === 'facet') {
      return <span><a href={indexUrl} style={{color: 'inherit', textDecoration: 'none'}}>WISE Facets</a> &gt; {state.facet.name}</span>;
    }

    if (state.view === 'cluster') {
      return <span><a href={indexUrl} style={{color: 'inherit', textDecoration: 'none'}}>WISE Facets</a> &gt; <a href={facetUrl} style={{color: 'inherit', textDecoration: 'none'}}>{state.facet.name}</a> &gt; {state.cluster.cluster_label}</span>;
    }
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: '#fff' }}>
        <Title level={3} style={{ margin: 0 }}>
          {renderTitle()}
        </Title>
        <Button onClick={handleBack}>
          &larr; Back
        </Button>
      </Header>
      <Content style={{ padding: '24px' }}>
        {state.view === 'index' && <FacetsIndexView state={state} />}
        {state.view === 'facet' && <FacetsClusterOverviewView state={state} />}
        {state.view === 'cluster' && <FacetsClusterDetailView state={state} />}
      </Content>
    </Layout>
  );
}

export default FacetsApp;