import React, { useEffect, useRef, useState } from 'react';
import { Header } from 'antd/es/layout/layout';
import { Alert, Button, Collapse, Divider, Dropdown, Flex, Form, FormInstance, Input, Popover, Space, Tag, Tooltip, Upload, UploadFile, theme } from 'antd';
import { AudioFilled, /*AudioOutlined,*/ CaretRightOutlined, CloseOutlined, FontColorsOutlined, PictureOutlined, /*PlaySquareOutlined,*/ PlusOutlined, SearchOutlined, UploadOutlined } from '@ant-design/icons';
import { nanoid } from 'nanoid'

import './WiseHeader.scss';
import { WiseLogo } from './misc/logo.tsx';
import { TextSearchFormProps, MediaSearchFormProps, SearchExamplesProps, SearchDropdownProps, WiseHeaderProps, Query } from './misc/types.ts';
import config from './config.ts';

// TODO
// Update Tour feature, remove refsForTour.visualSearchButton and refsForTour.multimodalSearchButton

const examples = [
  {
    url: 'https://images.unsplash.com/photo-1559562328-bc48b8b32e2b?fm=jpg&w=640&fit=crop&q=80',
    text: 'in snow'
  },
  {
    url: 'https://images.unsplash.com/photo-1588064011404-57a7bc7133f5?auto=format&fit=crop&w=640&q=80',
    text: 'at night'
  },
  {
    url: 'https://images.unsplash.com/photo-1642653856727-957f76dd014e?auto=format&fit=crop&w=640&q=80'
  }
];

const TextSearchForm: React.FunctionComponent<TextSearchFormProps> = ({
  multimodalQueries, setMultimodalQueries,
  searchText, setSearchText,
  handleTextInputChange,
  submitSearch,
}) => {
  const formRef = useRef<FormInstance>(null);
  const { token } = useToken(); // Get theme styles

  useEffect(() => {
    formRef.current?.setFieldsValue({'text-query': searchText});
  }, [searchText]);

  const onFormSubmit = (e: any) => {
    console.log('Submit event', e);
    submitSearch();
  }

  const handleTextInputKeydown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addTextQuery();
    }
  }

  const addTextQuery = () => {
    let searchTextTrimmed = searchText.trim();
    if (!searchTextTrimmed) return;
    setMultimodalQueries([...multimodalQueries, { id: nanoid(), type: 'TEXT', displayText: searchTextTrimmed, value: searchTextTrimmed }]);
    setSearchText('');
    // setVisualSearchErrorMessage('');
    submitSearch();
  }

  return (
    <Form
      name="wise-text-search-form"
      className="wise-modality-form"
      ref={formRef}
      onFinish={onFormSubmit}
    >
      <p style={{marginTop: 0, color: token.colorTextDescription}}>Enter some text in the search bar above or the text box below. You can flexibly describe what you want to look for using natural language.</p>
      <Space.Compact>
        <Form.Item name="text-query">
          <Input placeholder="Search using natural language" onChange={handleTextInputChange} onKeyDown={handleTextInputKeydown} />
        </Form.Item>
        <Form.Item>
          <Button type="primary" onClick={addTextQuery}>Add text query</Button>
        </Form.Item>
      </Space.Compact>
    </Form>
  )
}


const MediaSearchForm: React.FunctionComponent<MediaSearchFormProps> = ({
  multimodalQueries, setMultimodalQueries,
  submitSearch,
  modality
}) => {
  const formRef = useRef<FormInstance>(null);
  const [urlText, setUrlText] = useState('');
  const [visualSearchErrorMessage, setVisualSearchErrorMessage] = useState('');

  const getFileList = (queryList: Query[]) => queryList.filter(query => query.type === 'FILE').map(query => query.value as UploadFile);
  const beforeUpload = (file: any) => {
    let fileList = getFileList(multimodalQueries);
    console.log('beforeUpload', file, fileList);
    if (fileList.length > 4) {
      setVisualSearchErrorMessage('Error: you can only upload a maximum of 5 images');
      throw new Error('Too many files selected');
    }
    setMultimodalQueries([...multimodalQueries, { id: nanoid(), type: 'FILE', displayText: file.name, value: file }]);
    setVisualSearchErrorMessage('');
    return true;
  }
  const handleFileSubmit = async (options: any) => {
    console.log('File action', options);
    // const { file, onSuccess, onError, onProgress } = options;
    formRef.current?.submit();
  }
  const onFormSubmit = (e: any) => {
    console.log('Submit event', e);
    addImageURLQuery();
    submitSearch();
  }

  const handleImageUrlInputKeydown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addImageURLQuery();
    }
  }
  const addImageURLQuery = (_urlText?: string) => {
    let urlTextTrimmed = (_urlText || urlText).trim();
    if (!urlTextTrimmed) return;
    try {
      urlTextTrimmed = new URL(urlTextTrimmed).href;
    } catch (e: any) {
      setVisualSearchErrorMessage('Invalid URL');
      console.error('Invalid URL', e);
      return;
    }

    setMultimodalQueries([...multimodalQueries, { id: nanoid(), type: 'URL', displayText: urlTextTrimmed, value: urlTextTrimmed }]);
    setUrlText('');
    setVisualSearchErrorMessage('');
    formRef.current?.setFieldsValue({'image-url': ''});
    formRef.current?.submit();
  }

  return (
    (modality) === 'video' ?
    <div>Searching with video has not been implemented yet. This functionality will be added later on.</div>
    :
    <Form
      name="wise-visual-search-form"
      className="wise-modality-form"
      ref={formRef}
      onFinish={onFormSubmit}
    >
      <Form.Item name="dragger" noStyle>
        {/* TODO accept different files based on modality */}
        <Upload.Dragger name="files" accept="image/*" beforeUpload={beforeUpload}
                        fileList={getFileList(multimodalQueries)} showUploadList={false} customRequest={handleFileSubmit}>
          <p className="ant-upload-drag-icon">
            <UploadOutlined />
          </p>
          <p className="ant-upload-text">Drag {modality === 'video' ? 'a' : 'an'} {modality} file or click here to upload</p>
        </Upload.Dragger>
      </Form.Item>
      <Divider>OR</Divider>
      <Space.Compact>
        <Form.Item name="image-url">
          <Input onChange={(e) => {setUrlText(e.target.value)}} onKeyDown={handleImageUrlInputKeydown} placeholder={`Paste ${modality} link`} />
        </Form.Item>
        <Form.Item>
          <Button type="primary" onClick={() => addImageURLQuery()}>Add {modality}</Button>
        </Form.Item>
      </Space.Compact>
      {
        modality === 'audio' &&
        <>
          <Divider>OR</Divider>
          <Space>
            <Button type="primary" shape="circle" icon={<AudioFilled />} size="large" />
            <span>Record audio</span>
          </Space>
        </>
      }
      {
        visualSearchErrorMessage && <Alert message={visualSearchErrorMessage} type="error" showIcon />
      }
    </Form>
  );
};

const SearchExamples: React.FunctionComponent<SearchExamplesProps> = ({
  setMultimodalQueries, setSearchText, submitSearch
}) => {
  const formRef = useRef<FormInstance>(null);
  const onFormSubmit = () => {
    submitSearch();
  }

  const handleExampleMultimodalQueryClick = (example: any) => {
    // Copied/modified from addImageURLQuery() and addTextQuery()
    let urlTextTrimmed = example.url.trim();
    let searchTextTrimmed = example.text.trim();
    setMultimodalQueries([
      { id: nanoid(), type: 'URL', displayText: urlTextTrimmed, value: urlTextTrimmed },
      { id: nanoid(), type: 'TEXT', displayText: searchTextTrimmed, value: searchTextTrimmed }
    ]);
    setSearchText('');
    formRef.current?.submit();
  }

  return <Form
    name="wise-search-examples-form"
    ref={formRef}
    onFinish={onFormSubmit}
  >
    {
      examples.slice(0,2).map(example => 
        <div className="wise-multimodal-example-query"
            onClick={() => handleExampleMultimodalQueryClick(example)}
            key={example.url}
        >
          <img src={example.url} />
          <span className="wise-multimodal-example-query-plus-sign">+</span>
          <Tag color='geekblue'>{example.text}</Tag>
        </div>
      )
    }
  </Form>
}


const modalities = [
  {
    id: 'text',
    label: 'Text',
    icon: <FontColorsOutlined />
  },
  {
    id: 'image',
    label: 'Image',
    icon: <PictureOutlined />
  },
  // {
  //   id: 'video',
  //   label: 'Video',
  //   icon: <PlaySquareOutlined />
  // },
  // {
  //   id: 'audio',
  //   label: 'Audio',
  //   icon: <AudioOutlined />
  // },
  // {
  //   id: 'metadata',
  //   label: 'Metadata',
  //   icon: <ControlOutlined /> // or <AlignLeftOutlined /> or <FileTextOutlined />
  // }
]
const { useToken } = theme;

const SearchDropdown: React.FunctionComponent<SearchDropdownProps> = ({
  multimodalQueries, setMultimodalQueries,
  searchText, setSearchText,
  handleTextInputChange,
  submitSearch, clearSearchBar,
  isHomePage
}) => {
  const { token } = useToken();
  const dropdownStyle: React.CSSProperties = {
    backgroundColor: token.colorBgElevated,
    borderRadius: token.borderRadiusLG,
    boxShadow: token.boxShadowSecondary
  };

  const [selectedModality, setSelectedModality] = useState<string>('image');
  const [isModalitySelected, setIsModalitySelected] = useState<boolean>(false);
  const selectModality = (modality: string) => {
    if (isModalitySelected && selectedModality === modality) {
      setIsModalitySelected(false);
    } else {
      setIsModalitySelected(true);
      setSelectedModality(modality);
    }
  };
  const _submitSearch = () => {
    setIsModalitySelected(false);
    submitSearch();
  };

  const collapseItems = [
    {
      key: 'examples',
      label: 'Examples',
      children: <SearchExamples setMultimodalQueries={setMultimodalQueries} setSearchText={setSearchText} submitSearch={_submitSearch} />,
    }
  ];
  const [activeKeys, setActiveKeys] = useState<string[]>(['examples']);
  const _setActiveKeys = (keys: string | string[]) => {
    if (Array.isArray(keys)) {
      setActiveKeys(keys);
    }
  };
  useEffect(() => {
    if (multimodalQueries.length > 0 || searchText) {
      // Close the examples panel
      setActiveKeys(activeKeys.filter(k => k !== 'examples'));
    }
  }, [multimodalQueries, searchText]);

  return (
    <div style={dropdownStyle}>
      <p style={{marginTop: 0, color: token.colorTextDescription}}>
        {
          // TODO improve this
          !isHomePage && (multimodalQueries.length > 0 || searchText) ?
          <><PlusOutlined /> Add another modality to your search:</>
          :
          "Search with text or an image, or a combination of both:"
          // "Search using any of the modalities below, or a combination of modalities:"
        }
      </p>
      <Space style={{marginBottom: 15}}>
        {
          modalities.map(modality => (
            <Button type="text" size="large" id={`wise-header-${modality.id}-modality-button`}
              className={(isModalitySelected && selectedModality === modality.id) ? 'selected' : isModalitySelected ? 'inactive' : undefined}
              onClick={() => selectModality(modality.id)}>
              {modality.icon} {modality.label}
            </Button>
          ))
        }
      </Space>
      <div id="wise-header-modality-collapsible" style={{maxHeight: isModalitySelected ? undefined : '0'}}>
        {
          selectedModality === 'text' ? 
            <TextSearchForm multimodalQueries={multimodalQueries} setMultimodalQueries={setMultimodalQueries}
                            searchText={searchText} setSearchText={setSearchText}
                            handleTextInputChange={handleTextInputChange} submitSearch={_submitSearch} />
          : <MediaSearchForm multimodalQueries={multimodalQueries} setMultimodalQueries={setMultimodalQueries}
                              submitSearch={_submitSearch} modality={selectedModality} />
        }
      </div>
      <div style={{borderTop: '1px solid #e3e3e3', marginTop: 20}} />
      <Collapse items={collapseItems}
        bordered={false} expandIcon={({ isActive }) => <CaretRightOutlined rotate={isActive ? 90 : 0} />}
        activeKey={activeKeys} onChange={_setActiveKeys}
        style={{background: 'unset'}}
      />
      <Flex style={{marginTop: 15}}>
        {(multimodalQueries.length > 0 || searchText) ? 
          <Button onClick={clearSearchBar}>Clear search</Button>
          : <></>
        }
      </Flex>
    </div>
  )
}

const QUERY_COLORS = {
  'TEXT': 'geekblue',
  'FILE': 'green',
  'URL': 'green',
  'INTERNAL_IMAGE': 'green',
}

const WiseHeader: React.FunctionComponent<WiseHeaderProps> = ({
  multimodalQueries, setMultimodalQueries, searchText, setSearchText, submitSearch, refsForTour, isHomePage = false, isSearching = false}: WiseHeaderProps) => {
  const handleTextInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchText(e.target.value);
  }

  const clearSearchBar = () => {
    setMultimodalQueries([]);
    setSearchText('');
  }

  const handleTagClose = (e: any, index: number) => {
    e.preventDefault();
    const newMultimodalQueries = multimodalQueries.slice();
    newMultimodalQueries.splice(index, 1);
    setMultimodalQueries(newMultimodalQueries);
  }

  const multimodalQueryTags = multimodalQueries.map((query, index) => {
    let icon = <></>;
    if (query.type === 'FILE') icon = <img src={URL.createObjectURL((query.value as unknown) as File)} />;
    else if (query.type === 'URL') icon = <img src={query.value} />;
    else if (query.type === 'INTERNAL_IMAGE') icon = <img src={config.API_BASE_URL + 'thumbs/' + query.value} />;

    const tag = <Tag closable
                  key={query.id}
                  className={(query.type === 'FILE' || query.type === 'URL' || query.type === 'INTERNAL_IMAGE') ? 'wise-search-tag-image' : undefined}
                  color={QUERY_COLORS[query.type]}
                  icon={icon}
                  onClose={(e) => handleTagClose(e, index)}
                >
                  {(query.isNegative ? '(Negative) ' : '') + query.displayText}
                </Tag>
    
    if (query.type === 'FILE') {
      return <Popover content={icon} key={query.id} title="Uploaded image" overlayClassName="wise-search-image-preview">{tag}</Popover>
    } else if (query.type === 'URL') {
      return <Popover content={icon} key={query.id} title="Online image" overlayClassName="wise-search-image-preview">{tag}</Popover>
    } else if (query.type === 'INTERNAL_IMAGE') {
      const full_image = <img src={config.API_BASE_URL + 'images/' + query.value} />;
      return <Popover content={full_image} key={query.id} title="Internal image" overlayClassName="wise-search-image-preview">{tag}</Popover>
    } else if (query.type === 'TEXT') {
      return <Tooltip title="Text query">{tag}</Tooltip>
    }
  });

  return (
    <Header className="wise-header">
      <div className="wise-header-primary-row" style={isHomePage ? { height: '90px' } : {}}>
        <a href="./" id="wise-logo">
          <WiseLogo />
        </a>
        <Dropdown
          overlayClassName="wise-search-dropdown"
          dropdownRender={_ => 
            <SearchDropdown multimodalQueries={multimodalQueries} setMultimodalQueries={setMultimodalQueries}
                            searchText={searchText} setSearchText={setSearchText}
                            handleTextInputChange={handleTextInputChange}
                            submitSearch={submitSearch} clearSearchBar={clearSearchBar}
                            isHomePage={isHomePage} />
          }
          // open={true}
          // TODO keep dropdown open when search input is focused
        >
          <Form onFinish={submitSearch} id="search-input-form">
            <Input
              id="search-input"
              autoComplete="off"
              size={isHomePage ? 'large' : 'middle'}
              placeholder={multimodalQueries.length === 0 ? 'Search' : ''}
              value={searchText}
              onChange={handleTextInputChange}
              prefix={multimodalQueryTags}
              suffix={
                <>
                  {
                    (multimodalQueries.length > 0 || searchText) && 
                    <Tooltip title="Clear">
                      <Button type="text" shape="circle" size="large" icon={<CloseOutlined />} onClick={clearSearchBar} />
                    </Tooltip>
                  }
                  <Button type="text" shape="circle" size="large" loading={isSearching} htmlType="submit" icon={<SearchOutlined />} />
                </>
              }
              ref={refsForTour.searchBar}
            />
          </Form>
        </Dropdown>
        <span className="wise-spacer"></span>
      </div>
    </Header>
  )
};

export default WiseHeader;

