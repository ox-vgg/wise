import React, { useRef, useState } from 'react';
import { Header } from 'antd/es/layout/layout';
import { Alert, Button, Form, FormInstance, Input, Popover, Space, Tag, Tooltip, Upload, UploadFile } from 'antd';
import { PictureOutlined, SearchOutlined, UploadOutlined } from '@ant-design/icons';
import { nanoid } from 'nanoid'

import './WiseHeader.scss';
import { WiseLogo } from './misc/logo.tsx';
import { CompoundSearchPopoverProps, WiseHeaderProps, Query } from './misc/types.ts';


const CompoundSearchPopover: React.FunctionComponent<CompoundSearchPopoverProps> = ({
  multimodalQueries, setMultimodalQueries,
  searchText, setSearchText,
  handleTextInputChange,
  submitSearch,
  onlyVisualSearch = false
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
    if (onlyVisualSearch) return true;
    else return false;
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
  const handleTextInputKeydown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addTextQuery();
    }
  }
  const addImageURLQuery = () => {
    let urlTextTrimmed = urlText.trim();
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

    if (onlyVisualSearch) formRef.current?.submit();
  }
  const addTextQuery = () => {
    // console.log(searchText)
    let searchTextTrimmed = searchText.trim();
    if (!searchTextTrimmed) return;
    setMultimodalQueries([...multimodalQueries, { id: nanoid(), type: 'TEXT', displayText: searchTextTrimmed, value: searchTextTrimmed }]);
    setSearchText('');
    setVisualSearchErrorMessage('');
    formRef.current?.setFieldsValue({'text-query': ''});
  }

  let divStyle = {};
  if (onlyVisualSearch) divStyle = {padding: '1px 1px'};
  else divStyle = {padding: '1px 15px', backgroundColor: '#f2f2f2', borderRadius: '10px'};

  return <div style={{width: '450px'}}>
    {
      onlyVisualSearch ? <></> : 
        <p>You can combine image and text queries here. You can add multiple images and/or multiple text queries.</p>
    }
    <Form
        name="wise-visual-search-form"
        ref={formRef}
        onFinish={onFormSubmit}
      >
      <div style={divStyle}>
        {
          onlyVisualSearch ? <></> : <h3>Image</h3>
        }
        <Form.Item name="dragger" noStyle>
          <Upload.Dragger name="files" accept="image/*" beforeUpload={beforeUpload}
                          fileList={getFileList(multimodalQueries)} showUploadList={false} customRequest={handleFileSubmit}>
            <p className="ant-upload-drag-icon">
              <UploadOutlined />
            </p>
            <p className="ant-upload-text">Drag an image or click here to upload</p>
            {/* <p className="ant-upload-hint">Support for a single or bulk upload.</p> */}
          </Upload.Dragger>
        </Form.Item>
        <div className="wise-visual-search-separator">
          <div></div>
          <span style={{margin: '0 10px'}}>OR</span>
          <div></div>
        </div>
        <Space.Compact style={{width: '100%'}}>
          <Form.Item name="image-url" style={{flex: 1}}>
            <Input onChange={(e) => {setUrlText(e.target.value)}} onKeyDown={handleImageUrlInputKeydown} placeholder="Paste image link" />
          </Form.Item>
          {
            onlyVisualSearch ? <></> : 
              <Form.Item>
                <Button type="primary" onClick={addImageURLQuery}>Add image</Button>
              </Form.Item>
          }
        </Space.Compact>
        {
          visualSearchErrorMessage ? <Alert message={visualSearchErrorMessage} type="error" showIcon /> : <></>
        }
      </div>
      {
        onlyVisualSearch ? <></> : 
          <>
            <div className="wise-visual-search-separator">
              <div></div>
              <span style={{margin: '0 10px'}}>AND</span>
              <div></div>
            </div>
            <div style={{...divStyle, marginBottom: '15px'}}>
              <h3>Text</h3>
              <Space.Compact style={{width: '100%'}}>
                <Form.Item name="text-query" style={{flex: 1}}>
                  <Input placeholder="Text query" onChange={handleTextInputChange} onKeyDown={handleTextInputKeydown} />
                </Form.Item>
                <Form.Item>
                  <Button type="primary" onClick={addTextQuery}>Add text query</Button>
                </Form.Item>
              </Space.Compact>
            </div>
            <Button type="primary" htmlType="submit">Search</Button>
          </>
      }
    </Form>
  </div>
};


const WiseHeader: React.FunctionComponent<WiseHeaderProps> = ({
  multimodalQueries, setMultimodalQueries, searchText, setSearchText, submitSearch, isHomePage = false, isSearching = false}: WiseHeaderProps) => {
  const handleTextInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchText(e.target.value);
  }
  
  const [isPopover1Open, setIsPopover1Open] = useState(false);
  const [isPopover2Open, setIsPopover2Open] = useState(false);

  const _submitSearch = () => {
    setIsPopover1Open(false);
    setIsPopover2Open(false);
    submitSearch();
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

    const tag = <Tag closable
                  key={query.id}
                  className={(query.type === 'FILE' || query.type === 'URL') ? 'wise-search-tag-image' : undefined}
                  color={(query.type === 'FILE' || query.type === 'URL') ? 'blue' : 'geekblue'}
                  icon={icon}
                  onClose={(e) => handleTagClose(e, index)}
                >
                  {query.displayText}
                </Tag>
    
    if (query.type === 'FILE') {
      return <Popover content={icon} key={query.id} title="Uploaded image" overlayClassName="wise-search-image-preview">{tag}</Popover>
    } else if (query.type === 'URL') {
      return <Popover content={icon} key={query.id} title="Online image" overlayClassName="wise-search-image-preview">{tag}</Popover>
    } else {
      return tag;
    }
  });

  return <Header className="wise-header" style={{"--wise-search-input-flex-grow": (isPopover1Open || isPopover2Open) ? '1' : 'unset'} as React.CSSProperties}>
    <div className="wise-header-primary-row" style={isHomePage ? { height: '90px' } : {}}>
      <a href="./" id="wise-logo">
        <WiseLogo />
      </a>
      <Form onFinish={submitSearch}>
        <Input
          id="search-input"
          size={isHomePage ? 'large' : 'middle'}
          placeholder={multimodalQueries.length === 0 ? 'Search' : ''}
          value={searchText}
          onChange={handleTextInputChange}
          prefix={multimodalQueryTags}
          suffix={
            <>
              <Popover content={
                  <CompoundSearchPopover multimodalQueries={multimodalQueries} setMultimodalQueries={setMultimodalQueries}
                                        searchText={searchText} setSearchText={setSearchText}
                                        submitSearch={_submitSearch} onlyVisualSearch={true} />
                } title="Visual search" open={isPopover1Open} onOpenChange={setIsPopover1Open} trigger="click">
                <Tooltip title="Search with an image">
                  <Button type="text" shape="circle" size="large" icon={<PictureOutlined />} />
                </Tooltip>
              </Popover>
              <Popover content={
                  <CompoundSearchPopover multimodalQueries={multimodalQueries} setMultimodalQueries={setMultimodalQueries}
                                        searchText={searchText} setSearchText={setSearchText}
                                        submitSearch={_submitSearch} handleTextInputChange={handleTextInputChange} />
                } title="Compound multi-modal search" open={isPopover2Open} onOpenChange={setIsPopover2Open} trigger="click">
                <Tooltip title="Compound multi-modal search">
                  <Button type="text" shape="circle" size="large" icon={<span className="anticon wise-multimodal-search-icon"><img src="compound_icon.svg" /></span>} />
                </Tooltip>
              </Popover>
              <Button type="text" shape="circle" size="large" loading={isSearching} htmlType="submit" icon={<SearchOutlined />} />
            </>
          }
        />
      </Form>
      <span className="wise-spacer"></span>
    </div>
  </Header>
};

export default WiseHeader;

