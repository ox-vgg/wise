import React, {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react';
import { Header } from 'antd/es/layout/layout';
import {
  Alert,
  AutoComplete,
  Button,
  Checkbox,
  Collapse,
  Divider,
  Dropdown,
  Flex,
  Form,
  FormInstance,
  Input,
  Popover,
  Select,
  Space,
  Tag,
  Tooltip,
  Upload,
  UploadFile,
  theme,
} from 'antd';
import {
  CaretRightOutlined,
  CloseOutlined,
  FileTextTwoTone,
  FontColorsOutlined,
  PictureOutlined,
  PictureTwoTone,
  PlusOutlined,
  QuestionCircleOutlined,
  SearchOutlined,
  SoundOutlined,
  SoundTwoTone,
  UploadOutlined,
  VideoCameraTwoTone,
} from '@ant-design/icons';
import { nanoid } from 'nanoid'

import './WiseHeader.scss';
import { WiseLogo } from './misc/logo.tsx';
import config from './config.ts';
import StillImageView from './misc/StillImageView.tsx';
import {
  MediaSearchFormProps,
  Query,
  SearchDropdownProps,
  SearchExamplesProps,
  TextSearchFormProps,
  ViewModality,
  WiseHeaderProps,
} from './misc/types.ts';
import { BoundingBoxes } from './misc/BoundingBox.tsx';
import { is_metadata_filter_supported } from './misc/utils.ts';

const QUERY_COLORS = {
  'TEXT': 'geekblue',
  'IMAGE_FILE': 'green',
  'IMAGE_URL': 'green',
  'INTERNAL_IMAGE': 'green',
  'AUDIO_FILE': 'orange',
  'AUDIO_URL': 'orange',
  'METADATA': 'purple',
}

const VIEW_MODALITY_OPTIONS = {
  image: {
    icon: <PictureTwoTone />,
    label: 'Image',
    longLabel: 'Image',
    value: 'Image',
  },
  video: {
    icon: <VideoCameraTwoTone />,
    label: 'Video',
    longLabel: 'Video (visual track)',
    value: 'Video',
  },
  audio: {
    icon: <SoundTwoTone />,
    label: 'Audio',
    longLabel: 'Audio track of video',
    value: 'VideoAudio',
  },
} as const;


const VIEW_MODALITY_OPTIONS_EXTRA = {
  "video:wise/metadata": {
    icon: <FileTextTwoTone />,
    label: 'Metadata',
    longLabel: 'Media Metadata',
    value: 'video:wise/metadata',
  }
}

type ViewModalityKey = keyof typeof VIEW_MODALITY_OPTIONS;


const TextSearchForm: React.FunctionComponent<React.PropsWithChildren<TextSearchFormProps>> = ({
  multimodalQueries, setMultimodalQueries,
  searchText = '',
  submitSearch,
  handleTextInputChange,
  placeholder = 'Search using natural language',
  queryType = 'TEXT',
  buttonText = 'Add Text Query',
  children,
}) => {
  const formRef = useRef<FormInstance>(null);

  const [searchTextLocal, setSearchTextLocal] = useState(searchText);
  useEffect(() => {
    formRef.current?.setFieldsValue({ 'text-query': searchTextLocal });
  }, [searchTextLocal]);

  useEffect(() => {
    setSearchTextLocal(searchText);
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
  const _handleTextInputChange = ((x: string) => { handleTextInputChange?.(x); setSearchTextLocal(x) });

  const addTextQuery = () => {
    let searchTextTrimmed = searchTextLocal.trim();
    if (!searchTextTrimmed) return;
    const _queries = [...multimodalQueries, { id: nanoid(), type: queryType, displayText: searchTextTrimmed, value: searchTextTrimmed }]
    setMultimodalQueries(_queries);
    handleTextInputChange?.('');
    setSearchTextLocal('');
    // setVisualSearchErrorMessage('');
    // formRef.current?.setFieldsValue({'text-query': ''});
    formRef.current?.submit();
  }

  return (
    <Form
      name="wise-text-search-form"
      className="wise-modality-form"
      ref={formRef}
      onFinish={onFormSubmit}
    >
      {children}
      <Space.Compact>
        <Form.Item name="text-query">
          <Input
            placeholder={placeholder}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => _handleTextInputChange(e.target.value)}
            onKeyDown={handleTextInputKeydown}
            autoComplete='off'
          />
        </Form.Item>
        <Form.Item>
          <Button type={queryType === "TEXT" ? "primary" : undefined} onClick={addTextQuery}>{buttonText}</Button>
        </Form.Item>
      </Space.Compact>
    </Form>
  )
}


const MediaSearchForm: React.FunctionComponent<MediaSearchFormProps> = ({
  multimodalQueries, setMultimodalQueries,
  submitSearch,
  modality,
  featureExtractorId,
}) => {
  const formRef = useRef<FormInstance>(null);
  const [urlText, setUrlText] = useState('');
  const [visualSearchErrorMessage, setVisualSearchErrorMessage] = useState('');

  const getFileList = (queryList: Query[]) => queryList.filter(query => query.type === 'IMAGE_FILE' || query.type === 'AUDIO_FILE').map(query => query.value as UploadFile);
  const beforeUpload = (file: any) => {
    let fileList = getFileList(multimodalQueries);
    console.log('beforeUpload', file, fileList);
    if (fileList.length > 4) {
      setVisualSearchErrorMessage('Error: you can only upload a maximum of 5 files');
      throw new Error('Too many files selected');
    }
    setMultimodalQueries([...multimodalQueries, { id: nanoid(), type: (modality == 'audio') ? 'AUDIO_FILE' : 'IMAGE_FILE', displayText: file.name, value: file }]);
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

    setMultimodalQueries([...multimodalQueries, { id: nanoid(), type: (modality == 'audio') ? 'AUDIO_URL' : 'IMAGE_URL', displayText: urlTextTrimmed, value: urlTextTrimmed }]);
    setUrlText('');
    setVisualSearchErrorMessage('');
    formRef.current?.setFieldsValue({ 'image-url': '' });
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
        {
          // Show warning message for object search
          featureExtractorId.includes("owlv2") &&
          <Alert
            message="Please choose an image with only one prominent object. Searching with an image containing multiple objects might not work as expected"
            type="info" showIcon style={{ marginBottom: 16 }}
          />
        }
        <Form.Item name="dragger" noStyle>
          <Upload.Dragger name="files" accept={modality + "/*"} beforeUpload={beforeUpload}
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
            <Input onChange={(e) => { setUrlText(e.target.value) }} onKeyDown={handleImageUrlInputKeydown} placeholder={`Paste ${modality} link`} />
          </Form.Item>
          <Form.Item>
            <Button type="primary" onClick={() => addImageURLQuery()}>Add {modality}</Button>
          </Form.Item>
        </Space.Compact>
        {
          // modality === 'audio' &&
          // <>
          //   <Divider>OR</Divider>
          //   <Space>
          //     <Button type="primary" shape="circle" icon={<AudioFilled />} size="large" />
          //     <span>Record audio</span>
          //   </Space>
          // </>
        }
        {
          visualSearchErrorMessage && <Alert message={visualSearchErrorMessage} type="error" showIcon />
        }
      </Form>
  );
};

const SearchExamples: React.FunctionComponent<SearchExamplesProps> = ({
  setMultimodalQueries, setSearchText, submitSearch, viewModality, featureExtractorId,
}) => {
  const formRef = useRef<FormInstance>(null);
  const onFormSubmit = () => {
    submitSearch();
  }

  const [_, _featureExtractorId] = featureExtractorId.split('/', 2);
  const multimodalQueries = config.MULTIMODAL_EXAMPLE_QUERIES[`${viewModality}:${_featureExtractorId}`] || [];
  const is_audio = ((viewModality as string) === 'Audio' || viewModality === 'VideoAudio');
  if (multimodalQueries.length === 0) {
    return <></>
  }
  const handleExampleMultimodalQueryClick = (example: { url?: string, text?: string, displayText?: string }) => {
    // Copied/modified from addImageURLQuery() and addTextQuery()
    const _searchqueries: Query[] = [];
    if (example.url) {
      const urlTextTrimmed = example.url?.trim();
      _searchqueries.push({
        id: nanoid(), type: (is_audio ? 'AUDIO_URL' : 'IMAGE_URL'), displayText: example.displayText || urlTextTrimmed, value: urlTextTrimmed
      })
    }
    if (example.text) {
      let searchTextTrimmed = example.text?.trim();
      _searchqueries.push({ id: nanoid(), type: 'TEXT', displayText: searchTextTrimmed, value: searchTextTrimmed })
    }
    setMultimodalQueries(_searchqueries);
    setSearchText('');
    formRef.current?.submit();
  }

  return <Form
    name="wise-search-examples-form"
    ref={formRef}
    onFinish={onFormSubmit}
  >
    {
      multimodalQueries.map(example => 
        <div className="wise-multimodal-example-query"
          onClick={() => handleExampleMultimodalQueryClick(example)}
          key={example.url}
        >
          {example.url && (is_audio ? <Tag color={QUERY_COLORS['AUDIO_URL']} icon={<SoundOutlined />}> {example.displayText || `Audio sample ${example.url}`}</Tag> : <img src={example.url} />)}
          {example.url && example.text && <span className="wise-multimodal-example-query-plus-sign">+</span>}
          {example.text && <Tag color='geekblue'>{example.text}</Tag>}
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
  {
    id: 'audio',
    label: 'Audio',
    icon: <SoundOutlined />
  },
  // {
  //   id: 'metadata',
  //   label: 'Metadata',
  //   icon: <ControlOutlined /> // or <AlignLeftOutlined /> or <FileTextOutlined />
  // }
]
const { useToken } = theme;

type SearchDropdownRefAttributes = {
  selectModality: (modality: string) => void;
}
const SearchDropdown = forwardRef<SearchDropdownRefAttributes, SearchDropdownProps>(({
  multimodalQueries, setMultimodalQueries,
  searchText, setSearchText,
  handleTextInputChange,
  viewModality,
  featureExtractorId,
  submitSearch, clearSearchBar,
  shotScaleFilter, setShotScaleFilter,
  tourVariables,
  projectInfo,
  isHomePage
}, ref) => {
  const { token } = useToken();
  const dropdownStyle: React.CSSProperties = {
    backgroundColor: token.colorBgElevated,
    borderRadius: token.borderRadiusLG,
    boxShadow: token.boxShadowSecondary
  };

  const [selectedModality, setSelectedModality] = useState<string>('image');
  const [isModalitySelected, setIsModalitySelected] = useState<boolean>(false);

  const [metadataFilterText, setMetadataFilterText] = useState('');
  const [metadataOptions, setMetadataOptions] = useState<{
    label: React.ReactNode;
    options?: { value: string; label: string }[];
  }[]>([]);
  const [isMetadataDropdownVisible, setIsMetadataDropdownVisible] = useState(false);

  const handleMetadataSearch = (value: string) => {
    if (!config.METADATA_TABLE_COLUMNS) {
      setMetadataOptions([]);
      return;
    }

    const lastSpaceIndex = value.lastIndexOf(' ');
    const currentTerm = value.substring(lastSpaceIndex + 1);

    if (currentTerm.includes(':')) {
      setMetadataOptions([]);
      return;
    }

    const allColumns = config.METADATA_TABLE_COLUMNS;
    const filteredColumns = currentTerm
      ? allColumns.filter(col =>
          col.toLowerCase().startsWith(currentTerm.toLowerCase())
        )
      : allColumns;

    if (filteredColumns.length > 0) {
      setMetadataOptions([
        {
          label: <b>Metadata Columns</b>,
          options: filteredColumns.map(col => ({
            value: col,
            label: col,
          })),
        },
      ]);
    } else {
      setMetadataOptions([]);
    }
  };

  const onMetadataSelect = (selectedValue: string) => {
    const currentText = metadataFilterText;
    const lastSpaceIndex = currentText.lastIndexOf(' ');
    const prefix = lastSpaceIndex === -1 ? '' : currentText.substring(0, lastSpaceIndex + 1);
    setMetadataFilterText(prefix + selectedValue + ':');
    setIsMetadataDropdownVisible(false);
  };

  const toggleModality = (modality: string) => {
    if (isModalitySelected && selectedModality === modality) {
      setIsModalitySelected(false);
    } else {
      setIsModalitySelected(true);
      setSelectedModality(modality);
    }
  };
  const selectModality = (modality: string) => {
    setIsModalitySelected(true);
    setSelectedModality(modality);
  }
  useImperativeHandle(ref, () => ({ selectModality }));

  const _submitSearch = () => {
    setIsModalitySelected(false);
    submitSearch();
  };

  const handleShotScaleFilterChange = (e: number[]) => {
    setShotScaleFilter(e);
  }

  const collapseItems = [
    {
      key: 'examples',
      label: 'Examples',
      children: <SearchExamples setMultimodalQueries={setMultimodalQueries} setSearchText={setSearchText} submitSearch={_submitSearch} viewModality={viewModality} featureExtractorId={featureExtractorId} />,
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

  const [_, _featureExtractorId] = featureExtractorId.split('/', 2);
  let _modalities = modalities;
  if (viewModality == 'Image' || viewModality == 'Video') {
    _modalities = _modalities.filter(modality => modality.id != 'audio');
    if (_featureExtractorId.includes('insightface')) {
      _modalities = _modalities.filter(modality => modality.id != 'text');
    }
  } else if (viewModality == 'VideoAudio') {
    _modalities = _modalities.filter(modality => modality.id != 'image');
  }
  return (
    <div style={dropdownStyle}>
      <p style={{ marginTop: 0, color: token.colorTextDescription }}>
        {
          // TODO improve this
          !isHomePage && (multimodalQueries.length > 0 || searchText) ?
            <><PlusOutlined /> Add another modality to your search:</>
            :
            "Search using any of the modalities below, or a combination of modalities:"
        }
      </p>
      <Space style={{ marginBottom: 15 }} ref={tourVariables.multimodalSearchArea}>
        {
          _modalities.map(modality => (
            <Button type="text" size="large" id={`wise-header-${modality.id}-modality-button`}
              key={modality.id}
              className={(isModalitySelected && selectedModality === modality.id) ? 'selected' : isModalitySelected ? 'inactive' : undefined}
              onClick={() => toggleModality(modality.id)}
              ref={(el) => {
                // Set ref to the 'Image' button for the tour
                if (modality.id === 'image') {
                  tourVariables.imageUploadButton.current = el;
                }
              }}
            >
              {modality.icon} {modality.label}
            </Button>
          ))
        }
      </Space>
      <div id="wise-header-modality-collapsible" style={{ maxHeight: isModalitySelected ? undefined : '0' }}>
        {
          selectedModality === 'text' ?
            <TextSearchForm multimodalQueries={multimodalQueries} setMultimodalQueries={setMultimodalQueries}
              searchText={searchText}
              submitSearch={_submitSearch}
              handleTextInputChange={handleTextInputChange}>
              <p style={{ marginTop: 0, color: token.colorTextDescription }}>
                Enter some text in the search bar above or the text box below. You can flexibly describe what you want to look for using natural language.
              </p>
            </TextSearchForm>
            : <MediaSearchForm multimodalQueries={multimodalQueries} setMultimodalQueries={setMultimodalQueries}
              submitSearch={_submitSearch}
              modality={selectedModality} featureExtractorId={featureExtractorId} />
        }
        {/* TODO remove this <br /> */}
        <br />
        {/* <Flex style={{marginBottom: 10, width: '100%'}}>
          
          <span className="wise-spacer"></span>
          <Space>
            <Dropdown.Button
              type="primary"
              menu={{
                items: [{ label: <><MinusCircleFilled style={{color: '#cf1322'}} /> Add as a negative query</>, key: 0 }],
                onClick: (e) => {console.log(' menu click', e)}
              }}
              onClick={(e) => { console.log(e) }}
            >
              Search
            </Dropdown.Button>
          </Space>
        </Flex> */}
      </div>
      {/* fixme: find a better way to disable the shot scale filter if audio is selected */}
      {(projectInfo.shot_based_filters?.shot_scale?.options && viewModality !== 'VideoAudio' && featureExtractorId.split('/')[1] !== 'clap') && (
        <>
          <Tooltip title="Show search results with only frames that match the selected shot scale (e.g. close up)">
            <Divider orientation="left">
              <Space>
                Filter by Shot Scale
                <QuestionCircleOutlined />
              </Space>
            </Divider>
          </Tooltip>
          <div>
            <Checkbox.Group
              value={shotScaleFilter}
              onChange={handleShotScaleFilterChange}
            >
              {projectInfo.shot_based_filters.shot_scale.options.map((value: number) => (
                <Checkbox key={value} value={value}>
                  {config.SHOT_SCALE_FILTER_LABEL?.[value] ?? value}
                </Checkbox>
              ))}
            </Checkbox.Group>
          </div>
        </>
      )}
      {projectInfo.is_metadata_supported && is_metadata_filter_supported(projectInfo, viewModality) && !featureExtractorId.includes('metadata') && (
        <>
          <Tooltip title="Add a metadata filter to restrict the search only to media files that match the metadata query.">
            <Divider orientation="left">
              <Space>
                Filter by Metadata
                <QuestionCircleOutlined />
              </Space>
            </Divider>
          </Tooltip>
          {config.METADATA_TABLE_COLUMNS && (
            <>
              <Space.Compact style={{ width: '100%' }}>
                <AutoComplete
                  value={metadataFilterText}
                  options={metadataOptions}
                  style={{ width: '100%' }}
                  onSelect={onMetadataSelect}
                  onSearch={handleMetadataSearch}
                  onChange={setMetadataFilterText}
                  onFocus={() => {
                    handleMetadataSearch(metadataFilterText);
                    setIsMetadataDropdownVisible(true);
                  }}
                  onKeyDown={(e) => {
                    if (e.ctrlKey && e.key === ' ') {
                      e.preventDefault();
                      handleMetadataSearch(metadataFilterText);
                      setIsMetadataDropdownVisible(true);
                    }
                  }}
                  onBlur={() => setIsMetadataDropdownVisible(false)}
                  open={isMetadataDropdownVisible}
                  placeholder={config.METADATA_FILTER_PLACEHOLDER}
                  popupMatchSelectWidth={false}
                  popupClassName="wise-metadata-autocomplete-dropdown"
                  dropdownRender={(menu) => (
                    <div
                      style={{
                        minWidth: 150,
                        width: 'fit-content',
                        backgroundColor: '#f6f6f6',
                        borderRadius: '8px',
                      }}
                    >
                      {menu}
                    </div>
                  )}
                />
                <Button
                  type="primary"
                  onClick={() => {
                    if (!metadataFilterText.trim()) return;
                    const newQuery = { id: nanoid(), type: 'METADATA' as const, value: metadataFilterText.trim(), displayText: metadataFilterText.trim() };
                    const updatedQueries = [...multimodalQueries, newQuery];
                    setMultimodalQueries(updatedQueries);
                    setMetadataFilterText('');
                    submitSearch(updatedQueries);
                  }}
                >
                  Add Filter
                </Button>
              </Space.Compact>
              <p style={{ marginTop: 5, color: token.colorTextDescription }}>
                {config.METADATA_FILTER_HELP}
              </p>
            </>
          )}
        </>
      )}
      {
        config.MULTIMODAL_EXAMPLE_QUERIES[`${viewModality}:${_featureExtractorId}`] && config.MULTIMODAL_EXAMPLE_QUERIES[`${viewModality}:${_featureExtractorId}`].length > 0 &&
        <>
          <div style={{ borderTop: '1px solid #e3e3e3', marginTop: 20 }} />
          <Collapse items={collapseItems}
            bordered={false} expandIcon={({ isActive }) => <CaretRightOutlined rotate={isActive ? 90 : 0} />}
            activeKey={activeKeys} onChange={_setActiveKeys}
            style={{ background: 'unset' }}
          />
        </>
      }
      {
        // !(multimodalQueries.length > 0 || searchText) &&
        // <div>
        //   <div style={{borderTop: '1px solid #e3e3e3', marginTop: 20}} />
        //   <p style={{color: token.colorTextDescription}}>Examples</p>
        //   {examplesJSX}
        // </div>
      }
      <Flex style={{ marginTop: 15 }}>
        {(multimodalQueries.length > 0 || searchText) ?
          <Button onClick={clearSearchBar}>Clear search</Button>
          : <></>
        }
      </Flex>
    </div>
  )
});



const WiseHeader: React.FunctionComponent<WiseHeaderProps> = ({
  multimodalQueries, setMultimodalQueries, searchText, setSearchText,
  viewModality, setViewModality,
  featureExtractorId, setFeatureExtractorId,
  shotScaleFilter, setShotScaleFilter,
  submitSearch, tourVariables, projectInfo,
  isHomePage = false, isLoadingNewSearch = false
}: WiseHeaderProps) => {
  // This state is set to true when the dropdown is triggered (by hovering over the search bar), and false when the mouse moves outside the search bar
  const [isSearchDropdownTriggered, setIsSearchDropdownTriggered] = useState(false);
  // This state is set to true when the search input field is focused, and false when the input is blurred
  const [isSearchInputFocused, setIsSearchInputFocused] = useState(false);

  const handleSearchTargetChange = (value: string) => {
    const [media_type, feature_extractor_id] = value.split(':');

    const _viewModality = {
      'image': 'Image',
      'video': 'Video',
      'audio': 'VideoAudio'
    }[media_type];
    if (typeof _viewModality !== 'string') {
      console.error('Invalid view modality', media_type);
      return;
    }
    setViewModality(_viewModality as ViewModality);
    setFeatureExtractorId(feature_extractor_id);
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

  const _submitSearch = (_q?: Query[]) => {
    // remove focus from search bar input element, to close the search dropdown
    setIsSearchDropdownTriggered(false);
    tourVariables.searchBar.current.blur();
    submitSearch(_q);
  }

  // Automatically open the search dropdown when the user is dragging a file into the browser window
  const searchDropdownRef = useRef<SearchDropdownRefAttributes>(null);
  const handleDragEnter = (e: DragEvent) => {
    if (e.dataTransfer?.types.includes('Files')) {
      tourVariables.searchBar.current.focus();
      // TODO automatically select modality based on file/mime type?
      searchDropdownRef.current?.selectModality('image')
    }
  }
  useEffect(() => {
    document.addEventListener('dragenter', handleDragEnter);

    return () => {
      document.removeEventListener('dragenter', handleDragEnter);
    }
  }, [handleDragEnter]);

  // useEffect(() => {
  //   // Trigger a search if viewModality was changed
  //   submitSearch();
  // }, [viewModality]);

  const multimodalQueryTags = multimodalQueries.map((query, index) => {
    let icon = <></>;
    if (query.type === 'IMAGE_FILE') icon = <img src={URL.createObjectURL((query.value as unknown) as File)} />;
    else if (query.type === 'IMAGE_URL') icon = <img src={query.value} />;
    else if (query.type === 'AUDIO_FILE') icon = <SoundOutlined />;
    else if (query.type === 'AUDIO_URL') icon = <SoundOutlined />;
    else if (query.type === 'INTERNAL_IMAGE') icon = <img src={query.value.thumbnail} />;

    const tag = <Tag closable
      key={query.id}
      className={(query.type === 'IMAGE_FILE' || query.type === 'IMAGE_URL' || query.type === 'INTERNAL_IMAGE') ? 'wise-search-tag-image' : undefined}
      color={QUERY_COLORS[query.type]}
      icon={icon}
      onClose={(e) => handleTagClose(e, index)}
    >
      {(query.isNegative ? '(Negative) ' : '') + query.displayText}
    </Tag>

    if (query.type === 'IMAGE_FILE') {
      return <Popover content={icon} key={query.id} title="Uploaded image" overlayClassName="wise-search-image-preview">{tag}</Popover>
    } else if (query.type === 'IMAGE_URL') {
      return <Popover content={icon} key={query.id} title="Online image" overlayClassName="wise-search-image-preview">{tag}</Popover>
    } else if (query.type === 'AUDIO_FILE') {
      const popoverPreview = <audio controls src={URL.createObjectURL((query.value as unknown) as File)} />
      return <Popover content={popoverPreview} key={query.id} title="Uploaded audio file">{tag}</Popover>
    } else if (query.type === 'AUDIO_URL') {
      return <Popover content={<audio controls src={query.value} />} key={query.id} title="Online audio file">{tag}</Popover>
    } else if (query.type === 'INTERNAL_IMAGE') {
      return (
        <Popover
          content={
            <div className='wise-image-wrapper'>
              <StillImageView
                imageDetails={query.value}
                isModalView={false}
                boundingBoxes={
                  <BoundingBoxes
                    imageDetails={query.value}
                    isModalView={false}
                    featureExtractorId={featureExtractorId}
                  />
                }
              />
            </div>
          }
          key={query.id}
          title="Internal image"
          overlayClassName="wise-search-image-preview"
        >
          {tag}
        </Popover>
      );
    } else if (query.type === 'TEXT') {
      return <Tooltip title="Text query">{tag}</Tooltip>
    } else if (query.type === 'METADATA') {
      return <Tooltip title="Metadata Filter">{tag}</Tooltip>
    }
  });

  return (
    <Header className="wise-header">
      <div className="wise-header-primary-row" style={isHomePage ? { height: '90px' } : {}}>
        {/* First row: Logo */}
        <div className="wise-header-row wise-header-row-logo">
          <a href="./" id="wise-logo">
            <WiseLogo />
          </a>
        </div>
        {/* Second row: Dropdown */}
        <div className="wise-header-row wise-header-row-dropdown">
          {
            projectInfo.search_targets && (
              Object.keys(projectInfo.search_targets).length > 1 ||
              Object.values(projectInfo.search_targets).some(arr => Array.isArray(arr) && arr.length > 1)
            ) && (
              <Tooltip title="Choose the media track / media type to search on">
                <Select
                  size="large"
                  variant="borderless"
                  className="wise-view-modality-select"
                  value={{ 'Image': 'image', 'Video': 'video', 'Audio': 'audio', 'VideoAudio': 'audio' }[viewModality] + ':' + featureExtractorId}
                  data-testid="wise-search-target-select"
                  onChange={handleSearchTargetChange}
                  options={
                    projectInfo.search_targets
                      ? ['image', 'video', 'audio']
                        .filter((media_type) => projectInfo.search_targets && Object.keys(projectInfo.search_targets).includes(media_type))
                        .map((media_type) => {
                          const key = media_type as ViewModalityKey;
                          return ({
                            label: (
                              <Space>
                                {VIEW_MODALITY_OPTIONS[key]?.icon}
                                {VIEW_MODALITY_OPTIONS[key].longLabel}
                              </Space>
                            ),
                            title: media_type.charAt(0).toUpperCase() + media_type.slice(1),
                            options: (projectInfo.search_targets?.[key] || []).map((feature_extractor_id: string) => {
                              const extra_key = `${media_type}:${feature_extractor_id}` as keyof typeof VIEW_MODALITY_OPTIONS_EXTRA;
                              const default_label = Object.entries(config.PREFERRED_SEARCH_TARGETS_NAME).find(
                                ([key]) => feature_extractor_id.includes(key)
                              )?.[1] ?? (feature_extractor_id.split('/')[1] || feature_extractor_id);
                              const optionTestId = `wise-search-target-select-option-${media_type}:${feature_extractor_id}`;
                              return {
                                ...VIEW_MODALITY_OPTIONS[key],
                                label: (
                                  <Space data-testid={optionTestId}>
                                    {VIEW_MODALITY_OPTIONS_EXTRA[extra_key]?.icon || VIEW_MODALITY_OPTIONS[key]?.icon}
                                    {VIEW_MODALITY_OPTIONS_EXTRA[extra_key]?.label || default_label}
                                  </Space>
                                ),
                                value: extra_key,
                              }
                            }),
                          })
                        })
                      : []
                  }
                  popupMatchSelectWidth={false}
                />
              </Tooltip>
            )}
        </div>
        {/* Third row: Search input */}
        <div className="wise-header-row wise-header-row-search">
          <Dropdown
            overlayClassName="wise-search-dropdown"
            dropdownRender={_ =>
              <SearchDropdown
                multimodalQueries={multimodalQueries} setMultimodalQueries={setMultimodalQueries}
                searchText={searchText} setSearchText={setSearchText}
                handleTextInputChange={setSearchText}
                viewModality={viewModality}
                featureExtractorId={featureExtractorId}
                submitSearch={_submitSearch} clearSearchBar={clearSearchBar}
                shotScaleFilter={shotScaleFilter} setShotScaleFilter={setShotScaleFilter}
                isHomePage={isHomePage}
                tourVariables={tourVariables} ref={searchDropdownRef}
                projectInfo={projectInfo}
              />
            }
            open={isSearchDropdownTriggered || isSearchInputFocused || tourVariables.isSearchDropdownOpenForTour}
            onOpenChange={(open) => setIsSearchDropdownTriggered(open)}
          >
            <Form onFinish={() => _submitSearch()} id="search-input-form">
              <Input
                id="search-input"
                autoComplete="off"
                size={isHomePage ? 'large' : 'middle'}
                placeholder={multimodalQueries.length === 0 ? 'Search' : ''}
                value={searchText}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchText(e.target.value)}
                prefix={multimodalQueryTags}
                suffix={
                  <>
                    {
                      (multimodalQueries.length > 0 || searchText) &&
                      <Tooltip title="Clear">
                        <Button type="text" shape="circle" size="large" icon={<CloseOutlined />} onClick={clearSearchBar} />
                      </Tooltip>
                    }
                    <Button type="text" shape="circle" size="large" loading={isLoadingNewSearch} htmlType="submit" icon={<SearchOutlined />} />
                  </>
                }
                ref={tourVariables.searchBar}
                onFocus={() => setIsSearchInputFocused(true)}
                onBlur={() => setIsSearchInputFocused(false)}
              />
            </Form>
          </Dropdown>
        </div>
        <span className="wise-spacer"></span>
      </div>
    </Header>
  )
};

export default WiseHeader;

