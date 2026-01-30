// Copyright 2026 University of Oxford
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { useState } from 'react';
import { FlagTwoTone } from '@ant-design/icons';
import { Modal, Form, Checkbox, Alert, message } from 'antd';

import { ReportImageModalProps } from './types';
import './ReportImageModal.scss';

const reasons = [
  'Nudity or sexual content',
  'Violent or repulsive content',
  'Hateful or offensive content',
  'Irrelevant/unrelated to the search query',
  'Other'
].map(x => ({label: x, value: x}));

const reasonsHomePageVersion = reasons.filter((_, i) => i !== 3); // Don't show the fourth element on the home page

const ReportImageModal: React.FunctionComponent<ReportImageModalProps> = ({dataService, isHomePage, selectedImageId, setSelectedImageId}) => {
  // Note: the modal is open when selectedImageId has a value (i.e. is not undefined). Calling setSelectedImageId() sets selectedImageId to undefined which closes the modal.

  const [form] = Form.useForm();
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const [messageApi, contextHolder] = message.useMessage();

  const handleSubmit = async () => {
    const formValues = await form.validateFields();
    if (!selectedImageId) {
      setErrorMessage('Application error: could not find id of selected image.')
      return;
    }
    let responseText;
    try {
      setConfirmLoading(true);
      responseText = await dataService.reportImage(selectedImageId, formValues.reasons);
    } catch (e) {
      console.error(e);
      setConfirmLoading(false);
      setErrorMessage('Submission failed. Please try again later.')
      return;
    }
    form.resetFields();
    setConfirmLoading(false);
    setErrorMessage('');
    setSelectedImageId(); // This closes the modal
    messageApi.open({
      type: 'success',
      content: responseText
    });
  }

  const handleCancel = () => {
    form.resetFields();
    setConfirmLoading(false); // TODO abort the fetch request in dataService.reportImage?
    setErrorMessage('');
    setSelectedImageId(); // This closes the modal
  }
  
  return <>
    <Modal title={<><FlagTwoTone twoToneColor="#faad14" /><span>Report image</span></>}
                  open={!!selectedImageId} closable={true} maskClosable={true}
                  okText='Submit'
                  onOk={handleSubmit}
                  onCancel={handleCancel}
                  confirmLoading={confirmLoading}
                  className='wise-report-image-modal'>
      <Form form={form} layout="vertical" requiredMark={false}>
        <Form.Item
          name="reasons"
          label="Please select the reason(s) for reporting this image:"
          required
          rules={[{ required: true, message: 'Please select one of the options above' }]}
        >
          <Checkbox.Group options={isHomePage ? reasonsHomePageVersion : reasons} />
        </Form.Item>
      </Form>
      { errorMessage ? <Alert message={errorMessage} type="error" showIcon /> : <></>}
    </Modal>
    {contextHolder /* for success message */}
  </>;
}

export default ReportImageModal;