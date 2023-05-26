import { useState } from "react";
import { Button } from "antd";
import { EyeInvisibleOutlined } from "@ant-design/icons";

import './SensitiveImageWarning.scss';

const SensitiveImageWarning = ({ isSensitive }: { isSensitive: boolean }) => {
  const [isVisible, setIsVisible] = useState(true);

  if (isSensitive && isVisible) {
    return <div className="wise-sensitive-image-warning">
      <EyeInvisibleOutlined />
      <span>Warning: this image may contain sensitive content</span>
      <Button ghost onClick={() => setIsVisible(false)}>View</Button>
    </div>
  } else {
    return <></>
  }
}

export default SensitiveImageWarning;
