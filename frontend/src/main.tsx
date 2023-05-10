import React from 'react';
import ReactDOM from 'react-dom/client';
import { ConfigProvider } from 'antd';

import { App } from './App.tsx';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <ConfigProvider
        theme={{
            token: {
                fontFamily: "'Noto Sans', Helvetica, Arial, sans-serif",
            },
        }}
    >
      <App />
    </ConfigProvider>
  </React.StrictMode>,
)
