import { createApp } from 'vue'
import { createPinia } from 'pinia'
import {
  App as AntApp,
  Button,
  Checkbox,
  Collapse,
  Dropdown,
  Modal,
  Empty,
  Popconfirm,
  Slider,
  Spin,
  StyleProvider,
  Switch,
  Tag,
} from 'ant-design-vue'

import App from './App.vue'
import './styles/app.css'

const app = createApp(App)

const pinia = createPinia()
app.use(pinia)

app.use(AntApp)
app.use(Button)
app.use(Checkbox)
app.use(Collapse)
app.use(Dropdown)
app.use(Empty)
app.use(Modal)
app.use(Popconfirm)
app.use(Slider)
app.use(Spin)
app.use(StyleProvider)
app.use(Switch)
app.use(Tag)

app.mount('#app')
