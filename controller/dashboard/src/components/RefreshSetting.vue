<script setup lang="ts">
import { computed, ref } from 'vue'
import { ReloadOutlined } from '@ant-design/icons-vue'

import { useStore } from '../stores'

const store = useStore()

const settingsVisible = ref(false)

const seconds = computed<string>(() => {
  if (!store.refreshInterval) {
    return 'Off'
  }

  return `${Math.round(store.refreshInterval / 1000)}s`
})

const handleOpenSettings = () => {
  settingsVisible.value = true
}

const handleRefresh = async () => {
  await store.fetchWorkers()
  await store.fetchSessions()
}

const refreshIntervalValue = computed(() => {
  if (!store.refreshInterval) {
    return 31
  }

  return Math.round(store.refreshInterval / 1000)
})

const handleRefreshInterval = (value: number) => {
  store.refreshInterval = value * 1000
  if (value === 31) {
    store.refreshInterval = 0
  }
}
</script>

<template>
  <div class="flex items-stretch">
    <a-button
      :class="store.refreshInterval ? '' : '!rounded-e-none !border-e-0'"
      @click="handleOpenSettings"
    >
      Auto Refresh: {{ seconds }}
    </a-button>

    <a-button
      v-if="!store.refreshInterval"
      class="!rounded-s-none"
      :loading="store.isFetchingWorkers || store.isFetchingSessions"
      @click="handleRefresh"
    >
      <template #icon>
        <reload-outlined />
      </template>
    </a-button>

    <a-modal
      v-model:open="settingsVisible"
      title="Auto Refresh Settings"
      :footer="null"
    >
      <p class="text-sm text-gray-800 mb-4">
        Refresh Interval: {{ seconds }}
      </p>
      <a-slider
        :value="refreshIntervalValue"
        :min="1"
        :max="31"
        :step="1"
        :marks="{
        1: '1s',
        5: '5s',
        10: '10s',
        20: '20s',
        31: 'Off',
      }"
        @update:value="handleRefreshInterval"
      />
    </a-modal>
  </div>
</template>
