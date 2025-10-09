<script setup lang="ts">
import { computed, h } from 'vue'
import { useTimeAgo } from '@vueuse/core'
import { Modal } from 'ant-design-vue'

import { useStore } from '../stores'
import { formatTimestamp } from '../utils/datetime.ts'
import type { SessionData } from '../types/controller.ts'

const store = useStore()

const props = defineProps<{
  id: number
  value: SessionData
}>()

const startedAt = computed<Date>(() => {
  return formatTimestamp(props.value.start)
})

const startedAtFormatted = useTimeAgo(startedAt, {
  showSecond: true,
  updateInterval: 1000
})

const lastUpdatedAt = computed<Date>(() => {
  return formatTimestamp(props.value.last_update)
})

const lastUpdatedAtFormatted = useTimeAgo(lastUpdatedAt, {
  showSecond: true,
  updateInterval: 1000
})

const isCustom = computed<boolean>(() => {
  return props.value.index === -1
})

const handleShowCustomTask = () => {
  const content = h('code', {
    class: 'whitespace-pre overflow-x-auto'
  }, JSON.stringify(props.value.custom_task, null, 2))

  Modal.info({
    title: `Custom Task of Session #${props.id}`,
    content: content,
    width: '80%',
    maskClosable: true
  })
}

const handleCancel = () => {
  fetch(`/api/cancel`, {
    method: 'POST',
    headers: {
      session_id: props.id.toString()
    }
  })
}
</script>

<template>
  <div class="flex items-stretch gap-4 px-4 py-2 last:rounded-b-lg hover:bg-black/5">
    <div class="flex-1 flex flex-col justify-center">
      <p class="flex items-center gap-2 m-0 text-sm text-gray-800 font-medium">
        Session #{{ id }}
        <span class="text-gray-400 font-semibold">/</span>
        Task #{{ isCustom ? 'custom' : value.index }}
        <template v-if="store.isNova">
          <a-tag
            v-if="value.cancelling"
            :bordered="false"
            color="red"
            class="!m-0"
          >
            C
          </a-tag>
          <a-tag
            v-else-if="value.locked"
            :bordered="false"
            color="blue"
            class="!m-0"
          >
            E
          </a-tag>
          <a-tag
            v-else
            :bordered="false"
            color="orange"
            class="!m-0"
          >
            A
          </a-tag>
        </template>
      </p>
      <p class="m-0 text-xs text-gray-500">
        Started {{ startedAtFormatted }}
        <br class="md:hidden" />
        <span class="hidden md:inline-block mx-1 text-gray-400 font-semibold">/</span>
        Updated {{ lastUpdatedAtFormatted }}
      </p>
    </div>
    <div class="shrink-0 flex flex-col justify-center items-stretch gap-2">
      <a-button
        v-if="isCustom"
        type="primary"
        size="small"
        ghost
        @click="handleShowCustomTask"
      >
        View
      </a-button>
      <a-popconfirm
        title="Are you sure you want to cancel this session?"
        @confirm="handleCancel"
      >
        <a-button
          danger
          size="small"
          ghost
        >
          Cancel
        </a-button>
      </a-popconfirm>
    </div>
  </div>
</template>
