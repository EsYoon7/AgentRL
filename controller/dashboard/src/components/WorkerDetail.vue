<script setup lang="ts">
import { useStore } from '../stores'
import type { WorkerData } from '../types/controller.ts'

const props = defineProps<{
  taskName: string
  value: WorkerData
}>()

const store = useStore()

const handleMarkStale = async () => {
  await fetch(`/api/mark_stale`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      name: props.taskName,
      worker_id: props.value.id,
      stale: !props.value.stale
    }),
  })
  await store.fetchWorkers()
}
</script>

<template>
  <div class="flex items-stretch gap-4 px-4 py-2">
    <div class="flex-1 flex flex-col">
      <p class="flex items-center gap-2 m-0 text-sm text-gray-800 font-medium">
        Sessions of Worker #{{ value.id }}
      </p>
      <p class="m-0 text-xs text-gray-500">
        {{ value.address }}
      </p>
    </div>
    <div
      v-if="store.isNova"
      class="shrink-0 flex flex-col justify-center items-stretch gap-2"
    >
      <a-button
        v-if="value.stale !== undefined"
        type="primary"
        size="small"
        ghost
        @click="handleMarkStale"
      >
        {{ value.stale ? 'Unstale' : 'Stale' }}
      </a-button>
    </div>
  </div>
</template>
