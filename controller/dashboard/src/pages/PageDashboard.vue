<script setup lang="ts">
import { ref, watch } from 'vue'

import { useStore } from '../stores'
import RefreshSetting from '../components/RefreshSetting.vue'
import SessionHistoryChart from '../components/SessionHistoryChart.vue'
import TaskCard from '../components/TaskCard.vue'
import WorkerStatusLegend from '../components/WorkerStatusLegend.vue'

const store = useStore()

const handleSyncAll = async () => {
  await fetch('/api/sync_all', {
    method: 'POST'
  })
}

const handleCancelAll = async () => {
  await fetch('/api/cancel_all?force=true', {
    method: 'POST'
  })
}

const expandedTasks = ref<string[]>([])

watch(() => store.tasks, (v) => {
  if (v && Object.keys(v).length === 1 && !expandedTasks.value.length) {
    expandedTasks.value = Object.keys(v)
  }
}, {
  immediate: true
})

const updateExpandedTasks = (taskName: string, value: boolean) => {
  if (value) {
    expandedTasks.value.push(taskName)
  } else {
    expandedTasks.value = expandedTasks.value.filter((name) => name !== taskName)
  }
}
</script>

<template>
  <div class="max-w-5xl mx-auto p-8 flex flex-col gap-6">
    <div class="flex flex-col items-stretch gap-6 lg:flex-row lg:items-center">
      <h1 class="!m-0 lg:flex-1">
        <span
          class="inline-block bg-gradient-to-br from-[#12c2e9] via-[#c471ed] to-[#f64f59] text-transparent bg-clip-text"
        >
          AgentRL
        </span>
        Controller
      </h1>
      <div class="flex justify-end items-center gap-4 flex-wrap">
        <refresh-setting />
        <a-button @click="handleSyncAll">
          Sync All
        </a-button>
        <a-popconfirm
          title="Are you sure you want to cancel all sessions in all tasks? This might be dangerous!"
          @confirm="handleCancelAll"
        >
          <a-button danger>
            Cancel All
          </a-button>
        </a-popconfirm>
      </div>
    </div>

    <template v-if="store.tasks && Object.keys(store.tasks).length">
      <session-history-chart />

      <task-card
        v-for="(task, taskName) of store.tasks"
        :key="taskName"
        :name="taskName"
        :value="task"
        :expanded="expandedTasks.includes(taskName)"
        @update:expanded="updateExpandedTasks(taskName, $event)"
      />

      <worker-status-legend />
    </template>

    <div
      v-else-if="store.isFetchingWorkers"
      class="flex items-center justify-center gap-4"
    >
      <a-spin size="large" />
      <h2>Loading Workers...</h2>
    </div>

    <a-empty
      v-else
      description="No Workers Available"
    />
  </div>
</template>
