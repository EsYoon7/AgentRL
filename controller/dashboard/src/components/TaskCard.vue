<script setup lang="ts">
import { computed, ref, type ComputedRef } from 'vue'
import { Empty, Modal } from 'ant-design-vue'

import { useStore } from '../stores'
import { getHealth, WorkerHealth } from '../utils/health.ts'
import SessionListItem from './SessionListItem.vue'
import WorkerButton from './WorkerButton.vue'
import WorkerDetail from './WorkerDetail.vue'

import type { SessionData, TaskData, WorkerData } from '../types/controller.ts'

const store = useStore()

const props = withDefaults(defineProps<{
  name: string
  value: TaskData
  expanded?: boolean
}>(), {
  expanded: true
})

const emit = defineEmits<{
  'update:expanded': [value: boolean]
}>()

const expandedKeys = computed(() => {
  return props.expanded ? [props.name] : []
})

const updateExpandedKeys = (value: string[]) => {
  emit('update:expanded', value.includes(props.name))
}

const workers = computed<WorkerData[]>(() => {
  return Object.values(props.value.workers)
})

const sessionEnvironmentCount = computed<number>(() => {
  if (store.isNova && store.sessions && Object.keys(store.sessions).length) {
    return Object.values(store.sessions).filter((session) => {
      return session.name === props.name && session.locked
    }).length
  } else {
    return workers.value.reduce((acc, worker) => acc + worker.current, 0)
  }
})

const sessionAgentCount = computed<number>(() => {
  if (store.isNova && store.sessions && Object.keys(store.sessions).length) {
    return Object.values(store.sessions).filter((session) => {
      return session.name === props.name && !session.locked
    }).length
  } else {
    return 0
  }
})

const sessionCapacity = computed<number>(() => {
  return workers.value
    .reduce((acc, worker) => {
      if (getHealth(worker) === WorkerHealth.ALIVE) {
        return acc + worker.capacity
      }
      return acc
    }, 0)
})

const sessionEnvironmentOccupancy = computed<number>(() => {
  return Math.min(sessionEnvironmentCount.value / sessionCapacity.value, 1) * 100
})

const sessionAgentOccupancy = computed<number>(() => {
  return Math.min(sessionAgentCount.value / sessionCapacity.value, 1) * 100
})

const useHealthData = (health: WorkerHealth): [ComputedRef<number>, ComputedRef<number>] => {
  const count = computed<number>(() => {
    return workers.value.filter((worker) => {
      return getHealth(worker) === health
    }).length
  })

  const rate = computed<number>(() => {
    if (!workers.value.length) {
      return 0
    }

    return (count.value / workers.value.length) * 100
  })

  return [count, rate]
}

const [aliveCount, aliveRate] = useHealthData(WorkerHealth.ALIVE)
const [comaCount, comaRate] = useHealthData(WorkerHealth.COMA)
const [deadCount, deadRate] = useHealthData(WorkerHealth.DEAD)

const handleShowIndices = () => {
  Modal.info({
    title: `Indices of ${props.name}`,
    content: props.value.indices.join(', '),
    width: '80%',
    maskClosable: true
  })
}

const activeWorker = ref<number | null>(null)

const handleToggleWorker = (workerId: number) => {
  if (activeWorker.value === workerId) {
    activeWorker.value = null
  } else {
    activeWorker.value = workerId
  }
}

const sessions = computed<Record<number, SessionData>>(() => {
  if (activeWorker.value === null || !store.sessions) {
    return {}
  }

  return Object.entries(store.sessions).reduce((sessions, [sessionId, session]) => {
    if (session.name === props.name && session.worker_id === activeWorker.value) {
      sessions[Number(sessionId)] = session
    }

    return sessions
  }, {} as Record<number, SessionData>)
})

const handleCancelAll = async () => {
  await fetch('/api/cancel_all?' + new URLSearchParams({
    name: props.name
  }).toString(), {
    method: 'POST'
  })
}
</script>

<template>
  <a-collapse
    :active-key="expandedKeys"
    expand-icon-position="end"
    @update:active-key="updateExpandedKeys"
  >
    <a-collapse-panel :key="name">
      <template #header>
        <div class="flex items-center gap-2 select-none">
          <h2
            class="!m-0 !text-lg !font-medium"
            v-text="name"
          />
          <a-button
            size="small"
            @click.stop="handleShowIndices"
          >
            {{ value.indices.length }}
          </a-button>
          <a-popconfirm
            v-if="store.isNova"
            title="Are you sure you want to cancel all sessions of this task?"
            @confirm="handleCancelAll"
          >
            <a-button
              size="small"
              danger
              @click.stop
            >
              Cancel
            </a-button>
          </a-popconfirm>
        </div>
      </template>

      <template #extra>
        <div class="flex flex-col items-stretch text-xs text-gray-600 gap-1 select-none">
          <div class="flex justify-end items-center gap-2">
            <span v-if="store.isNova">Sessions: {{ sessionEnvironmentCount }}+{{ sessionAgentCount }}/{{ sessionCapacity }}</span>
            <span v-else>Sessions: {{ sessionEnvironmentCount }}/{{ sessionCapacity }}</span>
            <div class="h-2.5 w-[20vw] max-w-48 flex rounded-full border-x border-y-2 border-white bg-white overflow-clip">
              <div
                v-if="sessionEnvironmentCount"
                class="h-full bg-blue-600 first:rounded-s-full last:rounded-e-full border-x border-white transition-all duration-200"
                :style="{ width: `${sessionEnvironmentOccupancy}%` }"
              />
              <div
                v-if="sessionAgentCount"
                class="h-full bg-blue-400 first:rounded-s-full last:rounded-e-full border-x border-white transition-all duration-200"
                :style="{ width: `${sessionAgentOccupancy}%` }"
              />
            </div>
          </div>
          <div class="flex justify-end items-center gap-2">
            <span>Workers: {{ aliveCount }}/{{ comaCount }}/{{ deadCount }}</span>
            <div class="h-2.5 w-[20vw] max-w-48 flex rounded-full border-x border-y-2 border-white bg-white overflow-clip">
              <div
                v-if="aliveCount"
                class="h-full bg-alive first:rounded-s-full last:rounded-e-full border-x border-white transition-all duration-200"
                :style="{ width: `${aliveRate}%` }"
              />
              <div
                v-if="comaCount"
                class="h-full bg-coma first:rounded-s-full last:rounded-e-full border-x border-white transition-all duration-200"
                :style="{ width: `${comaRate}%` }"
              />
              <div
                v-if="deadCount"
                class="h-full bg-dead first:rounded-s-full last:rounded-e-full border-x border-white transition-all duration-200"
                :style="{ width: `${deadRate}%` }"
              />
            </div>
          </div>
        </div>
      </template>

      <div class="flex flex-col sm:flex-row items-stretch sm:items-start gap-4">
        <div
          class="sm:flex-3/7 flex flex-wrap gap-3 justify-between sm:justify-start justify-items-stretch transition-all duration-200"
          :class="activeWorker === null ? '' : ''"
        >
          <worker-button
            v-for="worker of value.workers"
            :key="worker.id"
            :value="worker"
            :active="activeWorker === worker.id"
            @click="handleToggleWorker(worker.id)"
          />
        </div>
        <transition
          enter-active-class="transition-all duration-200 translate-gpu"
          leave-active-class="transition-all duration-200 translate-gpu"
          enter-to-class="translate-x-0 opacity-100"
          leave-from-class="translate-x-0 opacity-100"
          enter-from-class="translate-x-8 opacity-0"
          leave-to-class="translate-x-8 opacity-0"
        >
          <div
            v-if="activeWorker !== null"
            class="sm:flex-4/7 border border-gray-300 rounded-lg"
          >
            <template
              v-for="worker of value.workers"
              :key="worker.id"
            >
              <worker-detail
                v-if="worker.id === activeWorker"
                :task-name="name"
                :value="worker"
              />
            </template>
            <transition-group
              v-if="Object.keys(sessions).length"
              tag="div"
              class="flex flex-col items-stretch border-t border-gray-200 divide-y divide-gray-200"
              enter-active-class="transition-all duration-200 translate-gpu"
              move-class="transition-all duration-200 translate-gpu"
              leave-active-class="absolute transition-all duration-200 translate-gpu"
              enter-to-class="scale-none translate-x-0 opacity-100"
              leave-from-class="scale-none translate-x-0 opacity-100"
              enter-from-class="scale-y-5 translate-x-8 opacity-0"
              leave-to-class="scale-y-5 translate-x-8 opacity-0"
            >
              <session-list-item
                v-for="(session, sessionId) of sessions"
                :key="sessionId"
                :id="Number(sessionId)"
                :value="session"
              />
            </transition-group>
            <div
              v-else-if="!store.sessions && store.isFetchingSessions"
              class="flex justify-center items-center gap-4"
            >
              <a-spin />
              <h4>Fetching Sessions...</h4>
            </div>
            <a-empty
              v-else
              :image="Empty.PRESENTED_IMAGE_SIMPLE"
              description="No Sessions Available"
            />
          </div>
        </transition>
      </div>
    </a-collapse-panel>
  </a-collapse>
</template>
