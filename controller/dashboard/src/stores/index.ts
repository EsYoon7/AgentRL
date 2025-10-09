import { computed, watch } from 'vue'
import { defineStore } from 'pinia'
import { useFetch, useLocalStorage, useRefHistory, useTimeoutPoll } from '@vueuse/core'

import type { GetVersionResponse, ListSessionsResponse, ListWorkersResponse, SessionData } from '../types/controller.ts'

export const useStore = defineStore('main', () => {
  const refreshInterval = useLocalStorage('refresh-interval', 2000)

  const {
    execute: fetchWorkers,
    isFetching: isFetchingWorkers,
    data: tasks,
  } = useFetch('/api/list_workers', {
    initialData: {},
    immediate: true
  }).json<ListWorkersResponse>()

  const {
    pause: pauseRefreshWorkers,
    resume: resumeRefreshWorkers,
  } = useTimeoutPoll(fetchWorkers, refreshInterval, {
    immediate: false
  })

  const {
    execute: fetchSessions,
    isFetching: isFetchingSessions,
    data: sessions,
  } = useFetch('/api/list_sessions', {
    initialData: {},
    immediate: true
  }).json<ListSessionsResponse>()

  const {
    pause: pauseRefreshSessions,
    resume: resumeRefreshSessions,
  } = useTimeoutPoll(fetchSessions, refreshInterval, {
    immediate: false
  })

  // Track history per task
  const taskSessionsHistory = useRefHistory(sessions, {
    capacity: 500, // Store last 500 data points
    dump: (v) => {
      const counts: Record<string, number> = {}

      if (tasks.value) {
        Object.keys(tasks.value).forEach((name) => {
          counts[name] = 0
        })
      }

      if (v) {
        Object.values(v).forEach((session: SessionData) => {
          if (!counts[session.name]) {
            counts[session.name] = 1
          } else {
            counts[session.name]++
          }
        })
      }

      return counts
    }
  })

  watch(refreshInterval, (v) => {
    if (v) {
      resumeRefreshWorkers()
      resumeRefreshSessions()
    } else {
      pauseRefreshWorkers()
      pauseRefreshSessions()
    }
  }, {
    immediate: true
  })

  const {
    data: version
  } = useFetch('/api/version', {
    immediate: true
  }).json<GetVersionResponse>()

  const isNova = computed<boolean>(() => {
    return version.value?.variant.toLowerCase() === 'nova'
  })

  return {
    refreshInterval,
    fetchSessions,
    fetchWorkers,
    isFetchingSessions,
    isFetchingWorkers,
    sessions,
    tasks,
    version,
    isNova,
    taskSessionsHistory
  }
})
