<script setup lang="ts">
import { computed } from 'vue'

import { getHealth, WorkerHealth } from '../utils/health.ts'

import type { WorkerData } from '../types/controller.ts'

const props = withDefaults(defineProps<{
  value: WorkerData
  active?: boolean
}>(), {
  active: false
})

const health = computed<WorkerHealth>(() => {
  return getHealth(props.value)
})

const filledColors = computed<string>(() => {
  if (health.value === WorkerHealth.ALIVE) {
    return '!bg-alive !text-white'
  } else if (health.value === WorkerHealth.COMA) {
    return '!bg-coma !text-white'
  } else if (health.value === WorkerHealth.DEAD) {
    return '!bg-dead !text-white'
  }
  return ''
})

const blankColors = computed<string>(() => {
  if (health.value === WorkerHealth.ALIVE) {
    return '!border-alive !text-alive'
  } else if (health.value === WorkerHealth.COMA) {
    return '!border-coma !text-coma'
  } else if (health.value === WorkerHealth.DEAD) {
    return '!border-dead !text-dead'
  }
  return ''
})

const buttonClasses = computed<string>(() => {
  return props.active ? (filledColors.value + ' !border-white') : blankColors.value
})

const workerIdClasses = computed<string>(() => {
  return filledColors.value + (props.active ? ' border-e border-white' : '')
})
</script>

<template>
  <a-button
    type="primary"
    class="min-w-30 !h-10 !p-0 !flex items-stretch"
    :ghost="!active"
    :class="buttonClasses"
  >
    <div
      class="flex-1/3 text-center flex items-center justify-center rounded-s-[5px] transition-inherit font-bold"
      :class="workerIdClasses"
    >
      {{ value.id % 10000 }}
    </div>
    <div class="flex-2/3 flex items-center justify-center">
      {{ value.current }} / {{ value.capacity }}
    </div>
  </a-button>
</template>
