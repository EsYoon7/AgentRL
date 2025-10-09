<script setup lang="ts">
import { computed } from 'vue'
import { DateTime } from 'luxon'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { ComposeOption } from 'echarts/core'
import type { LineSeriesOption } from 'echarts/charts'
import type {
  TitleComponentOption,
  TooltipComponentOption,
  LegendComponentOption,
  GridComponentOption
} from 'echarts/components'

import { useStore } from '../stores'

use([
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  LineChart,
  CanvasRenderer
])

type EChartsOption = ComposeOption<
  | TitleComponentOption
  | TooltipComponentOption
  | LegendComponentOption
  | GridComponentOption
  | LineSeriesOption
>

const store = useStore()

// Generate a random color based on a string (task name)
const stringToColor = (str: string): string => {
  // Simple hash function
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash)
  }

  // Convert to color
  let color = '#'
  for (let i = 0; i < 3; i++) {
    const value = '00' + ((hash >> (i * 8)) & 0xFF).toString(16)
    color += value.substring(value.length - 2)
  }

  return color
}

// Get unique tasks from history
const uniqueTasks = computed(() => {
  const tasks = new Set<string>()

  store.taskSessionsHistory.history.forEach(item => {
    Object.keys(item.snapshot).forEach(taskName => {
      tasks.add(taskName)
    })
  })

  return Array.from(tasks)
})

const options = computed<EChartsOption>(() => ({
  title: {
    text: 'Sessions History by Task',
    textStyle: {
      fontSize: 14
    }
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '10%',
    containLabel: true
  },
  tooltip: {
    trigger: 'axis',
    confine: true,
    formatter: (params) => {
      // @ts-ignore
      const timestamp = params[0].data[0]
      let result = `${DateTime.fromMillis(timestamp).toFormat('HH:mm:ss')}<br/>`

      // @ts-ignore
      params.forEach((param) => {
        result += `${param.seriesName}: ${param.data[1]}<br/>`
      })

      return result
    }
  },
  legend: {
    data: uniqueTasks.value,
    type: 'scroll',
    orient: 'horizontal',
    bottom: 0
  },
  xAxis: {
    type: 'time',
    axisLabel: {
      formatter: '{HH}:{mm}:{ss}'
    }
  },
  yAxis: {
    type: 'value',
    minInterval: 1
  },
  series: uniqueTasks.value.map((taskName) => {
    // Get data points for this task
    const data = store.taskSessionsHistory.history
      .filter(item => item.snapshot[taskName] !== undefined)
      .map(item => [item.timestamp, item.snapshot[taskName] || 0])
      .sort((a, b) => a[0] - b[0])

    return {
      name: taskName,
      type: 'line',
      showSymbol: false,
      lineStyle: {
        width: 2
      },
      itemStyle: {
        color: stringToColor(taskName)
      },
      areaStyle: {
        opacity: 0.1
      },
      data: data
    }
  })
}))
</script>

<template>
  <v-chart
    class="w-full !h-60 bg-white p-4 rounded-lg border border-neutral-200"
    autoresize
    :option="options"
    :update-options="{
      notMerge: false,
    }"
  />
</template>
