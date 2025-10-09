<script setup lang="ts">
import { watch } from 'vue'
import { useLocalStorage } from '@vueuse/core'

const enabled = useLocalStorage('a11y-mode', false)

watch(enabled, (v) => {
  if (v) {
    document.documentElement.setAttribute('data-a11y', 'true')
  } else {
    document.documentElement.removeAttribute('data-a11y')
  }
}, {
  immediate: true
})
</script>

<template>
  <a-button>
    <a-checkbox v-model:checked="enabled">
      A11y Mode
    </a-checkbox>
  </a-button>
</template>
