<script setup lang="ts">
import { useStore } from '../stores'

const store = useStore()
</script>

<template>
  <div class="flex items-center justify-end gap-3 text-gray-500 text-sm">
    <div class="flex items-center gap-1">
      <div class="w-3 h-3 rounded-full bg-alive transition-colors" />
      <span>ALIVE</span>
    </div>
    <span class="text-gray-400">/</span>
    <div class="flex items-center gap-1">
      <div class="w-3 h-3 rounded-full bg-coma transition-colors" />
      <span>COMA</span>
    </div>
    <span class="text-gray-400">/</span>
    <div class="flex items-center gap-1">
      <div class="w-3 h-3 rounded-full bg-dead transition-colors" />
      <span v-if="store.isNova">STALE</span>
      <span v-else>DEAD</span>
    </div>
  </div>
</template>
