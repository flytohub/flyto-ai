<script setup lang="ts">
defineProps<{
  moduleId: string
  status: 'running' | 'ok' | 'failed' | 'healed'
  duration?: number
  error?: string
  fixHint?: string
}>()

const statusConfig = {
  running: { icon: '⏳', color: 'text-blue-400', bg: 'bg-blue-900/20' },
  ok: { icon: '✅', color: 'text-green-400', bg: 'bg-green-900/20' },
  failed: { icon: '❌', color: 'text-red-400', bg: 'bg-red-900/20' },
  healed: { icon: '🩹', color: 'text-amber-400', bg: 'bg-amber-900/20' },
}
</script>

<template>
  <div
    class="flex items-center gap-2 px-3 py-2 rounded-lg text-sm"
    :class="statusConfig[status].bg"
  >
    <span>{{ statusConfig[status].icon }}</span>
    <span class="flex-1 font-mono text-xs" :class="statusConfig[status].color">
      {{ moduleId }}
    </span>
    <span v-if="duration" class="text-xs text-gray-500">{{ duration }}ms</span>
  </div>
  <div v-if="error" class="ml-8 text-xs text-red-400 mt-0.5">{{ error }}</div>
  <div v-if="fixHint" class="ml-8 text-xs text-amber-400 mt-0.5">EMS: {{ fixHint }}</div>
</template>
