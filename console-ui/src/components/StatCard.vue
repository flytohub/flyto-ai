<script setup lang="ts">
defineProps<{
  label: string
  value: string | number
  sub?: string
  color?: string
}>()
</script>

<template>
  <div class="bg-gray-900 border border-gray-800 rounded-xl p-4">
    <p class="text-xs text-gray-500 uppercase tracking-wide">{{ label }}</p>
    <p class="text-2xl font-bold mt-1" :class="color || 'text-gray-100'">{{ value }}</p>
    <p v-if="sub" class="text-xs text-gray-500 mt-1">{{ sub }}</p>
  </div>
</template>
