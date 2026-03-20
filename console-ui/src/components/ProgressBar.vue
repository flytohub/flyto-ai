<script setup lang="ts">
defineProps<{
  value: number
  max: number
  label?: string
  color?: string
}>()
</script>

<template>
  <div>
    <div v-if="label" class="flex justify-between text-xs text-gray-500 mb-1">
      <span>{{ label }}</span>
      <span>{{ value.toLocaleString() }} / {{ max.toLocaleString() }}</span>
    </div>
    <div class="h-2 bg-gray-800 rounded-full overflow-hidden">
      <div
        class="h-full rounded-full transition-all duration-500"
        :class="color || (value / max > 0.8 ? 'bg-red-500' : 'bg-brand-500')"
        :style="{ width: Math.min(value / max * 100, 100) + '%' }"
      />
    </div>
  </div>
</template>
