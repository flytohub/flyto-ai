<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useApi } from '../composables/useApi'

const executions = useApi<any[]>('/executions')
const filter = ref('')
const statusFilter = ref<'all' | 'ok' | 'error'>('all')

onMounted(() => executions.fetch())

const filtered = computed(() => {
  if (!executions.data.value) return []
  let items = executions.data.value
  if (statusFilter.value === 'ok') items = items.filter(e => e.ok)
  if (statusFilter.value === 'error') items = items.filter(e => !e.ok)
  if (filter.value) {
    const q = filter.value.toLowerCase()
    items = items.filter(e =>
      (e.user_message || '').toLowerCase().includes(q) ||
      (e.model || '').toLowerCase().includes(q)
    )
  }
  return items
})

const stats = computed(() => {
  const all = executions.data.value || []
  return {
    total: all.length,
    ok: all.filter(e => e.ok).length,
    error: all.filter(e => !e.ok).length,
    totalTokens: all.reduce((s, e) => s + (e.total_tokens || 0), 0),
  }
})
</script>

<template>
  <div class="p-6 max-w-6xl mx-auto space-y-4">
    <h2 class="text-xl font-semibold">Executions</h2>

    <!-- Stats bar -->
    <div class="flex gap-6 text-sm">
      <span class="text-gray-400">Total: <b class="text-gray-200">{{ stats.total }}</b></span>
      <span class="text-green-400">OK: <b>{{ stats.ok }}</b></span>
      <span class="text-red-400">Error: <b>{{ stats.error }}</b></span>
      <span class="text-gray-400">Tokens: <b class="text-gray-200">{{ stats.totalTokens.toLocaleString() }}</b></span>
    </div>

    <!-- Filters -->
    <div class="flex gap-3">
      <input
        v-model="filter"
        placeholder="Search messages..."
        class="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-500"
      />
      <div class="flex rounded-lg overflow-hidden border border-gray-700">
        <button
          v-for="s in (['all', 'ok', 'error'] as const)"
          :key="s"
          @click="statusFilter = s"
          class="px-3 py-1.5 text-xs capitalize"
          :class="statusFilter === s ? 'bg-brand-600 text-white' : 'bg-gray-900 text-gray-400 hover:bg-gray-800'"
        >{{ s }}</button>
      </div>
    </div>

    <!-- Table -->
    <div class="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
      <table class="w-full text-sm" v-if="filtered.length">
        <thead>
          <tr class="text-left text-xs text-gray-500 bg-gray-900/50">
            <th class="px-4 py-3">Time</th>
            <th class="px-4 py-3">Message</th>
            <th class="px-4 py-3">Model</th>
            <th class="px-4 py-3">Tools</th>
            <th class="px-4 py-3">Tokens</th>
            <th class="px-4 py-3">Duration</th>
            <th class="px-4 py-3">Status</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="(exec, i) in filtered"
            :key="i"
            class="border-t border-gray-800/50 hover:bg-gray-800/30"
          >
            <td class="px-4 py-2.5 text-gray-400 whitespace-nowrap">
              {{ new Date((exec.timestamp || 0) * 1000).toLocaleString() }}
            </td>
            <td class="px-4 py-2.5 max-w-xs truncate" :title="exec.user_message">
              {{ exec.user_message || '—' }}
            </td>
            <td class="px-4 py-2.5 text-gray-400 text-xs">{{ exec.model || '—' }}</td>
            <td class="px-4 py-2.5 text-gray-400">{{ exec.tool_calls_count || 0 }}</td>
            <td class="px-4 py-2.5 text-gray-400">{{ (exec.total_tokens || 0).toLocaleString() }}</td>
            <td class="px-4 py-2.5 text-gray-400">{{ exec.duration_ms || 0 }}ms</td>
            <td class="px-4 py-2.5">
              <span
                class="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full"
                :class="exec.ok ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'"
              >
                <span class="w-1.5 h-1.5 rounded-full" :class="exec.ok ? 'bg-green-500' : 'bg-red-500'" />
                {{ exec.ok ? 'OK' : exec.error || 'Error' }}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else class="p-8 text-center text-gray-500 text-sm">
        {{ executions.loading.value ? 'Loading...' : 'No executions found' }}
      </div>
    </div>
  </div>
</template>
