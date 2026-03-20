<script setup lang="ts">
import { ref, onMounted, onActivated } from 'vue'
import StatCard from '../components/StatCard.vue'
import ProgressBar from '../components/ProgressBar.vue'
import { useApi } from '../composables/useApi'

const overview = useApi<any>('/overview')
const cost = useApi<any>('/cost')
const budget = useApi<any>('/budget')
const blueprints = useApi<any>('/blueprints')
const ems = useApi<any>('/ems')
const executions = useApi<any>('/executions')

function refreshAll() {
  overview.fetch()
  cost.fetch()
  budget.fetch()
  blueprints.fetch()
  ems.fetch()
  executions.fetch()
}

// Refresh on mount AND every time user navigates to this page
onMounted(() => {
  refreshAll()
  // Auto-refresh every 10 seconds when visible
  const interval = setInterval(refreshAll, 10000)
  // cleanup not critical for SPA
})

function fmt(n: number) {
  return n < 0.01 ? '$' + n.toFixed(4) : '$' + n.toFixed(2)
}
</script>

<template>
  <div class="p-6 max-w-6xl mx-auto space-y-6">
    <h2 class="text-xl font-semibold">Dashboard</h2>

    <!-- Stat Cards -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4" v-if="overview.data.value">
      <StatCard
        label="Total Cost"
        :value="fmt(overview.data.value.cost?.session_total_usd || 0)"
        :sub="`${overview.data.value.cost?.call_count || 0} LLM calls`"
        color="text-green-400"
      />
      <StatCard
        label="License"
        :value="overview.data.value.license_tier?.toUpperCase()"
        :sub="overview.data.value.license_tier === 'free' ? 'Open source features' : 'All features enabled'"
        :color="overview.data.value.license_tier !== 'free' ? 'text-brand-400' : 'text-gray-400'"
      />
      <StatCard
        label="Modules"
        :value="overview.data.value.module_count"
        sub="Available automation modules"
      />
      <StatCard
        label="Provider"
        :value="overview.data.value.model || 'Not configured'"
        :sub="overview.data.value.provider"
      />
    </div>

    <div class="grid md:grid-cols-2 gap-6">
      <!-- Budget Usage -->
      <div class="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
        <h3 class="font-medium text-sm text-gray-300">Budget Usage</h3>
        <template v-if="budget.data.value?.pro_budget">
          <ProgressBar
            label="Cost"
            :value="budget.data.value.pro_budget.max_cost_usd - budget.data.value.pro_budget.remaining_cost"
            :max="budget.data.value.pro_budget.max_cost_usd"
          />
          <ProgressBar
            label="Tokens"
            :value="budget.data.value.pro_budget.max_tokens - budget.data.value.pro_budget.remaining_tokens"
            :max="budget.data.value.pro_budget.max_tokens"
          />
          <ProgressBar
            label="Tool Calls"
            :value="cost.data.value?.controller?.tool_calls || 0"
            :max="budget.data.value.pro_budget.max_tool_calls"
          />
        </template>
        <p v-else class="text-sm text-gray-500">No budget configured</p>
      </div>

      <!-- Blueprints & EMS -->
      <div class="space-y-4">
        <div class="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 class="font-medium text-sm text-gray-300 mb-3">Blueprints</h3>
          <div class="flex items-baseline gap-3">
            <span class="text-3xl font-bold text-brand-400">
              {{ blueprints.data.value?.total || 0 }}
            </span>
            <span class="text-sm text-gray-500">learned patterns</span>
          </div>
          <div v-if="blueprints.data.value?.blueprints?.length" class="mt-3 space-y-1">
            <div
              v-for="bp in blueprints.data.value.blueprints.slice(0, 3)"
              :key="bp.id"
              class="text-xs text-gray-400 flex justify-between"
            >
              <span class="truncate mr-2">{{ bp.query }}</span>
              <span class="text-gray-600">score {{ bp.score }}</span>
            </div>
          </div>
        </div>

        <div class="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 class="font-medium text-sm text-gray-300 mb-3">Error Memory (EMS)</h3>
          <template v-if="ems.data.value?.available">
            <div class="flex gap-6 text-sm">
              <div>
                <span class="text-2xl font-bold text-amber-400">{{ ems.data.value.errors_recorded }}</span>
                <span class="text-gray-500 ml-1">errors</span>
              </div>
              <div>
                <span class="text-2xl font-bold text-green-400">{{ ems.data.value.lessons_learned }}</span>
                <span class="text-gray-500 ml-1">fixes learned</span>
              </div>
            </div>
          </template>
          <p v-else class="text-sm text-gray-500">
            {{ ems.data.value?.reason || 'Requires Pro license' }}
          </p>
        </div>
      </div>
    </div>

    <!-- Recent Executions -->
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <h3 class="font-medium text-sm text-gray-300 mb-3">Recent Executions</h3>
      <div v-if="executions.data.value?.length" class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left text-xs text-gray-500 border-b border-gray-800">
              <th class="pb-2 pr-4">Time</th>
              <th class="pb-2 pr-4">Mode</th>
              <th class="pb-2 pr-4">Tools</th>
              <th class="pb-2 pr-4">Tokens</th>
              <th class="pb-2 pr-4">Duration</th>
              <th class="pb-2">Status</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(exec, i) in executions.data.value.slice(0, 10)"
              :key="i"
              class="border-b border-gray-800/50"
            >
              <td class="py-2 pr-4 text-gray-400">
                {{ new Date(exec.timestamp * 1000).toLocaleTimeString() }}
              </td>
              <td class="py-2 pr-4">{{ exec.mode || 'execute' }}</td>
              <td class="py-2 pr-4 text-gray-400">{{ exec.tool_calls_count || 0 }}</td>
              <td class="py-2 pr-4 text-gray-400">{{ (exec.total_tokens || 0).toLocaleString() }}</td>
              <td class="py-2 pr-4 text-gray-400">{{ exec.duration_ms || 0 }}ms</td>
              <td class="py-2">
                <span
                  class="inline-block w-2 h-2 rounded-full mr-1"
                  :class="exec.ok ? 'bg-green-500' : 'bg-red-500'"
                />
                {{ exec.ok ? 'OK' : 'Error' }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-else class="text-sm text-gray-500">No executions yet</p>
    </div>
  </div>
</template>
