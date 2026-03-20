import { ref } from 'vue'

const BASE = '/console/api'

export function useApi<T>(endpoint: string) {
  const data = ref<T | null>(null)
  const loading = ref(false)
  const error = ref('')

  async function fetch() {
    loading.value = true
    error.value = ''
    try {
      const res = await globalThis.fetch(`${BASE}${endpoint}`)
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      data.value = await res.json()
    } catch (e: any) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  return { data, loading, error, fetch }
}

export async function post(endpoint: string, body: any) {
  const res = await globalThis.fetch(`${BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return res.json()
}
