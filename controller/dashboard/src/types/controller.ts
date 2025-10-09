export interface WorkerData {
  id: number
  address: string
  capacity: number
  current: number
  last_visit: string  // '%Y-%m-%d %H:%M:%S' at UTC+8
  status: string      // 'ALIVE' | 'COMA' | 'DEAD'
  stale?: boolean
  locked: boolean
}

export type TaskIndex = string | number

export interface TaskData {
  indices: TaskIndex[]
  workers: Record<number, WorkerData>
}

export interface SessionData {
  name: string
  index: TaskIndex
  custom_task?: any
  start: string        // '%Y-%m-%d %H:%M:%S' at UTC+8
  last_update: string  // '%Y-%m-%d %H:%M:%S' at UTC+8
  worker_id: number
  locked: boolean
  cancelling: boolean
}

export type ListWorkersResponse = Record<string, TaskData>

export type ListSessionsResponse = Record<number, SessionData>

export interface GetVersionResponse {
  variant: string
  version: number
}
