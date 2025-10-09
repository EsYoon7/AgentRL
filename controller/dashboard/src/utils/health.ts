import type { WorkerData } from '../types/controller.ts'

export enum WorkerHealth {
  ALIVE = 'ALIVE',
  COMA = 'COMA',
  DEAD = 'DEAD'
}

export const getHealth = (worker: WorkerData): WorkerHealth => {
  if (worker.stale) {
    return WorkerHealth.DEAD
  }

  if (worker.status.includes(WorkerHealth.ALIVE)) {
    return WorkerHealth.ALIVE
  }

  if (worker.status.includes(WorkerHealth.COMA)) {
    return WorkerHealth.COMA
  }

  return WorkerHealth.DEAD
}
