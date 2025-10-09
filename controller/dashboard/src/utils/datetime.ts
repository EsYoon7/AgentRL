import { DateTime } from 'luxon'

export const formatTimestamp = (input: string): Date => {
  return DateTime.fromFormat(input, 'yyyy-MM-dd HH:mm:ss', {
    zone: 'UTC+8'
  }).toJSDate()
}
