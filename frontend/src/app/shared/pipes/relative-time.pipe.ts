// Created by Codex - Section 2

import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'relativeTime',
  standalone: true,
})
export class RelativeTimePipe implements PipeTransform {
  transform(value: string | null | undefined): string {
    if (!value) {
      return 'Just now';
    }

    const timestamp = new Date(value).getTime();
    if (Number.isNaN(timestamp)) {
      return 'Just now';
    }

    const deltaSeconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
    if (deltaSeconds < 60) {
      return 'Just now';
    }
    if (deltaSeconds < 3600) {
      return `${Math.floor(deltaSeconds / 60)}m ago`;
    }
    if (deltaSeconds < 86400) {
      return `${Math.floor(deltaSeconds / 3600)}h ago`;
    }
    return `${Math.floor(deltaSeconds / 86400)}d ago`;
  }
}
