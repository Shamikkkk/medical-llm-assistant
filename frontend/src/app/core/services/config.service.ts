// Created by Codex - Section 2

import { Injectable, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class ConfigService {
  private readonly http = inject(HttpClient);

  readonly config = signal<Record<string, unknown> | null>(null);

  load(): Observable<Record<string, unknown>> {
    return this.http.get<Record<string, unknown>>('/api/config').pipe(
      tap((config) => this.config.set(config)),
    );
  }
}
