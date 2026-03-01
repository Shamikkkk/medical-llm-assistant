// Created by Codex - Section 2

import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map } from 'rxjs';

import { ChatInvokeEnvelope, ChatRequest, PipelineResponse, StreamEvent } from '../models/chat.models';

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly http = inject(HttpClient);

  invokeChat(request: ChatRequest): Observable<PipelineResponse> {
    return this.http.post<ChatInvokeEnvelope>('/api/chat/invoke', request).pipe(
      map((response) => response.payload),
    );
  }

  streamChat(request: ChatRequest): Observable<StreamEvent> {
    return new Observable<StreamEvent>((observer) => {
      const controller = new AbortController();

      void (async () => {
        try {
          const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
            signal: controller.signal,
          });

          if (!response.ok || !response.body) {
            throw new Error(`Streaming request failed with status ${response.status}`);
          }

          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              break;
            }
            buffer += decoder.decode(value, { stream: true });
            buffer = this.flushBuffer(buffer, observer);
          }

          if (buffer.trim()) {
            this.flushBuffer(`${buffer}\n\n`, observer);
          }
          observer.complete();
        } catch (error) {
          observer.error(error);
        }
      })();

      return () => controller.abort();
    });
  }

  private flushBuffer(
    buffer: string,
    observer: { next: (value: StreamEvent) => void },
  ): string {
    const blocks = buffer.split('\n\n');
    const remainder = blocks.pop() ?? '';
    for (const block of blocks) {
      const line = block
        .split('\n')
        .find((candidate) => candidate.startsWith('data: '));
      if (!line) {
        continue;
      }
      observer.next(JSON.parse(line.slice(6)) as StreamEvent);
    }
    return remainder;
  }
}
