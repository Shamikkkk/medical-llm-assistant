// Created by Codex - Section 2

import { HttpClient } from '@angular/common/http';
import { Injectable, computed, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { firstValueFrom } from 'rxjs';

import { Branch, BranchCreateRequest, ChatSession, Message } from '../models/chat.models';

@Injectable({ providedIn: 'root' })
export class SessionService {
  private readonly http = inject(HttpClient);

  readonly sessions = signal<ChatSession[]>([]);
  readonly branches = signal<Branch[]>([]);
  readonly messages = signal<Message[]>([]);
  readonly activeSessionId = signal<string | null>(null);
  readonly activeBranchId = signal<string>('main');
  readonly sidebarOpen = signal<boolean>(true);
  readonly topN = signal<number>(this.readNumber('topN', 10));
  readonly followUpMode = signal<boolean>(this.readBoolean('followUpMode', true));
  readonly showPapers = signal<boolean>(this.readBoolean('showPapers', true));
  readonly computeDevice = signal<string>(this.readString('computeDevice', 'auto'));
  readonly lastResponseMs = signal<number | null>(null);

  readonly activeSession = computed(() =>
    this.sessions().find((session) => session.chat_id === this.activeSessionId()) ?? null,
  );
  readonly activeBranch = computed(() =>
    this.branches().find((branch) => branch.branch_id === this.activeBranchId()) ?? null,
  );

  async bootstrapRoute(
    sessionId: string | null,
    branchId: string | null,
    router: Router,
  ): Promise<void> {
    await this.refreshSessions();
    if (!sessionId) {
      if (!this.sessions().length) {
        const created = await this.createSession();
        await router.navigate(['/chat', created.chat_id, 'branch', created.branch_id]);
        return;
      }
      const target = this.sessions()[0];
      await router.navigate(['/chat', target.chat_id, 'branch', branchId || 'main']);
      return;
    }

    await this.loadConversation(sessionId, branchId || 'main');
  }

  async refreshSessions(): Promise<void> {
    const sessions = await firstValueFrom(this.http.get<ChatSession[]>('/api/sessions'));
    this.sessions.set(sessions);
  }

  async createSession(): Promise<{ chat_id: string; branch_id: string }> {
    const created = await firstValueFrom(
      this.http.post<{ chat_id: string; branch_id: string }>('/api/sessions', {}),
    );
    await this.refreshSessions();
    return created;
  }

  async deleteSession(chatId: string): Promise<void> {
    await firstValueFrom(this.http.delete(`/api/sessions/${chatId}`));
    await this.refreshSessions();
    if (this.activeSessionId() === chatId) {
      this.activeSessionId.set(null);
      this.activeBranchId.set('main');
      this.branches.set([]);
      this.messages.set([]);
    }
  }

  async loadConversation(chatId: string, branchId: string): Promise<void> {
    const [branches, messages] = await Promise.all([
      firstValueFrom(this.http.get<Branch[]>(`/api/sessions/${chatId}/branches`)),
      firstValueFrom(
        this.http.get<Message[]>(`/api/sessions/${chatId}/branches/${branchId}/messages`),
      ),
    ]);
    this.activeSessionId.set(chatId);
    this.activeBranchId.set(branchId);
    this.branches.set(branches);
    this.messages.set(messages);
    await this.refreshSessions();
  }

  async createBranch(chatId: string, request: BranchCreateRequest): Promise<string> {
    const created = await firstValueFrom(
      this.http.post<{ branch_id: string }>(`/api/sessions/${chatId}/branches`, request),
    );
    await this.loadConversation(chatId, created.branch_id);
    return created.branch_id;
  }

  setMessages(messages: Message[]): void {
    this.messages.set(messages);
  }

  setTopN(value: number): void {
    this.topN.set(value);
    this.writeSetting('topN', String(value));
  }

  setFollowUpMode(value: boolean): void {
    this.followUpMode.set(value);
    this.writeSetting('followUpMode', String(value));
  }

  setShowPapers(value: boolean): void {
    this.showPapers.set(value);
    this.writeSetting('showPapers', String(value));
  }

  setComputeDevice(value: string): void {
    this.computeDevice.set(value);
    this.writeSetting('computeDevice', value);
  }

  setLastResponseMs(value: number | null): void {
    this.lastResponseMs.set(value);
  }

  toggleSidebar(): void {
    this.sidebarOpen.update((value) => !value);
  }

  buildConversationSummary(messages: Message[] = this.messages()): string {
    return messages
      .slice(-4)
      .map((message) => `${message.role}: ${message.content}`)
      .join('\n')
      .slice(0, 600);
  }

  exportCurrentBranchMarkdown(): string {
    const branch = this.activeBranch();
    const session = this.activeSession();
    const lines = [
      `# ${session?.title || 'Medical LLM Assistant Chat'}`,
      '',
      `- Branch: \`${branch?.branch_id || 'main'}\``,
      `- Branch title: ${branch?.title || 'Conversation branch'}`,
    ];
    if (branch?.parent_branch_id) {
      lines.push(`- Parent branch: \`${branch.parent_branch_id}\``);
    }
    lines.push('');
    for (const message of this.messages()) {
      const role = message.role === 'assistant' ? 'Assistant' : 'User';
      if (!message.content.trim()) {
        continue;
      }
      lines.push(`## ${role}`, '', message.content.trim(), '');
    }
    return `${lines.join('\n').trim()}\n`;
  }

  exportCurrentBranchJson(): string {
    return JSON.stringify(
      {
        chat_id: this.activeSessionId(),
        chat_title: this.activeSession()?.title || '',
        branch: this.activeBranch(),
        messages: this.messages(),
      },
      null,
      2,
    );
  }

  private readBoolean(key: string, fallback: boolean): boolean {
    if (typeof window === 'undefined') {
      return fallback;
    }
    const value = window.localStorage.getItem(key);
    return value === null ? fallback : value === 'true';
  }

  private readNumber(key: string, fallback: number): number {
    if (typeof window === 'undefined') {
      return fallback;
    }
    const rawValue = window.localStorage.getItem(key);
    const parsed = Number(rawValue);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
  }

  private readString(key: string, fallback: string): string {
    if (typeof window === 'undefined') {
      return fallback;
    }
    return window.localStorage.getItem(key) || fallback;
  }

  private writeSetting(key: string, value: string): void {
    if (typeof window === 'undefined') {
      return;
    }
    window.localStorage.setItem(key, value);
  }
}
