// Created by Codex - Section 2

import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, DestroyRef, ElementRef, ViewChild, effect, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';

import { ChatRequest, Message, PipelineResponse } from '../../core/models/chat.models';
import { ChatService } from '../../core/services/chat.service';
import { SessionService } from '../../core/services/session.service';
import { EmptyStateComponent } from '../../shared/components/empty-state/empty-state.component';
import { LoadingIndicatorComponent } from '../../shared/components/loading-indicator/loading-indicator.component';
import { BranchComposerComponent } from './branch-composer/branch-composer.component';
import { MessageBubbleComponent } from './message-bubble/message-bubble.component';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [
    FormsModule,
    DecimalPipe,
    EmptyStateComponent,
    LoadingIndicatorComponent,
    BranchComposerComponent,
    MessageBubbleComponent,
  ],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ChatComponent {
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly chatService = inject(ChatService);
  protected readonly sessionService = inject(SessionService);
  private readonly destroyRef = inject(DestroyRef);

  @ViewChild('scrollAnchor') private scrollAnchor?: ElementRef<HTMLDivElement>;

  protected readonly messages = signal<Message[]>([]);
  protected readonly draftQuery = signal('');
  protected readonly editingMessageIndex = signal<number | null>(null);
  protected readonly isStreaming = signal(false);
  protected readonly waitingForFirstChunk = signal(false);
  protected readonly branchBusy = signal(false);

  constructor() {
    effect(
      () => {
        this.messages.set([...this.sessionService.messages()]);
        queueMicrotask(() => this.scrollToBottom());
      },
      { allowSignalWrites: true },
    );

    this.route.paramMap.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((params) => {
      void this.loadRoute(params.get('sessionId'), params.get('branchId'));
    });
  }

  protected async submitQuery(): Promise<void> {
    const query = this.draftQuery().trim();
    if (!query || this.isStreaming()) {
      return;
    }
    this.draftQuery.set('');
    await this.executeAssistantTurn(query, true);
  }

  protected startEditing(index: number): void {
    this.editingMessageIndex.set(index);
  }

  protected cancelEdit(): void {
    this.editingMessageIndex.set(null);
  }

  protected async createBranch(query: string): Promise<void> {
    const chatId = this.sessionService.activeSessionId();
    const parentBranchId = this.sessionService.activeBranchId();
    const messageIndex = this.editingMessageIndex();
    const editedQuery = query.trim();
    if (!chatId || messageIndex === null || !editedQuery) {
      return;
    }

    this.branchBusy.set(true);
    try {
      const branchId = await this.sessionService.createBranch(chatId, {
        parent_branch_id: parentBranchId,
        fork_message_index: messageIndex,
        edited_query: editedQuery,
      });
      this.editingMessageIndex.set(null);
      await this.router.navigate(['/chat', chatId, 'branch', branchId]);
      await this.executeAssistantTurn(editedQuery, false);
    } finally {
      this.branchBusy.set(false);
    }
  }

  protected isDimmed(index: number): boolean {
    const editingIndex = this.editingMessageIndex();
    return editingIndex !== null && index > editingIndex;
  }

  protected loadingQuery(index: number): string {
    const previous = index > 0 ? this.messages()[index - 1] : undefined;
    return this.draftQuery() || previous?.content || '';
  }

  private async loadRoute(sessionId: string | null, branchId: string | null): Promise<void> {
    await this.sessionService.bootstrapRoute(sessionId, branchId, this.router);
    this.editingMessageIndex.set(null);
  }

  private async executeAssistantTurn(query: string, appendUser: boolean): Promise<void> {
    const sessionId = this.sessionService.activeSessionId();
    const branchId = this.sessionService.activeBranchId();
    if (!sessionId || !branchId) {
      return;
    }

    const baseMessages = [...this.messages()];
    const request = this.buildRequest(query, sessionId, branchId, baseMessages);
    const nextMessages = appendUser
      ? [
          ...baseMessages,
          { role: 'user', content: query, created_at: new Date().toISOString() },
          { role: 'assistant', content: '', created_at: new Date().toISOString(), streaming: true },
        ]
      : [
          ...baseMessages,
          { role: 'assistant', content: '', created_at: new Date().toISOString(), streaming: true },
        ];

    this.messages.set(nextMessages);
    this.isStreaming.set(true);
    this.waitingForFirstChunk.set(true);

    await new Promise<void>((resolve) => {
      let streamedAnswer = '';
      let finished = false;

      this.chatService.streamChat(request).subscribe({
        next: (event) => {
          if (event.type === 'chunk') {
            streamedAnswer += event.text;
            this.waitingForFirstChunk.set(false);
            this.replaceLastAssistant({ content: streamedAnswer, streaming: true });
            return;
          }
          if (event.type === 'done') {
            finished = true;
            this.finalizeResponse(
              event.payload,
              event.payload.answer || event.payload.message || streamedAnswer || '',
            );
            resolve();
            return;
          }
          if (event.type === 'error') {
            void this.fallbackInvoke(request, streamedAnswer).then(resolve);
          }
        },
        error: () => {
          if (!finished) {
            void this.fallbackInvoke(request, streamedAnswer).then(resolve);
          }
        },
        complete: () => {
          if (!finished && !this.isStreaming()) {
            resolve();
          }
        },
      });
    });
  }

  private async fallbackInvoke(request: ChatRequest, streamedAnswer: string): Promise<void> {
    this.waitingForFirstChunk.set(false);
    const payload = await new Promise<PipelineResponse>((resolve) => {
      this.chatService.invokeChat(request).subscribe({
        next: (response) => resolve(response),
        error: () =>
          resolve({
            status: 'error',
            answer: streamedAnswer || 'Unable to complete the request.',
            timings: {},
          }),
      });
    });
    this.finalizeResponse(payload, payload.answer || payload.message || streamedAnswer);
  }

  private finalizeResponse(payload: PipelineResponse, answerText: string): void {
    this.waitingForFirstChunk.set(false);
    this.isStreaming.set(false);
    this.replaceLastAssistant(this.buildAssistantMessage(payload, answerText));
    this.sessionService.setLastResponseMs(payload.timings?.['total_ms'] ?? null);
    const sessionId = this.sessionService.activeSessionId();
    const branchId = this.sessionService.activeBranchId();
    if (sessionId && branchId) {
      void this.sessionService.loadConversation(sessionId, branchId);
    }
  }

  private buildRequest(
    query: string,
    sessionId: string,
    branchId: string,
    baseMessages: Message[],
  ): ChatRequest {
    const history =
      baseMessages.at(-1)?.role === 'user' && baseMessages.at(-1)?.content === query
        ? baseMessages.slice(0, -1)
        : baseMessages;

    return {
      query,
      session_id: sessionId,
      branch_id: branchId,
      top_n: this.sessionService.topN(),
      agent_mode: false,
      follow_up_mode: this.sessionService.followUpMode(),
      chat_messages: history.map((message) => ({
        role: message.role,
        content: message.content,
      })),
      show_papers: this.sessionService.showPapers(),
      conversation_summary: this.sessionService.buildConversationSummary(history),
      compute_device: this.sessionService.computeDevice(),
    };
  }

  private buildAssistantMessage(payload: PipelineResponse, answerText: string): Message {
    return {
      role: 'assistant',
      content: answerText,
      created_at: new Date().toISOString(),
      status: payload.status,
      sources: payload.sources || [],
      retrieved_contexts: payload.retrieved_contexts || [],
      validation_issues: payload.validation_issues || [],
      invalid_citations: payload.invalid_citations || [],
      timings: payload.timings || {},
      answer_cache_hit: payload.answer_cache_hit,
      evidence_quality: payload.evidence_quality,
      streaming: false,
    };
  }

  private replaceLastAssistant(messagePatch: Partial<Message>): void {
    const next = [...this.messages()];
    for (let index = next.length - 1; index >= 0; index -= 1) {
      if (next[index].role === 'assistant') {
        next[index] = { ...next[index], ...messagePatch };
        this.messages.set(next);
        return;
      }
    }
  }

  private scrollToBottom(): void {
    this.scrollAnchor?.nativeElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }
}
